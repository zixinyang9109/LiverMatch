import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
import torch
from lib.util import load_config
from easydict import EasyDict as edict
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import collate_fn_descriptor, calibrate_neighbors
from lib.visualization import viz_coarse_nn_correspondence_mayavi, compare_pcd
from scipy.spatial.transform import Rotation
import pyvista as pv
from tqdm import tqdm


def cal_error(gt, pred, print_error=False):
    diff = np.linalg.norm(gt - pred, axis=1)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    diff_max = np.max(diff)

    diff = pred - gt
    RE = np.sqrt(np.sum(diff * diff) / len(diff))
    if print_error:
        print("mean error: %0.2f, max: %0.2f, std: %0.2f, RE: %0.2f" % (diff_mean, diff_max, diff_std, RE))
    return diff_mean, diff_max, diff_std, RE


def vis_pc_marker(src_vs=None, src_marker=None, tgt_vs=None, tgt_marker=None):
    plotter = pv.Plotter()
    plotter.set_background('white')

    if src_vs is not None:
        plotter.add_points(src_vs, color=[0, 150, 255], point_size=3, render_points_as_spheres=True, opacity=0.4)

    if tgt_vs is not None:
        plotter.add_points(tgt_vs, color=[254, 92, 92], point_size=3, render_points_as_spheres=True, opacity=0.4)

    if src_marker is not None:
        plotter.add_points(src_marker, color=[0, 0, 255], point_size=12, render_points_as_spheres=True,
                           opacity=0.9)  # ,render_points_as_spheres=True, point_size=5

    if tgt_marker is not None:  # 255
        plotter.add_points(tgt_marker, color=[255, 0, 0], point_size=12, render_points_as_spheres=True, opacity=0.9)

    plotter.show()


def vis_mesh_pc(src_vs=None, src_marker=None, tgt_vs=None, tgt_marker=None):
    plotter = pv.Plotter()
    plotter.set_background('white')

    if src_vs is not None:
        plotter.add_points(src_vs, color='blue', render_points_as_spheres=True, opacity=0.5)

    if tgt_vs is not None:
        plotter.add_points(tgt_vs, color='red', render_points_as_spheres=True, opacity=0.5)

    if src_marker is not None:
        plotter.add_points(src_marker, color='lightblue', point_size=15)  # ,render_points_as_spheres=True, point_size=5

    if tgt_marker is not None:
        plotter.add_points(tgt_marker, color='darkred', point_size=15)

    plotter.show()


def ply2np(file, scale=100.0):
    pcd = np.asarray(o3d.io.read_point_cloud(file).points)
    pcd = pcd / scale
    return pcd


def ply2np_vox(file, voxel_size=0.005, scale=100.0):
    pcd = o3d.io.read_point_cloud(file)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_pts = np.asarray(downpcd.points)
    pcd_pts = pcd_pts / scale
    return pcd_pts


def center(points):
    centroid = np.median(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid
    return points_centered, centroid


class LiverDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, config, test_root_path, test_list_file_path, voxel_size, sigma, rot):
        super(LiverDemo, self).__init__()
        self.config = config
        self.test_root = test_root_path
        self.test_list = np.load(test_list_file_path)['test']
        self.sigma = sigma
        self.rot = rot
        self.voxel_size = voxel_size

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, item):

        src_pcd, tgt_pcd, src_marker, tgt_marker, s_c, t_c, m,_,_ = self.get_data_np(item)

        # fake the ground truth information
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3, 1)).astype(np.float32)
        correspondences = torch.ones(1, 2).long()

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

    def get_data_np(self, index):
        src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt = self.load_test_from_npz(self.test_root + self.test_list[index],
                                                                             sigma=self.sigma, rot=self.rot)

        # src_marker = src_raw
        # tgt_marker = src_raw + flow

        src_xyz, m = self.norm_vox(src_raw)
        tgt_xyz, _ = self.norm_vox(tgt_raw)

        s_c = np.mean(src_xyz[:, :3], axis=0)
        t_c = np.mean(tgt_xyz[:, :3], axis=0)

        src_xyz = (src_xyz - s_c) / m
        tgt_xyz = (tgt_xyz - t_c) / m


        src_markers = (src_markers - s_c) / m
        tgt_markers = (tgt_markers - t_c) / m

        return src_xyz, tgt_xyz, src_markers, tgt_markers, s_c, t_c, m, R_gt, t_gt

    def load_test_from_npz(self, file, sigma=None, rot=False):
        with np.load(file, allow_pickle=True) as entry:
            src_vs = entry['src_pcd']
            tgt_vs = entry['tgt_pcd']
            # flow = entry['flow']

            src_markers = entry['src_vol']
            tgt_markers = entry['tgt_vol']

            R_gt = entry['R_gt']
            t_gt = entry['t_gt']

            # faces = entry['src_f']
            # edges = entry['src_edges']
            # tgt_f = entry['tgt_f']


            if sigma is not None:
                noise = entry[str(sigma)]
                tgt_vs = tgt_vs + noise

            if rot:
               # rot_src = entry['rot_src']
                rot_tgt = entry['rot_tgt']
                #src_vs = (np.matmul(rot_src, src_vs.T)).T
                tgt_vs = (np.matmul(rot_tgt, tgt_vs.T)).T

                #src_markers = (np.matmul(rot_src, src_markers.T)).T
                tgt_markers = (np.matmul(rot_tgt, tgt_markers.T)).T

                R_gt = np.matmul(R_gt, rot_tgt)
                t_gt = np.matmul(rot_tgt, t_gt)

        return src_vs, tgt_vs, src_markers, tgt_markers, R_gt, t_gt

    # def load_data(self, index):
    #
    #     src_raw, tgt_raw, src_markers, tgt_markers = self.load_test_from_npz(self.test_root + self.test_list[index],
    #                                                                          sigma=self.sigma, rot=self.rot)
    #
    #     # src_marker = src_raw
    #     # tgt_marker = src_raw + flow
    #
    #     src_xyz, m = self.norm_vox(src_raw)
    #     tgt_xyz, _ = self.norm_vox(tgt_raw)
    #     src_xyz = (src_xyz - np.mean(src_xyz[:, :3], axis=0)) / m
    #     tgt_xyz = (tgt_xyz - np.mean(tgt_xyz[:, :3], axis=0)) / m
    #
    #     src_markers = (src_markers - np.mean(src_markers[:, :3], axis=0)) / m
    #     tgt_markers = (tgt_markers - np.mean(tgt_markers[:, :3], axis=0)) / m
    #
    #     return src_xyz, tgt_xyz, src_markers, tgt_markers, m

    def pc_normalize(self, pc, centroid=None, m=None):
        if centroid is None:
            centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        if m is None:
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc, centroid, m

    def ply2np_vox(self, xyz, voxel_size=2, scale=1.0):
        if type(xyz) is str:
            pcd = o3d.io.read_point_cloud(xyz)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_pts = np.asarray(downpcd.points)
        pcd_pts = pcd_pts / scale
        return pcd_pts

    def norm_vox(self, xyz):
        pc, centroid, m = self.pc_normalize(xyz)
        pc_vox = self.ply2np_vox(pc, voxel_size=self.voxel_size, scale=1.0) * m + centroid
        return pc_vox, m


def eval_matrics(tgt_pcd_full, tgt_pcd, pred_corr, th=0.02):
    diff = tgt_pcd_full[pred_corr[:, 0], :] - tgt_pcd[pred_corr[:, 1], :]

    dist = np.linalg.norm(diff, axis=1)
    # print(torch.median(dist))
    # print(torch.max(dist))

    len_pred = len(pred_corr)
    len_gt = len(tgt_pcd)

    inliers = (dist < th) * 1.0
    # print("old inliers:", inliers.sum().float())
    # print("n_match", len(inliers))

    # inliers = torch.sum((diff) ** 2, dim=1) < th ** 2
    # print("new inliers:", inliers.sum().float())

    inlier_ratio = inliers.mean()

    inliers_scores = np.sum(inliers) / len_gt

    rmse_dist = np.sqrt(np.sum(dist ** 2) / len_pred)

    return rmse_dist, inlier_ratio, inliers_scores, inliers


def to_array(tensor):
    """
    Conver tensor to array
    """
    if (not isinstance(tensor, np.ndarray)):
        if (tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def save_ply(file):
    """
    Convert numpy to open3d PointCloud
    xyz:       [N, 3]
    """
    import pyvista as pv
    xyz = pv.read(file).points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(file.replace(".vtk", ".ply"), pcd)
    return pcd, file.replace(".vtk", ".ply")


def to_o3d_pcd(xyz):
    """
    Convert numpy to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def eva_regist_RANASC(src_pcd, tgt_pcd, corrs, src_marker=None, distance_threshold=0.05, ransac_n=4, debug=False):
    src_pcd_o3d = to_o3d_pcd(src_pcd)
    tgt_pcd_o3d = to_o3d_pcd(tgt_pcd)
    corrs = to_array(corrs).astype(np.int32)
    corrs_o3d = np.array([corrs[:, 0], corrs[:, 1]]).T

    corrs_o3d = o3d.utility.Vector2iVector(corrs_o3d)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=src_pcd_o3d, target=tgt_pcd_o3d, corres=corrs_o3d,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,

        # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )  # criteria=o3d.pi

    tsfm = np.array(result_ransac.transformation)

    rot_ = tsfm[:3, :3]
    trans_ = tsfm[:3, 3:]

    src_warped = (np.matmul(rot_, src_pcd.T) + trans_).T

    if debug:
        # print("compare pred with gt")
        # compare_pcd(tgt_pcd_full_pred, tgt_pcd_full)
        # print("compare tgt with gt")
        # compare_pcd(tgt_pcd, tgt_pcd_full)
        print("compare pred with tgt")
        compare_pcd(src_warped, tgt_pcd)

    if src_marker is not None:

        src_marker_pred = (np.matmul(rot_, src_marker.T) + trans_).T
        return src_warped, src_marker_pred, rot_, trans_
    else:
        return src_warped, rot_, trans_

    # gc.collect()
    # src_pcd_o3d.clear()
    # tgt_pcd_o3d.clear()
    # del src_pcd_o3d
    # del tgt_pcd_o3d


def batch_weighted_procrustes(X, Y, w, eps=0.0001):
    '''
    @param X: source frame [B, N,3]
    @param Y: target frame [B, N,3]
    @param w: weights [B, N,1]
    @param eps:
    @return:
    '''
    # https://ieeexplore.ieee.org/document/88573

    bsize = X.shape[0]
    device = X.device
    # w = w.cpu()
    W1 = torch.abs(w).sum(dim=1, keepdim=True)
    w_norm = w / (W1 + eps)
    mean_X = (w_norm * X).sum(dim=1, keepdim=True)
    mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
    Sxy = torch.matmul((Y - mean_Y).transpose(1, 2), w_norm * (X - mean_X))
    Sxy = Sxy.cpu().double()
    U, D, V = Sxy.svd()  # small SVD runs faster on cpu
    condition = D.max(dim=1)[0] / D.min(dim=1)[0]
    S = torch.eye(3)[None].repeat(bsize, 1, 1).double()
    UV_det = U.det() * V.det()
    S[:, 2:3, 2:3] = UV_det.view(-1, 1, 1)
    svT = torch.matmul(S, V.transpose(1, 2))
    R = torch.matmul(U, svT).to(device).to(torch.float64)
    t = mean_Y.transpose(1, 2) - torch.matmul(R, mean_X.transpose(1, 2))
    return R, t, condition


def eva_regist_SVD(src_pcd, tgt_pcd, matches, src_marker=None, debug=False):
    s_pcd = torch.from_numpy(src_pcd).unsqueeze(0)
    t_pcd = torch.from_numpy(tgt_pcd).unsqueeze(0)
    src_pcd_sampled = s_pcd[0, matches[:, 0], :].unsqueeze(0)
    tgt_pcd_sampled = t_pcd[0, matches[:, 1], :].unsqueeze(0)
    w = torch.ones([1, len(matches), 1]).to(src_pcd_sampled.device)
    # w = conf[:, matches[:, 0], matches[:, 1]].unsqueeze(2).detach().cpu()
    R, t, condition = batch_weighted_procrustes(src_pcd_sampled, tgt_pcd_sampled, w)
    rot_ = R.cpu().detach().numpy().squeeze(0)
    trans_ = t.cpu().detach().numpy().squeeze(0)

    src_warped = (np.matmul(rot_, src_pcd.T) + trans_).T

    if debug:
        # print("compare pred with gt")
        # compare_pcd(tgt_pcd_full_pred, tgt_pcd_full)
        # print("compare tgt with gt")
        # compare_pcd(tgt_pcd, tgt_pcd_full)
        print("compare pred with tgt")
        compare_pcd(src_warped, tgt_pcd)

    if src_marker is not None:

        src_marker_pred = (np.matmul(rot_, src_marker.T) + trans_).T
        return src_warped, src_marker_pred, rot_, trans_
    else:
        return src_warped, rot_, trans_


def eva_one(index, debug=False, use_SVD=True, voxel_size=0.04):
    src_pcd, tgt_pcd, src_marker, tgt_marker, s_c, t_c, scale, R_gt, t_gt = demo_set.get_data_np(index)

    # compare_pcd(src_gt/scale, tgt_raw/scale)

    if use_SVD:
        eva_regist = eva_regist_SVD
    else:
        eva_regist = eva_regist_RANASC

    if debug:
        vis_mesh_pc(src_pcd, src_marker)
        vis_mesh_pc(tgt_vs=tgt_pcd, tgt_marker=tgt_marker)
        vis_mesh_pc(src_vs=src_pcd, tgt_vs=tgt_pcd)

    # neighborhood_limits = [19, 23, 29, 34]
    #neighborhood_limits = [7, 20, 30, 38]
    neighborhood_limits = [6, 17, 28, 36]
    # loader, neighborhood_limits = get_dataloader(dataset=demo_set,
    #                                              batch_size=1,
    #                                              shuffle=False,
    #                                              num_workers=0, neighborhood_limits=None
    #                                              )
    # print(neighborhood_limits)
    list_data = demo_set.__getitem__(index)

    inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)

    with torch.no_grad():
        ##################################
        # load inputs to device.
        for k, v in inputs.items():
            if type(v) == list:
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)

    data = model(inputs)
    match_pred = data['match_pred'].detach().cpu()
    match_pred = match_pred[:, 1:]

    match_pred_scores = match_pred

    correspondences = np.zeros(np.shape(match_pred_scores))
    correspondences[:, 0] = np.arange(match_pred_scores.shape[0])
    correspondences[:, 1] = np.arange(match_pred_scores.shape[0])

    # src_pcd_sampled = src_pcd[match_pred_scores[:, 0], :]
    # tgt_pcd_sampled = tgt_pcd[match_pred_scores[:, 1], :]

    # src_warped, src_marker_pred, rot_, trans_ = eva_regist(src_pcd_sampled, tgt_pcd_sampled, correspondences,
    #                                                        src_marker=src_marker,
    #                                                        debug=debug)

    src_warped, src_marker_pred, rot_, trans_ = eva_regist(src_pcd, tgt_pcd, match_pred_scores, src_marker=src_marker,
                                                           debug=debug)
    if debug:
        scale_factor = 0.013 *4

        viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd + 1, match_pred_scores.T, f_src_pcd=None,
                                            f_tgt_pcd=None,
                                            scale_factor=scale_factor )

        vis_pc_marker(src_warped, src_marker_pred, tgt_pcd, tgt_marker)
        compare_pcd(src_warped, tgt_pcd, scale_factor)

    #cal_error(tgt_marker * scale, src_marker * scale, True)
    diff_mean, diff_max, diff_std, RE = cal_error(tgt_marker * scale, src_marker_pred * scale, True)

    # src_pcd_sampled = src_gt[match_pred_scores[:, 0], :]
    # tgt_pcd_sampled = tgt_raw[match_pred_scores[:, 1], :]

    src_gt = src_pcd * scale + s_c
    src_gt = (np.matmul(R_gt, src_gt.T) + t_gt).T
    tgt_raw = tgt_pcd * scale + t_c

    _, inlier_ratio, ms, _ = eval_matrics(src_gt, tgt_raw, match_pred_scores, th=scale*voxel_size)

    # viz_coarse_nn_correspondence_mayavi(src_gt/100, tgt_raw/100 +1, match_pred_scores.T, f_src_pcd=None,
    #                                     f_tgt_pcd=None,
    #                                     scale_factor=0.013 * 2)

    return inlier_ratio, ms, RE


if __name__ == '__main__':

    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/new_task3_004_002.yaml"
    pretrain_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth"

    test_root_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/Test/"
    test_list_file_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/list.npz"
    result_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Results/Rigid_results/LiverMatch/"
    rot = True
    voxel_size = 0.04
    # "/home/yzx/yzx/Deformable_Registration/LiverMatch/snapshot/liver_task3_grid_off/checkpoints/model_best_loss.pth"

    config = load_config(config_path)
    config = edict(config)
    config.architecture = architectures[config.model_name]
    config.device = torch.device('cuda:0')
    model = KPFCNN(config).to(config.device).eval()
    state = torch.load(pretrain_path)
    model.load_state_dict(state['state_dict'])

    sigma = None
    use_SVD = True
    if not use_SVD:
        result_path = result_path.replace("Rigid_results", "Rigid_results_ransac")
    #
    # demo_set = LiverDemo(config, test_root_path, test_list_file_path, voxel_size, sigma, rot)
    #
    # IR, MS, RE = eva_one(3506, debug=True, use_SVD=use_SVD) # 3506, 5742

    for sigma in [None, 1, 2, 3, 4, 5 ]: # None, 1, 2, 3, 4, 5 , 2, 4

        demo_set = LiverDemo(config, test_root_path, test_list_file_path, voxel_size, sigma, rot)

        RE_list = []
        IR_list = []
        MS_list = []

        for i in tqdm(range(len(demo_set))):
            _, _, RE = eva_one(i, debug=False, use_SVD=use_SVD)
            print("IR: ", IR*100, "MS: ", MS*100)
            RE_list.append(RE)
            # IR_list.append(IR*100)
            # MS_list.append(MS*100)
        np.save(result_path + str(sigma)+"004002", RE_list)
        # np.save(result_path + str(sigma) + "004002_IR", IR_list)
        # np.save(result_path + str(sigma) + "004002_MS", MS_list)














