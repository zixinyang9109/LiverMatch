import os.path

import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
import torch
from datasets.dataloader import get_dataloader
from lib.util import load_config
from easydict import EasyDict as edict
from configs.models import architectures
#from models.framework_cluster_pe import KPFCNN
#from models.framework_cluste_generic import KPFCNN
from models.framework_cluster import KPFCNN
#from models.framework_cl_wo_tf import KPFCNN
#from models.framework import KPFCNN
from datasets.dataloader import collate_fn_descriptor, calibrate_neighbors
from lib.visualization import viz_coarse_nn_correspondence_mayavi, compare_pcd
from scipy.spatial.transform import Rotation
import pyvista as pv
from tqdm import tqdm
from lib.utils import generate_node_clusters, index_select

from pointnet2_ops import pointnet2_utils

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                  2).contiguous()
    return sub_pc

def cal_error(gt, pred, print_error=False):
    diff = np.linalg.norm(gt - pred, axis=1)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    diff_max = np.max(diff)

    diff = pred - gt
    RE = np.sqrt(np.sum(diff * diff) / len(diff))
    if print_error:
        print("mean error: %0.2f, max: %0.2f, std: %0.2f, RE: %0.2f" % (diff_mean, diff_max, diff_std, RE))
    return diff_mean,diff_max, diff_std, RE

# def vis_pc_marker(src_vs=None, src_marker=None, tgt_vs=None, tgt_marker=None):
#
#     plotter = pv.Plotter()
#     plotter.set_background('white')
#
#     if src_vs is not None:
#         plotter.add_points(src_vs, color=[0, 150, 255], point_size=3, render_points_as_spheres=True, opacity=0.4)
#
#     if tgt_vs is not None:
#         plotter.add_points(tgt_vs, color=[254, 92, 92], point_size=3, render_points_as_spheres=True, opacity=0.4)
#
#     if src_marker is not None:
#         plotter.add_points(src_marker, color=[0, 0, 255], point_size=12, render_points_as_spheres=True, opacity=1) #,render_points_as_spheres=True, point_size=5
#
#     if tgt_marker is not None: # 255
#         plotter.add_points(tgt_marker, color=[255, 0, 0], point_size=12, render_points_as_spheres=True, opacity=0.9)
#
#     plotter.show()

def vis_mesh_pc(src_vs=None, src_marker=None, tgt_vs=None, tgt_marker=None):

    plotter = pv.Plotter()
    plotter.set_background('white')

    if src_vs is not None:
        plotter.add_points(src_vs, color='blue', render_points_as_spheres=True, opacity=0.5, point_size=15)

    if tgt_vs is not None:
        plotter.add_points(tgt_vs, color='red', render_points_as_spheres=True, opacity=0.5, point_size=15)

    if src_marker is not None:
        plotter.add_points(src_marker, color='lightblue', point_size=15) #,render_points_as_spheres=True, point_size=5

    if tgt_marker is not None:
        plotter.add_points(tgt_marker, color='darkred', point_size=15)

    plotter.show()
def ply2np(file, scale = 100.0):
    pcd = np.asarray(o3d.io.read_point_cloud(file).points)
    pcd = pcd/scale
    return pcd

def ply2np_vox(file, voxel_size=0.005, scale =100.0):
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

    # def get_data_np(self, index):
    #     src_raw, tgt_raw, src_markers, tgt_markers = self.load_test_from_npz(self.test_root + self.test_list[index],
    #                                                                          sigma=self.sigma, rot=self.rot)
    #
    #     # src_marker = src_raw
    #     # tgt_marker = src_raw + flow
    #
    #     src_xyz, m = self.norm_vox(src_raw)
    #     tgt_xyz, _ = self.norm_vox(tgt_raw)
    #
    #     s_c = np.mean(src_xyz[:, :3], axis=0)
    #     t_c = np.mean(tgt_xyz[:, :3], axis=0)
    #
    #     src_xyz = (src_xyz - s_c) / m
    #     tgt_xyz = (tgt_xyz - t_c) / m
    #
    #     src_markers = (src_markers - s_c) / m
    #     tgt_markers = (tgt_markers - t_c) / m
    #
    #     return src_xyz, tgt_xyz, src_markers, tgt_markers, s_c, t_c, m
    #
    # def load_test_from_npz(self, file, sigma=None, rot=False):
    #     with np.load(file, allow_pickle=True) as entry:
    #         src_vs = entry['src_pcd']
    #         tgt_vs = entry['tgt_pcd']
    #         # flow = entry['flow']
    #
    #         src_markers = entry['src_vol']
    #         tgt_markers = entry['tgt_vol']
    #
    #         #cal_error(src_markers, tgt_markers)
    #
    #         # faces = entry['src_f']
    #         # edges = entry['src_edges']
    #         # tgt_f = entry['tgt_f']
    #
    #         if sigma is not None:
    #             noise = entry[str(sigma)]
    #             tgt_vs = tgt_vs + noise
    #
    #         if rot:
    #             rot_src = entry['rot_src']
    #             rot_tgt = entry['rot_tgt']
    #             src_rot = (np.matmul(rot_src, src_vs.T)).T
    #             tgt_vs_rot = (np.matmul(rot_tgt, tgt_vs.T)).T
    #
    #             src_markers_rot = (np.matmul(rot_src, src_markers.T)).T
    #             tgt_markers_rot = (np.matmul(rot_tgt, tgt_markers.T)).T
    #
    #             return src_rot, tgt_vs_rot, src_markers_rot, tgt_markers_rot
    #
    #     return src_vs, tgt_vs, src_markers, tgt_markers

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

    def get_data_np(self, index):
        src_raw, tgt_raw, src_markers, tgt_markers, R_gt, t_gt = self.load_test_from_npz(
            self.test_root + self.test_list[index],
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
                # src_vs = (np.matmul(rot_src, src_vs.T)).T
                tgt_vs = (np.matmul(rot_tgt, tgt_vs.T)).T

                # src_markers = (np.matmul(rot_src, src_markers.T)).T
                tgt_markers = (np.matmul(rot_tgt, tgt_markers.T)).T

                R_gt = np.matmul(R_gt, rot_tgt)
                t_gt = np.matmul(rot_tgt, t_gt)

        return src_vs, tgt_vs, src_markers, tgt_markers, R_gt, t_gt

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

    inliers = (dist < th)*1.0
    # print("old inliers:", inliers.sum().float())
    # print("n_match", len(inliers))

    # inliers = torch.sum((diff) ** 2, dim=1) < th ** 2
    # print("new inliers:", inliers.sum().float())

    inlier_ratio = inliers.mean()

    inliers_scores = np.sum(inliers)/len_gt

    rmse_dist = np.sqrt(np.sum(dist ** 2) / len_pred)

    return rmse_dist, inlier_ratio, inliers_scores, inliers

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
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

def vis_pc_marker(src_vs=None, src_marker=None, tgt_vs=None, tgt_marker=None, point_opacity=1, marker_opacity=0.6, point_size=12, marker_size=10):

    plotter = pv.Plotter()
    plotter.set_background('white')

    if src_vs is not None:
        plotter.add_points(src_vs, color=[0, 150, 255], point_size=point_size, render_points_as_spheres=True, opacity=point_opacity)

    if tgt_vs is not None:
        plotter.add_points(tgt_vs, color=[254, 92, 92], point_size=point_size, render_points_as_spheres=True, opacity=point_opacity)

    if src_marker is not None:
        plotter.add_points(src_marker, color=[175, 175, 175], point_size=marker_size, render_points_as_spheres=True, opacity=marker_opacity) #,render_points_as_spheres=True, point_size=5

    if tgt_marker is not None: # 255
        plotter.add_points(tgt_marker, color=[175, 175, 175], point_size=marker_size, render_points_as_spheres=True, opacity=marker_opacity)

    plotter.show()


def eva_regist_RANASC(src_pcd, tgt_pcd, corrs, src_marker=None, distance_threshold=0.1, ransac_n=4, debug =False):
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
        #criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )  # criteria=o3d.pi

    tsfm = np.array(result_ransac.transformation)

    # src_pcd_o3d.clear()
    # tgt_pcd_o3d.clear()

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




def eva_one(index, debug=False, use_SVD=True, cluster=False, use_corr_cl=False, use_all_matches=True):

    src_pcd, tgt_pcd, src_marker, tgt_marker, s_c, t_c, scale,  R_gt, t_gt = demo_set.get_data_np(index)

    data_to_save = {'src_pcd': src_pcd, 'tgt_pcd':tgt_pcd,
                    'src_marker':src_marker, 'tgt_marker':tgt_marker,
                    's_c':s_c,
                    't_c':t_c,
                    'scale':scale,
                    'R_gt':R_gt,
                    't_gt':t_gt}
    #{'tensor_one': tensor_a, 'tensor_two': tensor_b}
    torch.save(data_to_save, '/media/yzx/yzx_store1/Task03_Liver/Train_Test/P2P_Demo/data.pt')


    if use_SVD:
        eva_regist = eva_regist_SVD
    else:
        eva_regist = eva_regist_RANASC


    if debug:

        # vis_mesh_pc(src_pcd, src_marker)
        # vis_mesh_pc(tgt_vs=tgt_pcd, tgt_marker=tgt_marker)
        #vis_mesh_pc(src_vs=src_pcd, tgt_vs=tgt_pcd)
        vis_pc_marker(src_vs=src_pcd, tgt_vs=tgt_pcd)
        vis_pc_marker(src_vs=src_marker, tgt_vs=tgt_marker)


    neighborhood_limits = [6, 17, 28, 36] #[8 22 32 38]
    # loader, neighborhood_limits = get_dataloader(dataset=demo_set,
    #                                              batch_size=1,
    #                                              shuffle=True,
    #                                              num_workers=0, neighborhood_limits=None
    #                                              )

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

    if use_corr_cl:
        if use_all_matches:
            match_pred = data['matches_all_unique'].detach().cpu() # FILTERED
        else:
            match_pred = data['cl_corr_g_unique'].detach().cpu()

    else:
        match_pred = data['match_pred'].detach().cpu()
        match_pred = match_pred[:, 1:]

    if debug:
        scale_factor = 0.013 * 4

        match_pred_g = data['match_pred'].detach().cpu()[:, 1:]
        match_pred_all = data['cl_corr_g_unique'].detach().cpu()
        match_pred_all_f = data['matches_all_unique'].detach().cpu()

        viz_coarse_nn_correspondence_mayavi(src_pcd + s_c / scale, tgt_pcd + t_c / scale + 1, match_pred_g.T,
                                            f_src_pcd=None,
                                            f_tgt_pcd=None,
                                            scale_factor=scale_factor)

        viz_coarse_nn_correspondence_mayavi(src_pcd + s_c / scale, tgt_pcd + t_c / scale + 1, match_pred_all.T,
                                            f_src_pcd=None,
                                            f_tgt_pcd=None,
                                            scale_factor=scale_factor)

        viz_coarse_nn_correspondence_mayavi(src_pcd + s_c / scale, tgt_pcd + t_c / scale + 1, match_pred_all_f.T,
                                            f_src_pcd=None,
                                            f_tgt_pcd=None,
                                            scale_factor=scale_factor)


    if debug:

        # # vis scores
        # scores_vis = data['scores_vis'].detach().cpu()
        # scores_vis_mask = scores_vis > 0.5
        # vis_src_pcd = src_pcd[scores_vis_mask]

        vis_src_pcd = data['vis_src_pcd'].detach().cpu().numpy()
        #vis_mesh_pc(src_pcd, vis_src_pcd)
        vis_pc_marker(vis_src_pcd, src_pcd)

        #cluster anchors
        num = 5
        fps_idx = pointnet2_utils.furthest_point_sample(torch.from_numpy(vis_src_pcd).unsqueeze(0).float().cuda(), num)
        fps_idx = fps_idx.squeeze(0).cpu().numpy()
        #vis_mesh_pc(src_pcd, vis_src_pcd[fps_idx])

        vis_pc_marker(vis_src_pcd[fps_idx], src_pcd, point_size=28)

        src_pcd_torch = torch.from_numpy(src_pcd)
        tgt_pcd_torch = torch.from_numpy(tgt_pcd)
        src_node_torch = torch.from_numpy(vis_src_pcd[fps_idx])

        # # anchors to cluster
        # if len(tgt_pcd) > len(src_pcd):
        #     tgt_node_xyz = tgt_node_xyz[:len(src_node_xyz)]
        #     tgt_node_feats = tgt_node_feats[:len(src_node_xyz)]

        src_indices_cl = generate_node_clusters(src_pcd_torch.double(), src_node_torch.double(), point_limit=len(tgt_pcd_torch))
        src_xyz_cl = index_select(src_pcd_torch, src_indices_cl, dim=0)
        cl_index = 0
        for cl_index in np.arange(len(fps_idx)):
            #vis_mesh_pc(src_pcd, src_xyz_cl[cl_index].numpy())
            vis_pc_marker(src_xyz_cl[cl_index].numpy(), src_pcd)




        # # vote
        # vote = data['vote_xyz'].detach().cpu().squeeze(0).numpy()
        # vote_centers = src_pcd + vote
        # vis_mesh_pc(src_pcd, vote_centers[scores_vis_mask])

    match_pred_scores = match_pred

    if cluster:
        rot_cl = data['R_cl']
        trans_cl = data['t_cl']

        best_cl_index = data['best_index_cp']
        print("best cluster: ", best_cl_index)

        # best_cl_index = data['best_index_il']
        # print("best cluster: ", best_cl_index)

        CP_res = data['CP_res']
        print("Global res ", CP_res[0] * scale, "Best res ", CP_res[best_cl_index] * scale)

        rot_ = rot_cl[best_cl_index].detach().cpu().numpy()
        trans_ = trans_cl[best_cl_index].detach().cpu().numpy()[:, None]

        src_marker_pred = (np.matmul(rot_, src_marker.T) + trans_).T
        #
        # src_warp = torch.matmul(src_pcd, rot_.T) + trans_.T
        # from lib.loss import pairwise_distance
        # dist_mat = pairwise_distance(tgt_pcd, src_warp)
        # cp_res = torch.mean(dist_mat.min(dim=1).values)
        # print("res", cp_res)

        if debug:
            if best_cl_index>0:
                best_cl_index = best_cl_index -1
            src_xyz_cl = data['src_xyz_cl'].detach().cpu().numpy()
              # .detach().cpu()
            src_xyz_cl_best = src_xyz_cl[best_cl_index]
            src_xyz_cl_best_pred = (np.matmul(rot_, src_xyz_cl_best.T) + trans_).T
            src_pcd_pred = (np.matmul(rot_, src_pcd.T) + trans_).T

            # compare_pcd(src_xyz_cl_best_pred, tgt_pcd)
            # compare_pcd(src_pcd_pred, tgt_pcd)

            vis_pc_marker(src_vs=src_xyz_cl_best_pred, tgt_vs=tgt_pcd)
            vis_pc_marker(src_vs=src_pcd_pred, tgt_vs=tgt_pcd)

            vis_mesh_pc(src_pcd, src_xyz_cl_best)

            cl_corr = data['cl_corr'][best_cl_index].detach().cpu()
            scale_factor = 0.013 *4

            compare_pcd(src_xyz_cl_best_pred[cl_corr[:, 0]], tgt_pcd[cl_corr[:, 1]])

            viz_coarse_nn_correspondence_mayavi(src_xyz_cl_best + s_c / scale, tgt_pcd + t_c / scale + 1,
                                                cl_corr.T,
                                                f_src_pcd=None,
                                                f_tgt_pcd=None,
                                                scale_factor=scale_factor )

    else:

        src_pcd_sampled = src_pcd[match_pred_scores[:, 0], :]
        tgt_pcd_sampled = tgt_pcd[match_pred_scores[:, 1], :]
        correspondences = np.zeros(np.shape(match_pred_scores))
        correspondences[:, 0] = np.arange(match_pred_scores.shape[0])
        correspondences[:, 1] = np.arange(match_pred_scores.shape[0])
        src_warped_sampled, src_marker_pred, rot_, trans_ = eva_regist(src_pcd_sampled, tgt_pcd_sampled, correspondences,
                                                               src_marker=src_marker,
                                                               debug=debug)

    if debug:
        # scale_factor = 0.013 * 4

        # viz_coarse_nn_correspondence_mayavi(src_pcd + s_c/scale, tgt_pcd + t_c/scale +1, match_pred_scores.T, f_src_pcd=None,
        #                                     f_tgt_pcd=None,
        #                                     scale_factor=scale_factor )

        src_warped = (np.matmul(rot_, src_pcd.T) + trans_).T

        vis_pc_marker(src_warped, src_marker_pred, tgt_pcd, tgt_marker)
        # vis_pc_marker(src_vs=src_warped, tgt_vs= tgt_pcd)
        vis_pc_marker(src_warped, None, tgt_pcd, None)
        #compare_pcd(src_warped, tgt_pcd)

    cal_error(tgt_marker * scale, src_marker * scale, True)

    diff_mean, diff_max, diff_std, RE = cal_error(tgt_marker * scale, src_marker_pred * scale, True)

    src_gt = src_pcd * scale + s_c
    src_gt = (np.matmul(R_gt, src_gt.T) + t_gt).T
    tgt_raw = tgt_pcd * scale + t_c

    # src_marker_gt = src_marker * scale + s_c
    # src_marker_gt = (np.matmul(R_gt, src_marker_gt.T) + t_gt).T
    # tgt_marker_gt = tgt_marker * scale + t_c

   # cal_error(tgt_marker_gt, src_marker_gt, True)

    # _, inlier_ratio, ms, _ = eval_matrics(src_gt, tgt_raw, match_pred_scores, th=scale * voxel_size)
    inlier_ratio = 0
    ms = 0


    return inlier_ratio, ms, RE


if __name__ == '__main__':

    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/new_task3_004_002.yaml"
    pretrain_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth"

    test_root_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/Test/"
    test_list_file_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Dataset/Deform_mesh_npz_test/list.npz"

    result_path = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Results/Rigid_results/Our/"
    result_path_corr = "/media/yzx/yzx_store1/Task03_Liver/Train_Test/Results/Corr/Our/"

    rot = True
    voxel_size = 0.04
    K = 5

    config = load_config(config_path)
    config = edict(config)
    config.architecture = architectures[config.model_name]
    config.device = torch.device('cuda:0')
    model = KPFCNN(config, K=K).to(config.device).eval()
    #model = KPFCNN(config).to(config.device).eval()
    state = torch.load(pretrain_path)
    model.load_state_dict(state['state_dict'])

    # """ONE"""
    sigma = None
    cluster = True  # registration results from cluster, the first priority, otherwise, go to SVD /
    use_SVD = True # if false, go to ransac
    use_corr_cl = True # raw corr from cl
    use_all_matches = False # filtered corr from cl

    demo_set = LiverDemo(config, test_root_path, test_list_file_path, voxel_size, sigma, rot)
    # 5205
    IR, MS, RE = eva_one(3596, debug=True, use_SVD=use_SVD, cluster=cluster,
                                   use_corr_cl=use_corr_cl, use_all_matches=use_all_matches)  # 3506 good example, 286, 212, 5205, 225, 3596, 6112
    # bad example 3567, 723
    print("IR: ", IR * 100, "MS: ", MS * 100, "RE:", RE)

    # """All"""
    #
    # for K in [1]: #[2, 3, 4, 5, 6, 7, 8] [5]
    #
    #     config = load_config(config_path)
    #     config = edict(config)
    #     config.architecture = architectures[config.model_name]
    #     config.device = torch.device('cuda:0')
    #     model = KPFCNN(config, K=K).to(config.device).eval()
    #     state = torch.load(pretrain_path)
    #     model.load_state_dict(state['state_dict'])
    #
    #     #sigma = None # Noise magnitude
    #     for sigma in [None]:
    #         use_SVD = True
    #         cluster = True # registration results from cluster
    #         use_corr_cl = True # corr from clusters
    #         use_all_matches = True # filter the corr from cluster with res
    #
    #         demo_set = LiverDemo(config, test_root_path, test_list_file_path, voxel_size, sigma, rot)
    #
    #         #diff_mean, diff_max, diff_std, RE = eva_one(286, debug=True, use_SVD=use_SVD, cluster=cluster, use_corr_cl=use_corr_cl) # 3506 good example
    #
    #         RE_list = []
    #         IR_list = []
    #         MS_list = []
    #
    #         for i in tqdm(range(len(demo_set))):
    #             IR, MS, RE = eva_one(i, debug=False, use_SVD=use_SVD, cluster=cluster, use_corr_cl=use_corr_cl, use_all_matches=use_all_matches)
    #             RE_list.append(RE)
    #             IR_list.append(IR * 100)
    #             MS_list.append(MS * 100)
    #
    #         # np.save(result_path + str(K) + "_w_tf", RE_list)
    #         # np.save(result_path_corr + str(K) +"_w_tf_IR", IR_list)
    #         # np.save(result_path_corr + str(K) +"_w_tf_MS", MS_list)
    #         #
    #         # np.save(result_path + "None" + "_w_tf", RE_list)
    #         # np.save(result_path_corr + "None" +"_w_tf_IR", IR_list)
    #         # np.save(result_path_corr + "None" +"_w_tf_MS", MS_list)
    #
    #         # np.save(result_path + str(sigma) + "_w_cl_LiverMatch", RE_list)
    #         # np.save(result_path_corr + str(sigma) + "_w_cl_LiverMatch", IR_list)
    #         # np.save(result_path_corr + str(sigma) + "_w_cl_LiverMatch", MS_list)
    #
    #         np.save(result_path + "None_K_" + str(K) + "_w_cl_LiverMatch", RE_list)
    #
    #         # np.save(result_path + str(sigma) + "_w_cl_LiverMatch", RE_list)
    #         # np.save(result_path_corr + str(sigma) + "_w_cl_LiverMatch", IR_list)
    #         # np.save(result_path_corr + str(sigma) + "_w_cl_LiverMatch", MS_list)
    #
    #         # np.save(result_path + "None" + "_wo_tf", RE_list)
    #         # np.save(result_path_corr + "None" + "_wo_tf_IR", IR_list)
    #         # np.save(result_path_corr + "None" + "_wo_tf_MS", MS_list)
    #
    #         # np.save(result_path + str(K) + "_no_cl", RE_list)
    #         # np.save(result_path_corr + str(K) + "_no_cl_IR", IR_list)
    #         # np.save(result_path_corr + str(K) + "_no_cl_MS", MS_list)
    #         #
    #         # np.save(result_path + "None" + "_w_tf", RE_list)
    #         # np.save(result_path_corr + "None" + "_w_tf_IR", IR_list)
    #         # np.save(result_path_corr + "None" + "_w_tf_MS", MS_list)


















