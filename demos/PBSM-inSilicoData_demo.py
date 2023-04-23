import os.path

import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
import torch
from datasets.dataloader import get_dataloader
from lib.util import load_config
from easydict import EasyDict as edict
from configs.models import architectures
from models.framework import KPFCNN
from datasets.dataloader import collate_fn_descriptor, calibrate_neighbors
from lib.visualization import viz_coarse_nn_correspondence_mayavi, compare_pcd
from scipy.spatial.transform import Rotation

import warnings
warnings.filterwarnings("ignore")

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
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid
    return points_centered

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

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

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

def eva_regist(src_pcd, tgt_pcd, corrs, tgt_pcd_full,  distance_threshold=0.01, ransac_n=4, debug =False):
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


    tgt_pcd_full_pred = (np.matmul(rot_, src_pcd.T) + trans_).T
    diff = tgt_pcd_full - tgt_pcd_full_pred
    RE = np.sqrt(np.sum(diff * diff) / len(diff))

    # gc.collect()
    # src_pcd_o3d.clear()
    # tgt_pcd_o3d.clear()
    # del src_pcd_o3d
    # del tgt_pcd_o3d

    if debug:
        # print("compare pred with gt")
        # compare_pcd(tgt_pcd_full_pred, tgt_pcd_full)
        # print("compare tgt with gt")
        # compare_pcd(tgt_pcd, tgt_pcd_full)
        print("compare pred with tgt")
        compare_pcd(tgt_pcd_full_pred, tgt_pcd)

    return RE

class LiverDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, config, src_file, tgt_file):
        super(LiverDemo, self).__init__()
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.config = config

    def __len__(self):
        return 1

    def __getitem__(self, item):
        src_raw = ply2np(self.src_file)
        tgt_raw = ply2np(self.tgt_file)
        src_pcd = center(src_raw)
        tgt_pcd = center(tgt_raw)

        # fake the ground truth information
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3, 1)).astype(np.float32)
        correspondences = torch.ones(1, 2).long()

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

if __name__ == '__main__':

    use_score = True
    debug = True
    save = False
    eva = False
    reg = True
    scale_factor = 0.013

    src_file = "/home/yzx/yzx/Deformable_Registration/LiverMatch/test_data/Liver1/rot_demo/src.ply"
    tgt_file = "/home/yzx/yzx/Deformable_Registration/LiverMatch/test_data/Liver1/rot_demo/tgt_tf_1.0.ply"
    gt_file = "/home/yzx/yzx/Deformable_Registration/LiverMatch/test_data/Liver1/rot_demo/gt_tf_1.0.ply"
    file_name_cor = "/home/yzx/yzx/Deformable_Registration/LiverMatch/test_data/Liver1/rot_demo/pred_corr.txt" #if you want to save the matches

    
    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver.yaml"
    pretrain_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/snapshot/liver_3D_1_one_transformer/checkpoints/model_best_loss.pth"

    src_pcd = ply2np(src_file)
    tgt_pcd = ply2np(tgt_file)
    if os.path.exists(gt_file):
        gt_pcd = ply2np(gt_file)

    inlier_th = 0.02

    config = load_config(config_path)
    config = edict(config)
    config.architecture = architectures[config.model_name]
    config.device = torch.device('cuda:0')

    demo_set = LiverDemo(config, src_file, tgt_file)
    neighborhood_limits = [19, 23, 29, 34]
    loader, neighborhood_limits = get_dataloader(dataset=demo_set,
                                                              batch_size=1,
                                                              shuffle=True,
                                                              num_workers=0,neighborhood_limits=neighborhood_limits
                                                              )

    model = KPFCNN(config).to(config.device).eval()
    state = torch.load(pretrain_path)
    model.load_state_dict(state['state_dict'])

    list_data = demo_set.__getitem__(0)

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
    scores_vis = data['scores_vis'].detach().cpu()

    if use_score:
        th_score = 0.9
        scores_vis_mask = scores_vis > th_score
        se_index = []
        for i in np.arange(len(scores_vis_mask)):
            if scores_vis_mask[i]:
                if i in match_pred[:, 0]:
                    idx = np.where(match_pred == i)[0][0]
                    se_index.append(idx)

        match_pred_scores = match_pred[se_index, :]
    else:
        match_pred_scores = match_pred

    if debug:
        euler_ab = np.random.rand(3) * np.pi * 0  # anglez, angley, anglex
        # rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
        rot = Rotation.from_euler('zyx', euler_ab).as_matrix()
        trans = 0.8  # $*np.random.rand(3, 1)
        tgt_pcd_debug = (np.matmul(rot, tgt_pcd.T) + trans).T

        viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd_debug, match_pred.T, f_src_pcd=None, f_tgt_pcd=None,
                                            scale_factor=scale_factor * 2)


    if eva:
        rmse_dist_s, inlier_ratio_s, inliers_scores_s, inlier_mask = eval_matrics(gt_pcd, tgt_pcd, match_pred_scores,
                                                                                  inlier_th)
        result = [rmse_dist_s, inlier_ratio_s * 100, inliers_scores_s * 100]
        print("\n  " + ("{:>8} | " * 3).format("rmse", "inlier_ratios%", "inliers_scores%"))
        print(("&{: 8.4f}  " * 3).format(*result))

    if save:
        np.savetxt(file_name_cor, match_pred_scores, fmt='%i')

    if reg:
        RE = eva_regist(src_pcd, tgt_pcd, match_pred_scores, gt_pcd, distance_threshold=0.01, ransac_n=4, debug=True)
        print("RE:", RE)



