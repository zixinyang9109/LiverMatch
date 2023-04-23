from lib.util import load_config
from datasets.liver import livermatch
from configs.models import architectures
from easydict import EasyDict as edict
from datasets.dataloader import collate_fn_descriptor, calibrate_neighbors
import torch
from models.framework import KPFCNN
from lib.loss import LiverLoss
from lib.visualization import viz_coarse_nn_correspondence_mayavi, compare_pcd
import numpy as np
import open3d as o3d
import gc

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
        ransac_n=ransac_n
        # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )  # criteria=o3d.pi

    tsfm = np.array(result_ransac.transformation)

    # src_pcd_o3d.clear()
    # tgt_pcd_o3d.clear()

    rot_ = tsfm[:3, :3]
    trans_ = tsfm[:3, 3:]
    tgt_pcd_full_pred = (np.matmul(rot_, src_pcd.numpy().T) + trans_).T
    diff = tgt_pcd_full.numpy() - tgt_pcd_full_pred
    RE = np.sqrt(np.sum(diff * diff) / len(diff))

    # gc.collect()
    # src_pcd_o3d.clear()
    # tgt_pcd_o3d.clear()
    # del src_pcd_o3d
    # del tgt_pcd_o3d

    if debug:
        compare_pcd(tgt_pcd_full, tgt_pcd_full_pred)

    return RE

def get_match( conf_matrix, thr, mutual=True):

    mask = conf_matrix > thr

    #mutual nearest
    if mutual:
        mask = mask \
               * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
               * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

    #find all valid coarse matches
    #index = (mask==True).nonzero()
    index = torch.nonzero(mask==True)
    #print(index-new_index)
    b_ind, src_ind, tgt_ind = index[:,0], index[:,1], index[:,2]
    mconf = conf_matrix[b_ind, src_ind, tgt_ind]

    return index, mconf, mask


def eval_matrics(tgt_pcd_full, tgt_pcd, pred_corr, th=0.02):

    diff = tgt_pcd_full[pred_corr[:, 0], :] - tgt_pcd[pred_corr[:, 1], :]

    dist = torch.norm(diff, dim=1)
    # print(torch.median(dist))
    # print(torch.max(dist))

    len_pred = len(pred_corr)
    len_gt = len(tgt_pcd)

    inliers = (dist < th).float()
    # print("old inliers:", inliers.sum().float())
    # print("n_match", len(inliers))

    # inliers = torch.sum((diff) ** 2, dim=1) < th ** 2
    # print("new inliers:", inliers.sum().float())

    inlier_ratio = inliers.mean()

    inliers_scores = torch.sum(inliers)/len_gt

    rmse_dist = torch.sqrt(torch.sum(dist ** 2) / len_pred)

    return rmse_dist, inlier_ratio, inliers_scores, inliers

def eva_one(index, print_result=False, inlier_th=0.02, use_score=True, debug=True, eva_re = True):

    list_data = dataset.__getitem__(index)
    #gt_corr = list_data[6]
    tgt_pcd_full, src_pcd, tgt_pcd = dataset.entry2data(index, get_full=True)
    #tgt_pcd = tgt_pcd-tgt_noise
    tgt_pcd_full = torch.from_numpy(tgt_pcd_full)
    src_pcd = torch.from_numpy(src_pcd)
    tgt_pcd = torch.from_numpy(tgt_pcd)
    if debug:
        print("vis raw data")
        compare_pcd(s_pc=src_pcd)
        compare_pcd(tgt_pcd=tgt_pcd)
        compare_pcd(src_pcd, tgt_pcd)

    inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)


    if debug:
        print("vis key points data")
        len_src_c = inputs['stack_lengths'][-1][0]
        pcd_c = inputs['points'][-1]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c].cpu(), pcd_c[len_src_c:].cpu()
        compare_pcd(s_pc=src_pcd_c, scale_factor = 0.013*2)
        compare_pcd(tgt_pcd=tgt_pcd_c, scale_factor = 0.013*2)

        # print("key points")
        # compare_pcd(src_pcd_c.cpu(), tgt_pcd_c.cpu())

    with torch.no_grad():
        ##################################
        # load inputs to device.
        for k, v in inputs.items():
            if type(v) == list:
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)

    data = model(inputs)
    # loss_info = loss(data)
    # scores_vis = loss_info['scores_vis']
    # match_pred = loss_info['match_pred']

    # src_pcd = data['src_pcd_raw'].detach().cpu()
    # tgt_pcd = data['tgt_pcd_raw'].detach().cpu()
    # if debug:
    #     compare_pcd(src_pcd, tgt_pcd)

    match_pred = data['match_pred'].detach().cpu()
    match_pred = match_pred[:, 1:]

    # conf_matrix_pred = data['conf_matrix_pred'].detach().cpu()
    # index, mconf, mask = get_match(conf_matrix_pred,thr=0)
    #
    # c_value = mconf.numpy()
    # import matplotlib.pyplot as plt
    # #plt.scatter(np.arange(len(c_value)),c_value)

    scores_vis = data['scores_vis'].detach().cpu()

    if use_score:
        th_score = 0.0
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


    scale_factor = 0.013
    if debug:
        viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd, match_pred.T, f_src_pcd=None, f_tgt_pcd=None,
                                            scale_factor=scale_factor*2)
        viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd, match_pred_scores.T, f_src_pcd=None, f_tgt_pcd=None,
                                            scale_factor=scale_factor*2)
    # #compare_pcd(tgt_pcd_full, tgt_pcd)
    #
    #
    #rmse_dist, inlier_ratio, inliers_scores = eval_matrics(tgt_pcd_full, tgt_pcd, match_pred, th=0.01)
    rmse_dist_s, inlier_ratio_s, inliers_scores_s, inlier_mask = eval_matrics(tgt_pcd_full, tgt_pcd, match_pred_scores, inlier_th)

    # c_value_inlier = c_value[inlier_mask>0]
    # plt.scatter(np.arange(len(c_value_inlier)),c_value_inlier)

    vis_ratio = len(tgt_pcd) / len(tgt_pcd_full)

    if eva_re:
        RE = eva_regist(src_pcd, tgt_pcd, match_pred_scores, tgt_pcd_full, distance_threshold=0.01, ransac_n=4, debug=debug)
    else:
        RE = 0

        # search_new = True
    # idx = np.random.permutation(gt_corr.shape[0])[:20]
    # corr_gt = gt_corr[idx]
    # RE = eva_regist(src_pcd, tgt_pcd, corr_gt, tgt_pcd_full, distance_threshold=0.01, ransac_n=4, debug=debug)
    # viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd, gt_corr.T, f_src_pcd=None, f_tgt_pcd=None,
    #                                     scale_factor=scale_factor * 2)


    if print_result:
        # print("RE ", RE)
        # print("inlier ratio: ", inlier_ratio_s.numpy())
        # print("vis ratio: ", vis_ratio)

        print("\n  " + ("{:>8} | " * 3).format("rmse", "inlier_ratios%", "inliers_scores%"))
        result = [rmse_dist_s*100, inlier_ratio_s*100, inliers_scores_s*100]
        print(("&{: 8.4f}  " * 3).format(*result))

        # file_name_cor = "/media/yzx/yzx_store/Results/livermatch/" + str(index).zfill(4) + ".txt"
        #
        # np.savetxt(file_name_cor, match_pred_scores, fmt='%i')

    return rmse_dist_s*100, inlier_ratio_s*100, inliers_scores_s*100, vis_ratio*100, RE

def eva_all( num, inlier_th=0.02, use_score=True, debug=False):
    from tqdm import tqdm
    inlier_ratios = []
    match_scores = []
    Vis = []
    RE_list = []

    for i in tqdm(np.arange(num)): #len(dataset)
        _, ir, ms, vis, RE = eva_one(i, True, inlier_th, use_score=use_score, debug=debug)
        inlier_ratios.append(ir)
        match_scores.append(ms)
        Vis.append(vis)
        RE_list.append(RE)

    print("\n  " + ("{:>8} | " * 4).format("inlier_ratios", "std", "match_recalls",
                                           "std"))
    result = [np.mean(inlier_ratios), np.std(inlier_ratios),
              np.mean(match_scores), np.std(match_scores)]
    print(("&{: 8.2f}  " * 4).format(*result))

    print("RE: ", np.mean(RE_list))
    print("RE_std: ", np.std(RE_list))

    return np.array(inlier_ratios), np.array(match_scores), np.array(Vis), np.array(RE_list)




if __name__ == '__main__':



    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver.yaml"
    pretrain_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/snapshot/liver_3D_1_one_transformer/checkpoints/model_best_loss.pth"

    inlier_th = 0.01

    config = load_config(config_path)
    config = edict(config)
    dataset = livermatch(config, 'test')

    config.architecture = architectures[config.model_name]
    config.device = torch.device('cuda:0')
    model = KPFCNN(config).to(config.device).eval()
    state = torch.load(pretrain_path)
    model.load_state_dict(state['state_dict'])
    loss = LiverLoss(config)

    neighborhood_limits = calibrate_neighbors(livermatch(config, 'test'), config, collate_fn_descriptor)

    use_score = True

    #eva_one(0,  print_result=True, inlier_th=inlier_th, use_score=use_score, debug=True)

    inlier_ratios, match_scores, Vis, RE = eva_all(len(dataset), inlier_th=inlier_th, use_score=use_score)

    #for inlier_th in [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]: # 0.001 is regarded as 0, due to the number precision
    #inlier_ratios, match_scores, Vis = eva_all(len(dataset), inlier_th=inlier_th, use_score=use_score)

    # import matplotlib.pyplot as plt
    # plt.scatter(Vis, inlier_ratios)
    # 311,  346,  531,  536,  576,  577, 1085
    # npz_file = "/media/yzx/yzx_store/Results/livermatch.npz"
    # np.savez(npz_file, ir=inlier_ratios, ms=match_scores, re = RE)










