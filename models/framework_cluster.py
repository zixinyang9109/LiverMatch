import numpy as np
import torch
import torch.nn.functional as F
from models.blocks import *
from lib.visualization import compare_pcd
from models.transformer import Transformer
from models.one_way_transformer import Transformer_Src
from models.matching import Matching

from pointnet2_ops import pointnet2_utils
from lib.utils import generate_node_clusters, index_select
from lib.loss import pairwise_distance

def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t

class KPFCNN(nn.Module):

    def __init__(self, config, K=5):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############
        self.k = K
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2


        feature_dim = config.feature_dim
        self.bottle = nn.Conv1d(in_dim, feature_dim, kernel_size=1, bias=True)
        self.transformer = Transformer(config)
        self.matching = Matching(config)
        self.proj_vis_score = nn.Conv1d(config.feature_dim_m, 1, kernel_size=1, bias=True)
       # self.cluster_attention = Transformer_Src(config, dim=34)
        self.proj_match_score = nn.Conv1d(config.feature_dim_m, 1, kernel_size=1, bias=True)

        #####################
        # List Decoder blocks
        #####################
        out_dim = feature_dim
        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        return


    def forward(self, batch):
        # print(self.encoder_blocks)
        # print(self.decoder_blocks)
        # Get input features
        x = batch['features'].clone().detach()
        len_src_c = batch['stack_lengths'][-1][0]
        len_src_f = batch['stack_lengths'][0][0]
        sigmoid = nn.Sigmoid()

        #################################

        # 1. joint encoder part
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
            #print("encoder feats_dim at " + str(block_i) + " :", x.size())

        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0, 1).unsqueeze(0)  # [1, C, N]
        feats_c = self.bottle(feats_c)  # [1, C, N]

        #################################
        # 3. apply TF to communicate the features
        src_feats_c, tgt_feats_c = feats_c[:, :, :len_src_c], feats_c[:, :, len_src_c:]
        src_feats_c, tgt_feats_c = src_feats_c.transpose(1, 2), tgt_feats_c.transpose(1, 2)

        src_feats_c, tgt_feats_c = self.transformer(src_feats_c, tgt_feats_c
        ) # [1,C ,N]

        src_feats_c, tgt_feats_c = src_feats_c.transpose(1, 2), tgt_feats_c.transpose(1, 2)

        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        x = feats_c.squeeze(0).transpose(0, 1)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
            #print("decoder feats_dim at " + str(block_i) + " :", x.size())

        feats_f = x

        batch.update({'feats': feats_f})

        # matching
        src_feats, tgt_feats = feats_f[:len_src_f], feats_f[len_src_f:]
        conf_matrix_pred, match_pred = self.matching(src_feats.unsqueeze(0), tgt_feats.unsqueeze(0))
        batch.update({'conf_matrix_pred': conf_matrix_pred, 'match_pred': match_pred})

        feats = {'src_feats_cuda': src_feats, 'tgt_feats_cuda': tgt_feats}
        # {'tensor_one': tensor_a, 'tensor_two': tensor_b}
        torch.save(feats, '/media/yzx/yzx_store1/Task03_Liver/Train_Test/P2P_Demo/feats.pt')

        scores_vis = self.proj_vis_score(src_feats.unsqueeze(0).transpose(1, 2))
        scores_vis = torch.clamp(sigmoid(scores_vis.view(-1)), min=0, max=1)
        batch.update({'scores_vis': scores_vis})

        # cluster proposal
        src_pcd = batch['src_pcd_raw']
        tgt_pcd = batch['tgt_pcd_raw']

        if self.training:
            correspondence = batch['correspondences']

            cluster_num = 1
            fps_idx = correspondence[torch.randint(len(correspondence), (cluster_num,)),0]
            src_node = src_pcd[fps_idx]

        else:

            # scores_vis_mask = scores_vis > 0.5
            # vis_src_pcd = src_pcd[scores_vis_mask]

            sim_matrix = torch.einsum("bsc,btc->bst", src_feats.unsqueeze(0), tgt_feats.unsqueeze(0))
            src_conf = sim_matrix.sum(dim=2)  # .sort(descending=True, dim=1)

            _, src_idx = src_conf.squeeze(0).sort(descending=True, dim=0)
            vis_src_pcd = src_pcd[src_idx[:len(tgt_pcd.squeeze(0))], :]

            #vis_src_pcd = src_pcd

            cluster_num = self.k #len(src_pcd)/len(tgt_pcd)
            fps_idx = pointnet2_utils.furthest_point_sample(vis_src_pcd.unsqueeze(0), cluster_num)
            fps_idx = fps_idx.squeeze(0).long()
            src_node = vis_src_pcd[fps_idx]

        batch.update({'vis_src_pcd': vis_src_pcd})

        win_size = min([len(tgt_pcd), len(src_pcd)])
        src_indices_cl = generate_node_clusters(src_pcd, src_node, point_limit=win_size)
        src_xyz_cl = index_select(src_pcd, src_indices_cl, dim=0)
        src_feats_cl = index_select(src_feats, src_indices_cl, dim=0)
        batch.update({'src_xyz_cl': src_xyz_cl})

        # cluster atention and match
        tgt_feats_cl = tgt_feats.unsqueeze(0).repeat(cluster_num, 1, 1)
        #src_feats_cl, tgt_feats_cl = self.cluster_attention(src_feats_cl, tgt_feats_cl)
        conf_matrix_pred_cl, match_pred_cl = self.matching(src_feats_cl, tgt_feats_cl)
        batch.update({'conf_matrix_pred_cl': conf_matrix_pred_cl, 'match_pred_cl': match_pred_cl})

        # cluster wise rigid registration
        if not self.training:
            cl_unique, cl_counts = torch.unique(match_pred_cl[:, 0], return_counts=True, dim=0)

            # # batch approach
            # max_corr = max(cl_counts)
            # num_cl = len(cl_unique)
            # batch_conf = torch.zeros(num_cl, max_corr).cuda() # B, N
            # batch_src_xyz_cl_corr = torch.zeros(num_cl, max_corr, 3).cuda()
            # batch_tgt_xyz_cl_corr = torch.zeros(num_cl, max_corr, 3).cuda()
            #
            # for i, cl_i in enumerate(cl_unique):
            #     match_pred_i = match_pred_cl[match_pred_cl[:, 0] == cl_i]
            #     s_id, t_id = match_pred_i[:, 1], match_pred_i[:, 2]
            #     corr = torch.stack((s_id, t_id), dim=1)
            #
            #     src_xyz_cl_i = src_xyz_cl[cl_i, s_id].detach()  # .numpy()
            #     tgt_xyz_cl_i = tgt_pcd[t_id].detach()  # .numpy()
            #
            #     conf_v = conf_matrix_pred_cl[cl_i, s_id, t_id].detach()
            #
            #     batch_conf[i, :cl_counts[i]] = conf_v
            #     batch_src_xyz_cl_corr[i, :cl_counts[i], :] = src_xyz_cl_i
            #     batch_tgt_xyz_cl_corr[i, :cl_counts[i], :] = tgt_xyz_cl_i
            # batch_rot_cl, batch_t_cl = weighted_procrustes(batch_src_xyz_cl_corr, batch_tgt_xyz_cl_corr, batch_conf)
            # batch_src_xyz_cl_corr_warp = torch.matmul(batch_src_xyz_cl_corr, batch_rot_cl.transpose(-1, -2)) + batch_t_cl
            #
            # batch_corr_residuals = torch.linalg.norm(
            #     batch_tgt_xyz_cl_corr.unsqueeze(0) - batch_src_xyz_cl_corr_warp, dim=2
            # ).sum(dim=1)
            # batch_corr_residuals = batch_corr_residuals/cl_counts
            # min_index = batch_corr_residuals.argmin()
            # batch.update({'best_R': batch_rot_cl[min_index], 'best_t': batch_t_cl[min_index]})

            # iterative approach, longer time but less memory
            cl_res = []
            cl_R = []
            cl_t = []
            cl_corr = [] # cluster corr
            cl_corr_g = [] # cluster corr to global
            cl_inlier = []
            CP_res = []

            """ global"""

            corr_g = match_pred[:, 1:]
            src_id = corr_g[:, 0]
            tgt_id = corr_g[:, 1]

            rot_g, t_g = weighted_procrustes(src_pcd[src_id, :], tgt_pcd[tgt_id, :],
                                             conf_matrix_pred[0, src_id, tgt_id])

            # rot_g, t_g = weighted_procrustes(src_pcd[src_id, :], tgt_pcd[tgt_id, :],
            #                                  torch.ones([len(src_id)]).to(src_pcd.device))

            src_warp = torch.matmul(src_pcd, rot_g.T) + t_g.T
            dist_mat = pairwise_distance(tgt_pcd, src_warp)
            cp_res_g = torch.mean(dist_mat.min(dim=1).values)

            cl_R.append(rot_g)
            cl_t.append(t_g)
            CP_res.append(cp_res_g)

            """ cluster"""
            for cl_i in cl_unique:
                match_pred_i = match_pred_cl[match_pred_cl[:, 0] == cl_i]
                s_id, t_id = match_pred_i[:, 1], match_pred_i[:, 2]
                corr = torch.stack((s_id, t_id), dim=1)

                s_id_g = src_indices_cl[cl_i, s_id]
                corr_g_cl = torch.stack((s_id_g, t_id), dim=1)

                src_xyz_cl_i = src_xyz_cl[cl_i, s_id].detach()  # .numpy()
                tgt_xyz_cl_i = tgt_pcd[t_id].detach()  # .numpy()

                conf_v = conf_matrix_pred_cl[cl_i, s_id, t_id].detach()

                rot_cl_i, t_cl_i = weighted_procrustes(src_xyz_cl_i, tgt_xyz_cl_i, conf_v)

                src_xyz_cl_i_warp = torch.matmul(src_xyz_cl_i, rot_cl_i.T) + t_cl_i.T
                res = torch.linalg.norm(tgt_xyz_cl_i - src_xyz_cl_i_warp, dim=1)

                inlier_num = torch.lt(res, 0.05).sum()

                src_warp = torch.matmul(src_pcd, rot_cl_i.T) + t_cl_i.T
                dist_mat = pairwise_distance(tgt_pcd, src_warp)
                cp_res = torch.mean(dist_mat.min(dim=1).values) # closest point distance

                cl_res.append(res.mean())
                cl_R.append(rot_cl_i)
                cl_t.append(t_cl_i)
                cl_corr.append(corr)
                cl_corr_g.append(corr_g_cl)
                cl_inlier.append(inlier_num)
                CP_res.append(cp_res)
            # the best index
            min_index = cl_res.index(min(cl_res)) + 1
            min_index_il = cl_inlier.index(max(cl_inlier)) + 1
            min_index_cp = CP_res.index(min(CP_res))

            # aggregate corr

            cl_corr_g = torch.cat(cl_corr_g, dim=0)
            matches_all = torch.cat([cl_corr_g, corr_g], dim=0)
            cl_corr_g_unique, cl_corr_g_counts = torch.unique(cl_corr_g, return_counts=True, dim=0)
            matches_all_unique, matches_all_counts = torch.unique(matches_all, return_counts=True, dim=0)

            # filter with counts
            #cl_corr_g_unique = cl_corr_g_unique[cl_corr_g_counts>1]
            #matches_all_unique = matches_all_unique[matches_all_counts > 1]

            """"filter all corr with res"""

            rot_best = cl_R[min_index_cp]
            trans_best = cl_t[min_index_cp]
            src_warp = torch.matmul(src_pcd[matches_all_unique[:, 0]], rot_best.T) + trans_best.T
            res = torch.linalg.norm(tgt_pcd[matches_all_unique[:, 1]] - src_warp, dim=1)
            #matches_all_unique = matches_all_unique[res < (min(CP_res))]
            matches_all_unique = matches_all_unique[res < 0.1]

            # print(min(CP_res))

            # from lib.visualization import viz_coarse_nn_correspondence_mayavi, compare_pcd
            # index = min_index
            # viz_coarse_nn_correspondence_mayavi()

            batch.update({'R_cl': cl_R, 't_cl': cl_t,
                          'CP_res': CP_res,
                          'best_index': min_index, 'best_index_il': min_index_il, 'best_index_cp': min_index_cp, # best one
                          'cl_corr': cl_corr, 'cl_corr_g_unique': cl_corr_g_unique, 'matches_all_unique': matches_all_unique})

        # torch.save(batch, '/media/yzx/yzx_store1/Task03_Liver/Train_Test/P2P_Demo/demo_data.pt')

        return batch


def debug():
    from lib.util import load_config
    from datasets.liver import livermatch
    from datasets.liver_task3 import liverTask3
    from configs.models import architectures
    from easydict import EasyDict as edict
    from datasets.dataloader import collate_fn_descriptor
    from lib.loss import LiverLoss_cluster

    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver_new_task3_cluster.yaml"
    config = load_config(config_path)
    config = edict(config)

    dataset = liverTask3(config, "train")

    config.architecture = architectures[config.model_name]
    config.device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    model = KPFCNN(config).to(config.device).eval()
    loss = LiverLoss_cluster(config)

    neighborhood_limits = [5, 14, 24, 32]
    index = 0
    list_data = dataset.__getitem__(index)
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
    loss_info = loss(data)


if __name__ == '__main__':
    from lib.util import load_config
    from datasets.liver import livermatch
    from datasets.liver_task3 import liverTask3
    from configs.models import architectures
    from easydict import EasyDict as edict
    from datasets.dataloader import collate_fn_descriptor
    from lib.loss import LiverLoss_cluster

    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver.yaml"
    pretrain_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/snapshot/liver_new_task3/checkpoints/model_best_loss.pth"  # "/home/yzx/yzx/Deformable_Registration/LiverMatch/snapshot/liver_task3_grid_off_norm/checkpoints/model_best_loss.pth"

    config = load_config(config_path)
    config = edict(config)
    config.architecture = architectures[config.model_name]
    config.device = torch.device('cuda:0')
    model = KPFCNN(config).to(config.device)

    # Get the current model's state dictionary
    model_dict = model.state_dict()

    pretrained_dict = torch.load(pretrain_path)['state_dict']

    # Update the model's state dictionary with the pretrained weights
    # This step filters out unnecessary keys and ensures that the sizes match
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # model_dict.update(pretrained_dict)

    # Load the updated dictionary back into the model
    model.load_state_dict(model_dict)

    for name, param in model.named_parameters():
        if name in pretrained_dict:
            print(f"Layer {name} loaded with pretrained weights")
        else:
            print(f"Layer {name} NOT loaded with pretrained weights")

    # Freeze the layers by setting requires_grad to False
    for name, param in model.named_parameters():
        if name in pretrained_dict:
            param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Layer {name} is trainable")
    #     else:
    #         print(f"Layer {name} is frozen")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver.yaml"
    # # "/home/yzx/yzx/Deformable_Registration/My_Predator-main/configs/Liver3D_full/liver_train.yaml"
    # config = load_config(config_path)
    # config = edict(config)
    # dataset = livermatch(config, 'train', data_augmentation=True)
    #
    # config.architecture = architectures[config.model_name]
    # config.device = torch.device('cuda:0')
    # torch.cuda.set_device(0)
    # model = KPFCNN(config).to(config.device).eval()
    # loss = LiverLoss(config)
    #
    # neighborhood_limits = [13, 21, 29, 36]
    # index = 0
    # list_data = dataset.__getitem__(index)
    # inputs = collate_fn_descriptor([list_data], config, neighborhood_limits)
    #
    # with torch.no_grad():
    #
    #     ##################################
    #     # load inputs to device.
    #     for k, v in inputs.items():
    #         if type(v) == list:
    #             inputs[k] = [item.to(config.device) for item in v]
    #         else:
    #             inputs[k] = v.to(config.device)
    #
    # data = model(inputs)
    # loss_info = loss(data)
    #loss_info['loss'].backward()

