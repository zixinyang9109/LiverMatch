import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

def compute_match_recall(conf_matrix_gt, match_pred):  # , s_pcd, t_pcd, search_radius=0.3):
    '''
    @param conf_matrix_gt:
    @param match_pred:
    @return:
    '''

    pred_matrix = torch.zeros_like(conf_matrix_gt)

    b_ind, src_ind, tgt_ind = match_pred[:, 0], match_pred[:, 1], match_pred[:, 2]
    pred_matrix[b_ind, src_ind, tgt_ind] = 1.

    true_positive = (pred_matrix == conf_matrix_gt) * conf_matrix_gt

    recall = true_positive.sum() / conf_matrix_gt.sum()

    precision = true_positive.sum() / max(len(match_pred), 1)

    return recall, precision


class LiverLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Matching loss
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.pos_w = config.pos_weight
        self.neg_w = config.neg_weight

        # weight
        self.vis_w = config.vis_weight
        self.match_w = config.match_weight
        self.mat_w = config.matrix_weight

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                          zero_division=0, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def compute_correspondence_loss(self, conf, conf_gt, weight=None):
        '''
        @param conf: [B, L, S]
        @param conf_gt: [B, L, S]
        @param weight: [B, L, S]
        @return:
        '''
        pos_mask = conf_gt == 1
        neg_mask = conf_gt == 0

        pos_w, neg_w = self.pos_w, self.neg_w

        # corner case assign a wrong gt
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            neg_w = 0.

        # focal loss
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma


        pos_conf = conf[pos_mask]
        loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        loss = pos_w * loss_pos.mean()
        return loss



    def match_2_conf_matrix(self, matches_gt, matrix_pred):

        matrix_gt = torch.zeros_like(matrix_pred)
        for b, match in enumerate(matches_gt):
            matrix_gt[b][match[0], match[1]] = 1
        return matrix_gt

    def forward(self, data):
        loss_info = {}

        # visible loss
        src_pcd, tgt_pcd = data['src_pcd_raw'], data['tgt_pcd_raw']
        scores_vis = data['scores_vis']
        correspondence = data['correspondences']

        # only src scores
        src_idx = list(set(correspondence[:, 0].int().tolist()))
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        src_gt_labels = src_gt.to(torch.device('cuda'))
        vis_loss, vis_cls_precision, vis_cls_recall = self.get_weighted_bce_loss(scores_vis, src_gt_labels)
        loss_info.update({'vis_loss': vis_loss, 'vis_recall': vis_cls_recall, 'vis_precision': vis_cls_precision})

        # mat loss
        conf_matrix_pred = data['conf_matrix_pred']

        match_gt = [correspondence.transpose(0, 1)]

        conf_matrix_gt = self.match_2_conf_matrix(match_gt, conf_matrix_pred)
        data['conf_matrix_gt'] = conf_matrix_gt
        mat_loss = self.compute_correspondence_loss(conf_matrix_pred, conf_matrix_gt, weight=None)
        mat_recall, mat_precision = compute_match_recall(conf_matrix_gt, data['match_pred'])

        loss_info.update({'mat_loss': mat_loss, 'mat_recall': mat_recall, 'mat_precision': mat_precision})

        loss = self.vis_w * vis_loss + self.mat_w * mat_loss

        print("mat loss: ", mat_loss.item(),"\n")

        loss_info.update({'loss': loss})

        return loss_info



