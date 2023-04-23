import numpy as np
import torch.nn.functional as F
from models.blocks import *
from lib.visualization import compare_pcd
from models.transformer import Transformer
from models.matching import Matching
from lib.loss import LiverLoss

class KPFCNN(nn.Module):

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############
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

        feats_f = x

        batch.update({'feats': feats_f})

        # matching
        src_feats, tgt_feats = feats_f[:len_src_f], feats_f[len_src_f:]
        conf_matrix_pred, match_pred = self.matching(src_feats.unsqueeze(0), tgt_feats.unsqueeze(0))
        batch.update({'conf_matrix_pred': conf_matrix_pred, 'match_pred': match_pred})

        scores_vis = self.proj_vis_score(src_feats.unsqueeze(0).transpose(1, 2))
        scores_vis = torch.clamp(sigmoid(scores_vis.view(-1)), min=0, max=1)
        batch.update({'scores_vis': scores_vis})

        return batch

if __name__ == '__main__':

    from lib.util import load_config
    from datasets.liver import livermatch
    from configs.models import architectures
    from easydict import EasyDict as edict
    from datasets.dataloader import collate_fn_descriptor

    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver.yaml"
    # "/home/yzx/yzx/Deformable_Registration/My_Predator-main/configs/Liver3D_full/liver_train.yaml"
    config = load_config(config_path)
    config = edict(config)
    dataset = livermatch(config, 'train', data_augmentation=True)

    config.architecture = architectures[config.model_name]
    config.device = torch.device('cuda:1')
    torch.cuda.set_device(1)
    model = KPFCNN(config).to(config.device).eval()
    loss = LiverLoss(config)

    neighborhood_limits = [13, 21, 29, 36]
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
    #loss_info['loss'].backward()

