import sys
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


HMN_intrin = np.array([443, 256, 443, 250])
cam_intrin = np.array([443, 256, 443, 250])
from lib.visualization import viz_flow_mayavi, viz_coarse_nn_correspondence_mayavi, compare_pcd
from lib.util import to_o3d_pcd, to_tsfm, get_correspondences
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class livermatch(Dataset):

    def __init__(self, config, split, data_augmentation=True):
        super(livermatch, self).__init__()

        assert split in ['train','val','test']

        self.root_path = config.root_path
        self.split = split
        if split == 'test':
            self.root_path = config.test_root_path
        self.split = split
        npz_file = self.root_path + config.npz_list
        self.entries_split = np.load(npz_file)
        self.entries = self.entries_split[split]

        if config.use_slice:
            self.entries = self.read_entries(config.slice) # read part npz files
        else:
            if split != 'test':
                self.entries = self.read_entries()

        if split =='test':
            self.data_augmentation = False
            self.root_path = config.test_root_path
        else:
            self.data_augmentation = data_augmentation

        self.config = config

        self.rot_factor = 1.
        # if you want to add additonal noise, we already added noise to the files.
        self.augment_noise = config.augment_noise
        self.max_points = config.max_points
        # instead of using simulated ground truth, you can search the correspondences via kd tree.
        self.overlap_radius = config.overlap_radius


    def read_entries (self, d_slice=None, shuffle= True):

        if self.split =='test':
            shuffle = False
        if shuffle:
            random.shuffle(self.entries)
        if d_slice:
            return self.entries[:d_slice]
        return self.entries

    def entry2data(self, index, scale=100.0, get_full=False):
        name = self.root_path+self.entries[index]

        with np.load(name,allow_pickle=True) as entry:
            # get transformation
            if 'max_dist' in entry.files:
                max_dist = entry['max_dist']
                f_scale = scale/max_dist
            else:
                f_scale = scale

            rot = entry['rot']
            trans = entry['trans']/f_scale
            s2t_flow = entry['s2t_flow']/f_scale
            src_pcd = entry['s_pc']/f_scale
            tgt_pcd = entry['t_pc']/f_scale
            full_tgt_pcd = entry['t_pc_full']/f_scale
            correspondences = entry['correspondences'] # from simulation
            if 'f_mask' in entry.files:
                f_mask = entry['f_mask'] # anterior face

        # if you want to search corr
        # tsfm = to_tsfm(rot, trans)
        # correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm, self.overlap_radius)

        if get_full:
            return full_tgt_pcd, src_pcd, tgt_pcd
        else:
            return rot, trans, s2t_flow, src_pcd, tgt_pcd, correspondences, f_mask, scale

    def center(self, points):
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid
        return points_centered

    def crop(self, points, p_keep, rand_xyz=None):
        if rand_xyz is None:
            rand_xyz = self.uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :], mask, rand_xyz

    def uniform_2_sphere(self, num: int = None):
        """Uniform sampling on a 2-sphere
        Source: https://gist.github.com/andrewbolster/10274979
        Args:
            num: Number of vectors to sample (or None if single)
        Returns:
            Random Vector (np.ndarray) of size (num, 3) with norm 1.
            If num is None returned value will have size (3,)

        """
        if num is not None:
            phi = np.random.uniform(0.0, 2 * np.pi, num)
            cos_theta = np.random.uniform(-1.0, 1.0, num)
        else:
            phi = np.random.uniform(0.0, 2 * np.pi)
            cos_theta = np.random.uniform(-1.0, 1.0)

        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return np.stack((x, y, z), axis=-1)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index, debug=False, vis_corr=False, use_mask =True):

        rot, trans, s2t_flow, src_pcd, tgt_pcd_full, _, f_mask, scale = self.entry2data(index)
        src_pcd_deformed = src_pcd + s2t_flow

        if debug:
            compare_pcd(src_pcd, src_pcd_deformed)

        if self.data_augmentation: # crop surface every time
            p_f = len(src_pcd) / len(f_mask)
            # visibility ratio. As the original data has been volized, it is a reasonalbel assumption
            p = (0.20 + 0.04 * np.random.rand(1)[0]) * p_f
            if p > 1:
                p = 1.0
            if use_mask:
                tgt_pcd, mask, rand_xyz = self.crop(tgt_pcd_full[f_mask, :], p)
            else:
                tgt_pcd, mask, rand_xyz = self.crop(tgt_pcd_full, p)

            if debug:
                print("compare the partial red and the full blue")
                compare_pcd(tgt_pcd, tgt_pcd_full) # red ,blue
        else:
            tgt_pcd = tgt_pcd_full

        # if we get too many points, we do some downsampling
        if (src_pcd.shape[0] > self.max_points):

            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
            src_pcd_deformed = src_pcd_deformed[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]

        if self.split=='test':
            #tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tsfm = to_tsfm(rot, trans)
            correspondences = get_correspondences(to_o3d_pcd(src_pcd_deformed), to_o3d_pcd(tgt_pcd), tsfm,
                                                  self.overlap_radius)
        else:
            correspondences = np.zeros([np.sum(mask * 1), 2])
            correspondences[:, 0] = f_mask[mask]  # np.arange(np.sum(mask*1))
            correspondences[:, 1] = np.arange(np.sum(mask * 1))  # f_mask[mask]
            correspondences = correspondences.astype(int)
            correspondences =torch.from_numpy(correspondences)

        if (correspondences.size(0) < 20 and self.split == 'train'):
            print("Data jump at ",self.entries[index])
            return self.__getitem__(np.random.choice(len(self.entries), 1)[0])

        scale_factor = 0.013

        if debug:
            print("before augmentaion")
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)

            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(src_pcd[ :, 0] , src_pcd[ :, 1], src_pcd[:,  2], scale_factor=scale_factor , color=c_red)
            mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.show()


        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            #rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            rot = Rotation.from_euler('zyx', euler_ab).as_matrix()
            trans = np.random.rand(3, 1) * 0.1
            if (np.random.rand(1)[0] > 0.5):
                tgt_pcd = (np.matmul(rot, tgt_pcd.T) + trans).T

        if debug:
            import mayavi.mlab as mlab
            print("after augmentation")
            print("src: red, src_wrapped: pink, tgt_pcd: blue")
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            # wrapp_src = (np.matmul(rot, src_pcd.T)+ trans).T
            src_wrapped = (np.matmul(rot, src_pcd_deformed.T) + trans).T
            mlab.points3d(src_pcd[:, 0], src_pcd[:, 1], src_pcd[:, 2], scale_factor=scale_factor,
                         color=c_red)
            mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2], scale_factor=scale_factor, color=c_blue)
            mlab.show()

        if vis_corr:
            print("corr after augmentation")
            viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd, correspondences.T, f_src_pcd=None, f_tgt_pcd=None,
                                                scale_factor=scale_factor)


        if (trans.ndim == 1):
            trans = trans[:, None]

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        # move to center, so actually it does not care about the trainsition
        src_pcd = self.center(src_pcd)
        tgt_pcd = self.center(tgt_pcd)


        if debug:
            compare_pcd(src_pcd,tgt_pcd)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, \
               rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

def test_dataloader(mode, config,i):

    from easydict import EasyDict as edict
    from lib.tictok import Timers
    import yaml
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return '_'.join([str(i) for i in seq])

    yaml.add_constructor('!join', join)

    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = edict(config)
    config.timers = Timers()
    D = livermatch(config.dataset, mode)
    print("the total num: ",len(D))
    data = D.__getitem__(i, debug=True, vis_corr=True)
    return data


if __name__ == '__main__':


    mode = "train"
    config_file = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver.yaml"
    src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, _, _, _ = test_dataloader(mode, config_file, 153)




