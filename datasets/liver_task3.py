import sys
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d
HMN_intrin = np.array([443, 256, 443, 250])
cam_intrin = np.array([443, 256, 443, 250])
from lib.visualization import viz_flow_mayavi, viz_coarse_nn_correspondence_mayavi, compare_pcd
from lib.util import to_o3d_pcd, to_tsfm, get_correspondences_n
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from lib.visualization import vis_pc_marker

def read_dict(out_name):
    import json
    f = open(out_name)
    data = json.load(f)
    f.close()
    return data

def load_from_npz(file):
    with np.load(file, allow_pickle=True) as entry:
        vs = entry['vs_vox']
        sel_ids = entry['index']

    return vs, sel_ids

def load_test_from_npz(file, sigma=None, rot=False):
    with np.load(file, allow_pickle=True) as entry:
        vs = entry['src_pcd']
        # faces = entry['src_f']
        # edges = entry['src_edges']
        tgt_vs = entry['tgt_pcd']
        # tgt_f = entry['tgt_f']
        flow = entry['flow']
        if sigma is not None:

            noise = entry[str(sigma)]
            tgt_vs = tgt_vs + noise

        if rot:
            rot_src = entry['rot_src']
            rot_tgt = entry['rot_tgt']
            vs_rot = (np.matmul(rot_src, vs.T)).T
            tgt_vs_rot = (np.matmul(rot_tgt, tgt_vs.T)).T
            tgt_full = vs + flow
            tgt_full_rot = (np.matmul(rot_tgt, tgt_full.T)).T
            flow_rot = tgt_full_rot - vs_rot
            return vs_rot, tgt_vs_rot, flow_rot

    return vs, tgt_vs, flow

class liverTask3(Dataset):

    def __init__(self, config, mode):
        super(liverTask3, self).__init__()
        self.root_path = config.root_path
        self.mode = mode
        self.overfit = config.overfit
        self.file_list = read_dict(self.root_path + config.list)
        # augmentation parameters
        self.max_noise = config.max_noise # random noise
        self.overlap_radius = config.overlap_radius # as noise is added, searching new corr
        self.vox_size = config.vox_size
        self.max_vis = config.max_vis
        self.min_vis = config.min_vis
        self.config = config

        if mode == "test":
            self.test_root = config.test_root
            self.test_list = np.load(config.test_list)['test']

    def __len__(self):
        if self.mode == "train":
            return len(self.file_list)
        else:
            return len(self.test_list)

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
        pc_vox = self.ply2np_vox(pc, voxel_size=self.vox_size, scale=1.0) * m + centroid
        return pc_vox, m

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

    def rand_rot(self, pcd, euler_ab=None):
        if euler_ab is None:
            euler_ab = np.random.rand(3) * np.pi * 2
        rot = Rotation.from_euler('zyx', euler_ab).as_matrix()
        pcd = (np.matmul(rot, pcd.T)).T
        return pcd

    def center(self, points):
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid
        return points_centered

    def get_scale(self, points):

        center_p = points[:, :3] - np.mean(points[:, :3], axis=0)

        return np.max(np.sqrt(np.sum(center_p ** 2, axis=1)))


    def get_input_train(self, index, vis=False, p=None ):
        group = self.file_list[str(index)]

        if self.overfit:
            pair_file = [group[0], random.sample(group, 1)[0]]
        else:
            pair_file = random.sample(group, 2)

        src_file = self.root_path + pair_file[0]
        tgt_file = self.root_path + pair_file[1]

        src_vs, src_ids = load_from_npz(src_file)
        tgt_vs_full, tgt_ids = load_from_npz(tgt_file)
        src_vs, _ = self.norm_vox(src_vs)
        tgt_vs_full, _ = self.norm_vox(tgt_vs_full)

        # if self.grid:
        #     src_pcd = src_vs[src_ids, :]
        #     src_grid_deform = tgt_vs_full[src_ids, :]
        #     tgt_grid_full = tgt_vs_full[tgt_ids, :]
        # else:
        src_pcd = src_vs
        src_grid_deform = tgt_vs_full
        tgt_grid_full = tgt_vs_full

        # random crop
        if p is None:
            p = self.min_vis + (self.max_vis-self.min_vis)*np.random.rand(1)[0]
            if p > 1:
                p = 1.0

        tgt_pcd, mask, rand_xyz = self.crop(tgt_grid_full, p)

        # random noise
        sigma = np.random.rand(1)[0] * self.max_noise
        tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * sigma

        # search corr
        m = self.get_scale(src_grid_deform)
        correspondences = get_correspondences_n(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd),
                                              self.overlap_radius*m)

        centroid = np.mean(src_pcd[:, :3], axis=0)

        m = np.max(np.sqrt(np.sum((src_pcd - centroid) ** 2, axis=1)))

        src_pcd = (src_pcd - centroid) / m
        tgt_pcd = (tgt_pcd - centroid) / m
        trans = -np.mean(tgt_pcd[:, :3], axis=0)
        tgt_pcd += trans

        # get transformation and point cloud
        rot = np.zeros([3, 3])
        rot[0, 0] = 1
        rot[1, 1] = 1
        rot[2, 2] = 1
        trans = trans.T #np.zeros([3, 1])

        # rotate the point cloud
        euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
        rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
        if (np.random.rand(1)[0] > 0.5):
            src_pcd = np.matmul(rot_ab, src_pcd.T).T
            rot = np.matmul(rot, rot_ab.T)
        else:
            tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
            rot = np.matmul(rot_ab, rot)
            trans = np.matmul(rot_ab, trans)

            # src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise

        if trans.ndim == 1:
            trans = trans[:, None]

        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        if vis:
            src_xyz_pred = (np.matmul(rot, src_pcd.T) + trans).T
            vis_pc_marker(src_xyz_pred, None, tgt_pcd, None)



        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)


        if vis:
            print("num corr",len(correspondences))
            print("vis ratio",p)
            scale_factor = 0.013
            viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd+1, correspondences.T, f_src_pcd=None, f_tgt_pcd=None,
                                                scale_factor=scale_factor)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, \
            rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

    def get_test_sample(self, index, scale=1.0, sigma=None, rot=True):
           sample_name = self.test_list[index]
           src_raw, tgt_raw, flow = load_test_from_npz(self.test_root + sample_name, sigma, rot)

           src_deform = src_raw + flow

           return src_deform/scale, src_raw/scale, tgt_raw/scale

    def get_input_test(self, index, vis=False, sigma=None, rot=True):

        src_deform, src_raw, tgt_raw = self.get_test_sample(index, sigma=sigma, rot=rot)


        correspondences = get_correspondences_n(to_o3d_pcd(src_deform), to_o3d_pcd(tgt_raw),
                                                self.overlap_radius)
        src_pcd = self.center(src_raw)
        m = np.max(np.sqrt(np.sum(src_pcd ** 2, axis=1)))
        src_pcd = src_pcd / m
        tgt_pcd = self.center(tgt_raw) / m
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = np.zeros([3, 3])
        trans = np.zeros([3, 1])

        if vis:
            print(len(correspondences))
            scale_factor = 0.013
            viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd + 1, correspondences.T, f_src_pcd=None,
                                                f_tgt_pcd=None,
                                                scale_factor=scale_factor)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, \
            rot, trans, correspondences, src_raw, tgt_raw, torch.ones(1)



    def __getitem__(self, index, vis=False, sigma=None,p=None):

        if self.mode =="train":
             src_grid, tgt_pcd, src_feats, tgt_feats, rot, trans, \
                 correspondences, src_grid, tgt_pcd, sample = self.get_input_train(index, vis=vis,p=p)
             return src_grid, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_grid, tgt_pcd, sample
        else:
            return self.get_input_test(index, vis=vis, sigma=sigma)




def exm_train():
    from easydict import EasyDict as edict
    from dataloader import calibrate_neighbors, collate_fn_descriptor
    from configs.models import architectures
    from lib.util import load_config

    config_path = "/home/yzx/yzx/Deformable_Registration/LiverMatch/configs/liver_task3_grid_on.yaml"
    config = load_config(config_path)
    config = edict(config)

    dataset = liverTask3(config, "train")
    # data.get_input_train(0, vis=True, p=0.18)
    data = dataset.__getitem__(0, vis=True)

    config.architecture = architectures[config.model_name]

    neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn_descriptor)
    print(neighborhood_limits)

if __name__ == '__main__':

    from easydict import EasyDict as edict
    from dataloader import calibrate_neighbors, collate_fn_descriptor
    from configs.models import architectures
    from lib.util import load_config

    config_path = "/configs/new_task3_004_002.yaml"
    config = load_config(config_path)
    config = edict(config)

    dataset = liverTask3(config, "train")
    # data.get_input_train(0, vis=True, p=0.18)
    data = dataset.__getitem__(1, vis=True, sigma=0,p=0.2)

    config.architecture = architectures[config.model_name]

    neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn_descriptor)
    print(neighborhood_limits)


