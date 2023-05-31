import numpy as np
import trimesh
import os
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
import vgtk.pc as pctk
import vgtk.point3d as p3dtk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np, label_relative_rotation_np, label_relative_rotation_simple
from scipy.spatial.transform import Rotation as sciR

import matplotlib.pyplot as plt
import point_cloud_utils as pcu
import math
import time

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def random_point_dropout(pc, max_dropout_ratio=0.875, rng=None):
    # for b in range(batch_pc.shape[0]):
    if rng is None:
        dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
        drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
        # print ('use random drop', len(drop_idx))
    else:
        dropout_ratio = rng.random()*max_dropout_ratio # 0~0.875    
        drop_idx = np.where(rng.random((pc.shape[0]))<=dropout_ratio)[0]

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc   

def translate_pointcloud(pointcloud, rng=None):
    if rng is None:
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    else:
        xyz1 = rng.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = rng.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02, rng=None):
    N, C = pointcloud.shape
    if rng is None:
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype('float32')
    else:
         pointcloud += np.clip(sigma * rng.standard_normal((N, C)), -1*clip, clip).astype('float32')
    return pointcloud

def rotate_pointcloud(pointcloud, rng=None):
    if rng is None:
        theta = np.pi*2 * np.random.uniform()
    else:
        theta = np.pi*2 * rng.uniform()
    
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],
                    [np.sin(theta),np.cos(theta),0],
                    [0,0,1]]).astype('float32')
    pointcloud = Rz.dot(pointcloud.T).T
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).astype('float32')
    # pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud, Rz

def rotate_so3_pointcloud(pointcloud, rng=None):
    """ Randomly rotate the point clouds uniformly
        https://math.stackexchange.com/a/442423
        Input:
          BxNx3 array, point clouds
        Return:
          BxNx3 array, point clouds
    """
    if rng is None:
        rs = np.random.rand(3)
    else:
        rs = rng.random(3)
    angle_z1 = np.arccos(2 * rs[0] - 1)
    angle_y = np.pi*2 * rs[1]
    angle_z2 = np.pi*2 * rs[2]
    Rz1 = np.array([[np.cos(angle_z1),-np.sin(angle_z1),0],
                    [np.sin(angle_z1),np.cos(angle_z1),0],
                    [0,0,1]])
    Ry = np.array([[np.cos(angle_y),0,np.sin(angle_y)],
                    [0,1,0],
                    [-np.sin(angle_y),0,np.cos(angle_y)]])
    Rz2 = np.array([[np.cos(angle_z2),-np.sin(angle_z2),0],
                    [np.sin(angle_z2),np.cos(angle_z2),0],
                    [0,0,1]])
    R = np.dot(Rz1, np.dot(Ry,Rz2)).astype('float32')
    pointcloud = R.dot(pointcloud.T).T
    return pointcloud, R

def rotate_perturbation_point_cloud(pointcloud, angle_sigma=0.06, angle_clip=0.18, rng=None):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, point clouds
        Return:
          BxNx3 array, point clouds
    """
    if rng is None:
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    else:
        angles = np.clip(angle_sigma*rng.standard_normal(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx)).astype('float32')
    pointcloud = R.dot(pointcloud.T).T
    return pointcloud


class Dataloader_ModelNet40_VoxelSmooth(data.Dataset):
    def __init__(self, opt, mode=None, test_aug=False, rot='so3'):
        """For classification task. """
        super(Dataloader_ModelNet40_VoxelSmooth, self).__init__()
        self.opt = opt

        self.shift = self.opt.shift
        self.jitter = self.opt.jitter
        self.dropout = self.opt.dropout_pt
        self.test_aug = test_aug
        self.rot = rot    # None, 'z', 'perturb', 'so3'
        self.rng = np.random.default_rng(0)

        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        if self.opt.model.kanchor == 12:
            self.anchors = L.get_anchorsV()
            self.trace_idx_ori, self.trace_idx_rot = L.get_relativeV_index()
        else:
            self.anchors = L.get_anchors(self.opt.model.kanchor)

        cats = os.listdir(opt.dataset_path)

        path0 = os.path.join(opt.dataset_path, cats[0], self.mode)
        assert os.path.exists(path0), path0
        self.voxeldir = opt.dataset_path.replace('EvenAlignedModelNet40PC', 'ModelNet40VoxelRot')
        self.labeldir = opt.dataset_path
        self.meshdir = opt.dataset_path.replace('EvenAlignedModelNet40PC', 'modelnet_manual_align')

        # path0 = [os.path.join(self.voxeldir, cats[0], 'train', f"{cats[0]}_0001_{i}.npz") for i in range(4)]
        # self.gen_sdf = not all([os.path.exists(pathi) for pathi in path0]) 
        self.gen_sdf = False
        
        if self.gen_sdf:
            self.dataset_path = self.meshdir
            ext = '.off'
            if self.mode == 'testR':
                mode = 'test'
            else:
                mode = self.mode
        else:
            self.dataset_path = self.voxeldir
            if self.mode == 'testR':
                ext = '_0.npz'
            else:
                ext = '_0.npz'
            mode = self.mode

        # self.all_data = ['/scratch/hpeng_root/hpeng1/minghanz/EPN_data/modelnet_manual_align/chair/train/chair_0087.off']
        
        self.all_data = []
        for cat in cats:
            for fn in glob.glob(os.path.join(self.dataset_path, cat, mode, f"*{ext}")):
                self.all_data.append(fn)

        print("[Dataloader] : Training dataset size:", len(self.all_data))
        
        self.sig = self.opt.sigma
        print(f"[Dataloader]: self.sig={self.sig}")
        if self.opt.no_augmentation:
            print("[Dataloader]: USING ALIGNED MODELNET LOADER!")
        else:
            print("[Dataloader]: USING ROTATED MODELNET LOADER!")

        if self.gen_sdf:
            print("[Dataloader]: NEED TO GENERATE SDFS!")
            xs = np.linspace(-1,1,33, endpoint=False) +1/33
            xv, yv, zv = np.meshgrid(xs, xs, xs)    # 33*33*33, 33*33*33, 33*33*33
            xyzv = np.stack([xv, yv, zv], axis=-1)  # 33*33*33*3
            self.grid = xyzv.reshape(-1, 3) # [33*33*33]*3
        else:
            print("[Dataloader]: SDFS FOUND!")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        if self.gen_sdf:
            data_path_voxel_dir = self.all_data[index].replace(self.dataset_path, self.voxeldir)
            if self.mode == 'testR':
                data_path_voxel_dir = data_path_voxel_dir.replace('test/', 'testR/')
            data_dir = os.path.dirname(data_path_voxel_dir)
            os.makedirs(data_dir, exist_ok=True)

            data_path_label_dir = self.all_data[index].replace(self.dataset_path, self.labeldir)
            # data_path_mesh_dir = self.all_data[index].replace(self.dataset_path, self.meshdir)
            if self.mode == 'testR':
                data_path_label_dir = data_path_label_dir.replace('test/', 'testR/')

            if self.mode == 'testR':
                sdf_path = data_path_voxel_dir[:-4] + '.npz'
                # if os.path.exists(sdf_path):
                #     return dict(flag=False)
                path_to_mat = data_path_label_dir[:-3]+'mat'
                data = sio.loadmat(path_to_mat)
                label = data['label']
                fn = data['name']
                R = data['R']
                v, f = pcu.load_mesh_vf(self.all_data[index])
                v = p3dtk.normalize_np(v.T)
                v = np.ascontiguousarray(v.T) # n*3

                # v, R = pctk.rotate_point_cloud(v, R)
                # v = np.ascontiguousarray(v) # n*3
                # # Compute the sdf, the index of the closest face in the mesh, and the barycentric coordinates of
                # # closest point on the mesh, for each point in pts
                # sdfs, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(self.grid, v, f)
                # sdfs = sdfs.reshape(33,33,33)
                
                # np.savez(sdf_path, sdfs=sdfs, R=R, label=label, fn=fn)

                R0 = np.identity(3)
                v_z, R_z = rotate_pointcloud(v)
                v_z = np.ascontiguousarray(v_z) # n*3
                v_r, R = rotate_so3_pointcloud(v)
                v_r = np.ascontiguousarray(v_r) # n*3

                sdfs, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(self.grid, v, f)
                sdfs = sdfs.reshape(33,33,33)
                sdf_path = data_path_voxel_dir[:-4] + '_0.npz'
                np.savez(sdf_path, sdfs=sdfs, R=R0, label=label, fn=fn)

                sdfs_z, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(self.grid, v_z, f)
                sdfs_z = sdfs_z.reshape(33,33,33)
                sdf_path = data_path_voxel_dir[:-4] + '_z.npz'
                np.savez(sdf_path, sdfs=sdfs_z, R=R_z, label=label, fn=fn)

                sdfs_r, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(self.grid, v_r, f)
                sdfs_r = sdfs_r.reshape(33,33,33)
                sdf_path = data_path_voxel_dir[:-4] + '_r.npz'
                np.savez(sdf_path, sdfs=sdfs_r, R=R, label=label, fn=fn)

            else:
                sdf_path = data_path_voxel_dir[:-4] + '_0.npz'
                # if os.path.exists(sdf_path):
                #     return dict(flag=False)
                path_to_mat = data_path_label_dir[:-3]+'mat'
                data = sio.loadmat(path_to_mat)
                label = data['label']
                fn = data['name']
                v, f = pcu.load_mesh_vf(self.all_data[index])
                v = p3dtk.normalize_np(v.T)
                v = np.ascontiguousarray(v.T) # n*3

                for ii in range(1,4):
                    v_z, R = rotate_pointcloud(v)
                    v_z = np.ascontiguousarray(v_z) # n*3
                    sdfs, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(self.grid, v_z, f)
                    sdfs = sdfs.reshape(33,33,33)
                    sdf_path = data_path_voxel_dir[:-4] + f'_z{ii}.npz'
                    np.savez(sdf_path, sdfs=sdfs, R=R, label=label, fn=fn)

                # # Compute the sdf, the index of the closest face in the mesh, and the barycentric coordinates of
                # # closest point on the mesh, for each point in pts
                # sdfs, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(self.grid, v, f)
                # sdfs = sdfs.reshape(33,33,33)
                # sdf_path = data_path_voxel_dir[:-4] + '_0.npz'
                # R = np.identity(3)
                # np.savez(sdf_path, sdfs=sdfs, R=R, label=label, fn=fn)

                # for ii in range(1,4):
                #     v_rot, R = pctk.rotate_point_cloud(v)
                #     v_rot = np.ascontiguousarray(v_rot) # n*3
                #     sdfs, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(self.grid, v_rot, f)
                #     sdfs = sdfs.reshape(33,33,33)
                #     sdf_path = data_path_voxel_dir[:-4] + f'_{ii}.npz'
                #     np.savez(sdf_path, sdfs=sdfs, R=R, label=label, fn=fn)
        else:
            if self.mode == 'testR':
                if self.rot is None:
                    data_path = self.all_data[index]
                elif self.rot == 'z':
                    data_path = self.all_data[index][:-6] + f'_z.npz'
                else:
                    assert self.rot == 'so3', self.rot
                    data_path = self.all_data[index][:-6] + f'_r.npz'
            else:
                # idx_rot = 0 if self.opt.no_augmentation else np.random.randint(1,4)
                # data_path = self.all_data[index][:-6] + f'_{idx_rot}.npz'
                if self.rot is None:
                    data_path = self.all_data[index]
                elif self.rot == 'z':
                    idx_rot = np.random.randint(1,4)
                    data_path = self.all_data[index][:-6] + f'_z{idx_rot}.npz'
                else:
                    assert self.rot == 'so3', self.rot
                    idx_rot = np.random.randint(1,4)
                    data_path = self.all_data[index][:-6] + f'_{idx_rot}.npz'

            try:
                d = np.load(data_path)
            except Exception as e:
                print(data_path)
                raise Exception(f"data_path:{data_path}, {e}")
                # print(d)
            sdfs = d['sdfs']
            label = d['label']
            fn = d['fn']
            R = d['R']

        sdfs = np.clip(sdfs, 0, None)
        mu = 0
        # sig = 0.06 # (2/33, dim of one voxel)
        occs = gaussian(sdfs, mu, self.sig)
        occs = occs.reshape(1, *occs.shape)

        _, R_label, R0 = rotation_distance_np(R, self.anchors)

        in_dict = {'occ':torch.from_numpy(occs.astype(np.float32)),
                'label':torch.from_numpy(label.flatten()).long(),
                'fn': fn[0],
                'R': R,
                'R_label': torch.Tensor([R_label]).long(),
               }
        return in_dict

class Dataloader_ModelNet40(data.Dataset):
    def __init__(self, opt, mode=None, test_aug=False, rot='so3'):
        """For classification task. """
        super(Dataloader_ModelNet40, self).__init__()
        self.opt = opt

        self.shift = self.opt.shift
        self.jitter = self.opt.jitter
        self.dropout = self.opt.dropout_pt
        self.test_aug = test_aug
        self.rot = rot    # None, 'z', 'perturb', 'so3'
        self.rng = np.random.default_rng(0)

        self.train_frac = self.opt.train_frac

        # 'train' or 'eval' or 'testR
        self.mode = opt.mode if mode is None else mode
        # if self.mode == 'train':
        #     self.use_augmentation = not self.opt.no_augmentation
        # else:
        #     self.use_augmentation = True

        if self.opt.model.kanchor == 12:
            self.anchors = L.get_anchorsV()
            self.trace_idx_ori, self.trace_idx_rot = L.get_relativeV_index()
        else:
            self.anchors = L.get_anchors(self.opt.model.kanchor)

        cats = os.listdir(opt.dataset_path)

        self.dataset_path = opt.dataset_path
        self.all_data = []
        for cat in cats:
            fns = glob.glob(os.path.join(opt.dataset_path, cat, self.mode, "*.mat"))
            if self.mode == 'train' and self.train_frac < 1:
                use_n = int(math.ceil(len(fns) * self.train_frac))
                fns = fns[:use_n]
            for fn in fns:
                self.all_data.append(fn)

        print("[Dataloader {}] : Training dataset size:".format(self.mode), len(self.all_data))

        # if not self.use_augmentation:
        #     print("[Dataloader {}]: USING ALIGNED MODELNET LOADER!".format(self.mode))
        # else:
        #     print("[Dataloader {}]: USING ROTATED MODELNET LOADER!".format(self.mode))

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = sio.loadmat(self.all_data[index])
    
        if self.mode == 'train':
            _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)
        else:
            pc = data['pc']
            # print('pc.shape', pc.shape)   # testR: 1024*3
    
        pc = p3dtk.normalize_np(pc.T)
        pc = pc.T

        R = np.eye(3)
        R_label = 29

        if self.mode == 'train' or self.test_aug:
            if self.shift:
                pc = translate_pointcloud(pc) #self.rng
            if self.jitter:
                pc = jitter_pointcloud(pc)    #self.rng
            if self.dropout:
                pc = random_point_dropout(pc) #self.rng
            np.random.shuffle(pc)

        if self.rot == 'z':
            pc, R = rotate_pointcloud(pc) #self.rng
        elif self.rot == 'so3':
            pc, R = rotate_so3_pointcloud(pc) #self.rng
        elif self.rot == 'ico':
            # r_idx = self.rng.integers(60)
            r_idx = np.random.randint(60)
            R = self.anchors[r_idx]
            pc = R.dot(pc.T).T
        else:
            assert self.rot is None, self.rot

        # if self.use_augmentation:
        #     if 'R' in data.keys() and self.mode != 'train':
        #         pc, R = pctk.rotate_point_cloud(pc, data['R'])
        #     else:
        #         pc, R = pctk.rotate_point_cloud(pc)

        _, R_label, R0 = rotation_distance_np(R, self.anchors)
        # if self.rot == 'ico':
        #     print('r_idx', r_idx, 'R_label', R_label, flush=True)
        #     print('R', R, 'R0', R0[r_idx], flush=True)

        if self.opt.model.kanchor == 12:
            if self.opt.train_loss.anchor_ab_loss:
                label_anchor_aligned = np.eye(12)
                trace_idx_rot_true = self.trace_idx_rot[R_label]    # 12
                label_anchor_aligned = label_anchor_aligned[trace_idx_rot_true] # 12*12
                assert label_anchor_aligned.sum() == 12, "label_anchor_aligned: \n{}".format(label_anchor_aligned) # 130=12 + 59*2
                
                if self.opt.train_loss.cross_ab:
                    if self.opt.train_loss.cross_ab_T:
                        label_anchor_aligned = label_anchor_aligned.T
                    label_anchor_aligned = label_anchor_aligned.argmax(0)   # 12
                else:
                    label_anchor_aligned = label_anchor_aligned.astype(np.float32)
            else:
                trace_idx_ori_true = self.trace_idx_ori[[R_label]]    # 1*12
                label_anchor_aligned = self.trace_idx_ori == trace_idx_ori_true # 60*12
                assert label_anchor_aligned.sum() == 60, "label_anchor_aligned: {} \n{}".format(label_anchor_aligned.sum(), label_anchor_aligned)
                label_anchor_aligned = label_anchor_aligned.astype(np.float32)
            
            # if self.flag == 'rotation':
            #     R = R0

        in_dict = {'pc':torch.from_numpy(pc.astype(np.float32)),
                'label':torch.from_numpy(data['label'].flatten()).long(),
                'fn': data['name'][0],
                'R': R,
                'R_label': torch.Tensor([R_label]).long(),
               }
               
        if self.opt.model.kanchor == 12:
            in_dict['anchor_label'] = torch.from_numpy(label_anchor_aligned)
        return in_dict

class Dataloader_ModelNet40Alignment(data.Dataset):
    def __init__(self, opt, mode=None):
        """For relative rotation alignment task. """
        super(Dataloader_ModelNet40Alignment, self).__init__()
        self.opt = opt

        self.train_frac = self.opt.train_frac
        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        # attention method: 'attention | rotation | permutation'
        if self.opt.model.kanchor == 12:
            self.anchors = L.get_anchorsV()
            self.trace_idx_ori, self.trace_idx_rot = L.get_relativeV_index()
        else:
            self.anchors = L.get_anchors(self.opt.model.kanchor)

        # if self.flag == 'rotation':
        #     cats = ['airplane']
        #     print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")
        # else:
        #     cats = os.listdir(opt.dataset_path)

        cats = ['airplane']
        # cats = ['chair']
        # cats = ['airplane','chair','car']
        print(f"[Dataloader]: USING ONLY THE {cats} CATEGORY!!")
        # print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")

        self.dataset_path = opt.dataset_path
        self.all_data = []
        for cat in cats:
            fns = glob.glob(os.path.join(opt.dataset_path, cat, self.mode, "*.mat"))
            if self.mode == 'train' and self.train_frac < 1:
                use_n = int(math.ceil(len(fns) * self.train_frac))
                fns = fns[:use_n]
            for fn in fns:
                self.all_data.append(fn)
        print("[Dataloader] : Training dataset size:", len(self.all_data))


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = sio.loadmat(self.all_data[index])
        _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)  # data['pc']: 1.1k-3k

        # normalization
        pc = p3dtk.normalize_np(pc.T)
        pc = pc.T

        # R = np.eye(3)
        # R_label = 29

        # source shape
        # if 'R' in data.keys() and self.mode != 'train':
        #     pc_src, R_src = pctk.rotate_point_cloud(pc, data['R'])
        # else:
        #     pc_src, R_src = pctk.rotate_point_cloud(pc)

        ### pc_src.T = R_src * pc.T (3*N)
        if self.opt.debug_mode == 'check_equiv':
            i = np.random.randint(60)
            pc_src, R_src = pctk.rotate_point_cloud(pc, self.anchors[i])    # tmp!!!!
        else:
            pc_src, R_src = pctk.rotate_point_cloud(pc)
        ### notice that the seed of the Trainer is given, thus the sampling is deterministic

        # target shape

        # pc_tgt, R_tgt = pctk.rotate_point_cloud(pc)
        pc_tgt = pc

        # if self.mode == 'test':
        #     data['R'] = R
        #     output_path = os.path.join(self.dataset_path, data['cat'][0], 'testR')
        #     os.makedirs(output_path,exist_ok=True)
        #     sio.savemat(os.path.join(output_path, data['name'][0] + '.mat'), data)
        # _, R_label, R0 = rotation_distance_np(R, self.anchors)

        # T = R_src @ R_tgt.T
        T = R_src # @ R_tgt.T

        # RR_regress = np.einsum('abc,bj,ijk -> aick', self.anchors, T, self.anchors)
        # R_label = np.argmax(np.einsum('abii->ab', RR_regress),axis=1)
        # idxs = np.vstack([np.arange(R_label.shape[0]), R_label]).T
        # R = RR_regress[idxs[:,0], idxs[:,1]]
        if self.opt.model.kanchor == 12:
            R, R_label = label_relative_rotation_simple(self.anchors, T, self.opt.rot_ref_tgt)
            if self.opt.rot_ref_tgt:
                trace_idx_rot_true = self.trace_idx_ori[[R_label]]    # 1*12 
                label_anchor_aligned = self.trace_idx_ori == trace_idx_rot_true # 60*12
            else:
                trace_idx_rot_true = self.trace_idx_rot[[R_label]]    # 1*12
                label_anchor_aligned = self.trace_idx_rot == trace_idx_rot_true # 60*12
        else:
            R, R_label = label_relative_rotation_np(self.anchors, T)

        pc_tensor = np.stack([pc_src, pc_tgt])

        in_dict =  {'pc':torch.from_numpy(pc_tensor.astype(np.float32)),
                'fn': data['name'][0],
                'T' : torch.from_numpy(T.astype(np.float32)),
                'R': torch.from_numpy(R.astype(np.float32)),
                'R_label': torch.Tensor(np.array([R_label])).long(),
               }
        if self.opt.model.kanchor == 12:
            in_dict['anchor_label'] = torch.from_numpy(label_anchor_aligned.astype(np.float32))

        # ######## visualize pc_tgt and pc_src
        # fig = plt.figure()

        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(pc_tgt[:,0],pc_tgt[:,1],pc_tgt[:,2], marker=".", s=2)
        # ax.set_axis_off()

        # # plt.show()
        # plt.savefig("fig_tgt_{:02d}.png".format(index))
        # plt.close()

        # fig = plt.figure()

        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(pc_src[:,0],pc_src[:,1],pc_src[:,2], marker=".", s=2)
        # ax.set_axis_off()

        # # plt.show()
        # plt.savefig("fig_src_{:02d}.png".format(index))
        # plt.close()
        # ######### end visualize
        return in_dict
