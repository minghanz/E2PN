import math
import os
from turtle import forward
import numpy as np
import time
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vgtk.spconv import SphericalPointCloud
import vgtk.pc as pctk
from . import functional as L

# BasicSO3Conv = BasicZPConv

KERNEL_CONDENSE_RATIO = 0.7

def sort_level_vs(v_idxs, vs):
    idx_l1_0 = v_idxs[0]
    idx_l1_sorted = []
    idx_l1_sorted.append(idx_l1_0)
    vs_l1 = vs[v_idxs][:, :2]  # only take the xy plane
    vs_l1_0 = vs_l1[[0]]    # 1*2
    dtheta = 2 * np.pi / 5
    ct = np.cos(dtheta)
    st = np.sin(dtheta)
    rotmat = np.array([[ct, -st], [st, ct]], dtype=np.float32)
    for _ in range(4):
        vs_l1_0 = rotmat.dot(vs_l1_0.T).T   # 1*2
        diff = vs_l1 - vs_l1_0  # 5*2
        diff_l1 = np.abs(diff).sum(1)
        iidx_cur = np.argmin(diff_l1)
        idx_cur = v_idxs[iidx_cur]
        idx_l1_sorted.append(idx_cur)
    idx_l1_sorted = np.array(idx_l1_sorted)
    return idx_l1_sorted

class BasicS2Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, anchor_size, debug=False):
        """Linear layer projecting features aggregated at the kernel points to the centers.
        The weight matrix in this implementation does not satisfy the equivariant constraint. 
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super(BasicS2Conv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.anchor_size = anchor_size

        assert self.kernel_size == 13, f"kernel_size {kernel_size} not implemented"
        W = torch.empty(self.dim_out, self.dim_in, 5, anchor_size, dtype=torch.float32)      # c2, c1, 5, a
        nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
        # W = W.view(self.dim_out, self.dim_in*5)
        self.register_parameter('W', nn.Parameter(W))

        ### permute the weights under rotations
        trace_idx_ori, trace_idx_rot = L.get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12
        # trace_idxv_ori = trace_idxv_ori.transpose(1,0)  # 12(indices on s2)*12(rotation anchors)
        # trace_idxv_rot = trace_idxv_rot.transpose(1,0)  # 12*12

        # vertices = np.concatenate([kernels, np.zeros_like(kernels[[0]])], axis=0) # 13,3
        trace_idxv_ori = np.concatenate([trace_idx_ori,np.ones_like(trace_idx_ori[:, [0]])*12],axis=1)   # 12(na)*13(nk)
        trace_idxv_rot = np.concatenate([trace_idx_rot,np.ones_like(trace_idx_rot[:, [0]])*12],axis=1)   # 12*13

        self.register_buffer('trace_idxv_ori', torch.tensor(trace_idxv_ori.astype(np.int64)))   # 12(na)*13(nk)
        self.register_buffer('trace_idxv_rot', torch.tensor(trace_idxv_rot.astype(np.int64)))

        self.register_buffer('trace_idx_ori', torch.tensor(trace_idx_ori.astype(np.int64))) # 12(na rotations)*12(na channels)
        self.register_buffer('trace_idx_rot', torch.tensor(trace_idx_rot.astype(np.int64)))

        ### pick the self, neighbor, level2, opposite, center indices
        vs, v_adjs, v_level2s, v_opps, _ = L.get_icosahedron_vertices() # 12*5, 12*5, 12
        v0_adjs = v_adjs[0]         # 5
        v0_level2s = v_level2s[0]   # 5
        v0_opps = v_opps[0]         # a number
        inv_idxs = torch.empty(kernel_size, dtype=torch.int64)
        inv_idxs[0] = 0
        inv_idxs[v0_adjs] = 1
        inv_idxs[v0_level2s] = 2
        inv_idxs[v0_opps] = 3
        inv_idxs[12] = 4
        inv_idxs = inv_idxs[:, None, None].expand(-1, anchor_size, anchor_size) # k, a(channels), a(rotations)

        trace_idx_rot_k = self.trace_idxv_rot.transpose(0,1)[:,None,:].expand(-1, anchor_size, -1)  # a(rotations),k -> k, a(channels), a(rotations)
        idxs_k = torch.gather(inv_idxs, 0, trace_idx_rot_k)
        
        idxs_a = self.trace_idx_rot.transpose(0,1)[None].expand(kernel_size, -1, -1) # a(rotations),a(channels) -> k, a(channels), a(rotations)

        idxs_k = idxs_k[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        idxs_a = idxs_a[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_k', idxs_k)  #   c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_a', idxs_a)  #   c2, c1, k, a(channels), a(rotations)

    def forward(self, x):
        """x: b,c,k,p,a (a is feature dimension for each anchor, the coordinates are not rotated)"""

        W = self.W[..., None].expand(-1,-1,-1,-1, self.anchor_size)  # c2,c1,5,a(channels),a(rotations)
        W = torch.gather(W, 2, self.idxs_k)  # c2,c1,5,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)
        W = torch.gather(W, 3, self.idxs_a)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels permuted),a(rotations)
        x = torch.einsum("dckar, bckpa->bdpr", W, x)
        return x

class BasicS2ConvV2(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, anchor_size, debug=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.anchor_size = anchor_size

        assert self.kernel_size == 13, f"kernel_size {kernel_size} not implemented"
        W = torch.empty(self.dim_out, self.dim_in, 36, dtype=torch.float32)      # c2, c1, 36(3*4+2*12)
        nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
        # W = W.view(self.dim_out, self.dim_in*5)
        self.register_parameter('W', nn.Parameter(W))

        ### permute the weights under rotations
        trace_idx_ori, trace_idx_rot = L.get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12
        # trace_idxv_ori = trace_idxv_ori.transpose(1,0)  # 12(indices on s2)*12(rotation anchors)
        # trace_idxv_rot = trace_idxv_rot.transpose(1,0)  # 12*12

        # vertices = np.concatenate([kernels, np.zeros_like(kernels[[0]])], axis=0) # 13,3
        trace_idxv_ori = np.concatenate([trace_idx_ori,np.ones_like(trace_idx_ori[:, [0]])*12],axis=1)   # 12(na)*13(nk)
        trace_idxv_rot = np.concatenate([trace_idx_rot,np.ones_like(trace_idx_rot[:, [0]])*12],axis=1)   # 12*13

        self.register_buffer('trace_idxv_ori', torch.tensor(trace_idxv_ori.astype(np.int64)))   # 12(na)*13(nk)
        self.register_buffer('trace_idxv_rot', torch.tensor(trace_idxv_rot.astype(np.int64)))

        self.register_buffer('trace_idx_ori', torch.tensor(trace_idx_ori.astype(np.int64))) # 12(na rotations)*12(na channels)
        self.register_buffer('trace_idx_rot', torch.tensor(trace_idx_rot.astype(np.int64)))

        ### pick the self, neighbor, level2, opposite, center indices
        vs, v_adjs, v_level2s, v_opps, _ = L.get_icosahedron_vertices() # 12*5, 12*5, 12
        v0_adjs = v_adjs[0]         # 5
        v0_level2s = v_level2s[0]   # 5
        v0_opps = v_opps[0]         # a number
        inv_idxs = torch.empty(anchor_size, dtype=torch.int64)
        inv_idxs[0] = 0
        inv_idxs[v0_adjs] = 1
        inv_idxs[v0_level2s] = 2
        inv_idxs[v0_opps] = 3

        v0_adjs_sorted = sort_level_vs(v0_adjs, vs)
        v0_level2s_sorted = sort_level_vs(v0_level2s, vs)
        v0_adjs_sorted = torch.tensor(v0_adjs_sorted, dtype=torch.int64)
        v0_level2s_sorted = torch.tensor(v0_level2s_sorted, dtype=torch.int64)

        idx_map = torch.empty(kernel_size * anchor_size, dtype=torch.int64) # each element is an index in the range 36
        ### the three kernel points on the z axis
        idx_map[:anchor_size] = inv_idxs
        idx_map[v0_opps*anchor_size:(v0_opps+1)*anchor_size ] = inv_idxs + 4
        idx_map[-anchor_size:] = inv_idxs + 8
        ### the rest kernel points on the 2 rings
        idx_seq = torch.arange(12,24, dtype=torch.int64)
        idx_seq2 = torch.arange(24,36, dtype=torch.int64)
        idx_map[v0_adjs_sorted[0]*anchor_size:(v0_adjs_sorted[0]+1)*anchor_size] = idx_seq
        idx_map[v0_level2s_sorted[0]*anchor_size:(v0_level2s_sorted[0]+1)*anchor_size] = idx_seq2

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[1]*anchor_size:(v0_adjs_sorted[1]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[1]*anchor_size:(v0_level2s_sorted[1]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[2]*anchor_size:(v0_adjs_sorted[2]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[2]*anchor_size:(v0_level2s_sorted[2]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[3]*anchor_size:(v0_adjs_sorted[3]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[3]*anchor_size:(v0_level2s_sorted[3]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[4]*anchor_size:(v0_adjs_sorted[4]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[4]*anchor_size:(v0_level2s_sorted[4]+1)*anchor_size] = idx_seq2[idx_seq_new]
        self.register_buffer('idx_map', idx_map)

        idxs_k = self.trace_idxv_rot.transpose(0,1)[:,None,:].expand(-1, anchor_size, -1)  # a(rotations),k -> k, a(channels), a(rotations)
        
        idxs_a = self.trace_idx_rot.transpose(0,1)[None].expand(kernel_size, -1, -1) # a(rotations),a(channels) -> k, a(channels), a(rotations)

        idxs_k = idxs_k[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        idxs_a = idxs_a[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_k', idxs_k)  #   c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_a', idxs_a)  #   c2, c1, k, a(channels), a(rotations)

    def forward(self, x):
        W = self.W[:,:,self.idx_map].reshape(self.dim_out, self.dim_in, self.kernel_size, self.anchor_size)    #C2,C1,kernel_size * anchor_size
        W = W[..., None].expand(-1,-1,-1,-1, self.anchor_size)
        W = torch.gather(W, 2, self.idxs_k)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)
        W = torch.gather(W, 3, self.idxs_a)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels permuted),a(rotations)
        x = torch.einsum("dckar, bckpa->bdpr", W, x)
        return x

# Basic SO3Conv
# [b, c1, k, p, a] -> [b, c2, p, a]
class BasicSO3Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, debug=False):
        super(BasicSO3Conv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size

        # TODO: initialization argument
        # TODO: add bias

        if debug:
            W = torch.zeros(self.dim_out, self.dim_in*self.kernel_size) + 1
            self.register_buffer('W', W)
        else:
            W = torch.empty(self.dim_out, self.dim_in, self.kernel_size)
            # nn.init.xavier_normal_(W, gain=0.001)
            nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
            # nn.init.normal_(W, mean=0.0, std=0.3)
            W = W.view(self.dim_out, self.dim_in*self.kernel_size)

            self.register_parameter('W', nn.Parameter(W))
            # bias = torch.zeros(self.dim_out) + 1e-3
            # bias = bias.view(1,self.dim_out,1)
            # self.register_parameter('bias', nn.Parameter(bias))

        #self.W = nn.Parameter(torch.Tensor(self.dim_out, self.dim_in*self.kernel_size))

    def forward(self, x):
        bs, np, na = x.shape[0], x.shape[3], x.shape[4]
        x = x.view(bs, self.dim_in*self.kernel_size, np*na)
        x = torch.matmul(self.W, x)

        # x = x + self.bias
        x = x.view(bs, self.dim_out, np, na)
        return x

class KernelPropagation(nn.Module):
    def __init__(self, dim_in, dim_out, n_center, kernel_size, radius, sigma, kanchor=60):
        super(KernelPropagation, self).__init__()

        # get kernel points (ksx3)
        kernels = L.get_sphereical_kernel_points_from_ply(KERNEL_CONDENSE_RATIO * radius, kernel_size)

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors(kanchor)
        # if kpconv:
        #     anchors = anchors[29][None]
        kernels = np.transpose(anchors @ kernels.T, (2,0,1))

        self.radius = radius
        self.sigma = sigma
        self.n_center = n_center

        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('kernels', torch.from_numpy(kernels))

        self.basic_conv = BasicSO3Conv(dim_in, dim_out, kernels.shape[0])


    def _subsample(self, clouds):
        '''
            furthest point sampling
            [b, 3, n_sub, 3] -> [b, 3, n_center]
        '''
        idx, sample_xyz = pctk.furthest_sample(clouds, self.n_center, False)
        return sample_xyz

    def forward(self, frag, clouds):
        '''
        frag (m,3), center (b, 3, n_center), kernels(ks, na, 3)
        ->
        anchor weight (b, 1, ks, nc, na)

        '''
        if clouds.shape[2] == self.n_center:
            centers = clouds
        else:
            centers = self._subsample(clouds)

        wts, nnctn = L.initial_anchor_query(frag, centers, self.kernels, self.radius, self.sigma)

        # normalization!
        wts = wts / (nnctn + 1.0)

        ###################################
        # torch.set_printoptions(sci_mode=False)
        # print('----------------wts------------------------------')
        # print(wts[0,:,16,0])
        # print('---------------mean---------------------------')
        # print(wts[0].mean(-2))
        # print('---------------std----------------------------')
        # print(wts[0].std(-2))
        # print('-----------------------------------------------')
        # import ipdb; ipdb.set_trace()
        ####################################

        feats = self.basic_conv(wts.unsqueeze(1))

        return SphericalPointCloud(centers, feats, self.anchors)


class S2Conv(nn.Module):
    def __init__(self, dim_in, dim_out, 
                kernel_size, # not used for now (always 12 vertices of icosahedron)
                stride,
                radius, 
                sigma, 
                n_neighbor,
                lazy_sample=True, 
                pooling=None, 
                kanchor=12,
                sym_kernel=True) -> None:
        """Point grouping, subsampling, and convolution
        [b, c1, k, p, a] -> [b, c2, p, a] """
        super().__init__()
        # get kernel points
        KERNEL_CONDENSE_RATIO = 0.7
        vertices, v_adjs, v_level2s, v_opps, vRs = L.get_icosahedron_vertices() 
        kernels = vertices * KERNEL_CONDENSE_RATIO * radius
        kernels = np.concatenate([kernels, np.zeros_like(kernels[[0]])], axis=0) # 13,3

        # get so3 anchors (12x3x3 rotation matrices, the section of each element in S2)
        anchors = L.get_anchorsV12()   # 12*3*3

        # # debug only
        # if kanchor == 1:
        #     anchors = anchors[29][None]

        # register hyperparameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernels.shape[0]
        self.stride = stride
        self.radius = radius
        self.sigma = sigma
        self.n_neighbor = n_neighbor
        self.lazy_sample = lazy_sample
        self.pooling = pooling
        self.sym_kernel = sym_kernel
        
        # self.basic_conv = BasicS2Conv(dim_in, dim_out, self.kernel_size, anchors.shape[0])
        self.basic_conv = BasicS2ConvV2(dim_in, dim_out, self.kernel_size, anchors.shape[0])

        self.register_buffer('anchors', torch.tensor(anchors).to(torch.float32))
        self.register_buffer('kernels', torch.tensor(kernels).to(torch.float32))

    def forward(self, x, inter_idx=None, inter_w=None):
        inter_idx, inter_w, xyz, feats, sample_idx = \
            L.inter_so3conv_grouping(x.xyz, x.feats, self.stride, self.n_neighbor,
                                  self.anchors, self.kernels,
                                  self.radius, self.sigma,
                                  inter_idx, inter_w, self.lazy_sample, pooling=self.pooling,
                                  norot=self.sym_kernel)


        # torch.set_printoptions(sci_mode=False)
        # print(feats[0,0,:,16])
        # print("-----------mean -----------------")
        # print(feats[0].mean(-2))
        # print("-----------std -----------------")
        # print(feats[0].std(-2))
        # import ipdb; ipdb.set_trace()
        feats = self.basic_conv(feats)

        return inter_idx, inter_w, sample_idx, SphericalPointCloud(xyz, feats, self.anchors)

class S2InterSO3Conv(nn.Module):
    pass
class S2IntraSO3Conv(nn.Module):
    pass

# A single Inter SO3Conv
# [b, c1, p1, a] -> [b, c1, k, p2, a] -> [b, c2, p2, a]
class InterSO3Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, sigma, n_neighbor,
                 lazy_sample=True, pooling=None, kanchor=60):
        super(InterSO3Conv, self).__init__()

        # get kernel points
        kernels = L.get_sphereical_kernel_points_from_ply(KERNEL_CONDENSE_RATIO * radius, kernel_size)

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors(kanchor)

        # # debug only
        # if kanchor == 1:
        #     anchors = anchors[29][None]

        # register hyperparameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernels.shape[0]
        self.stride = stride
        self.radius = radius
        self.sigma = sigma
        self.n_neighbor = n_neighbor
        self.lazy_sample = lazy_sample
        self.pooling = pooling

        self.basic_conv = BasicSO3Conv(dim_in, dim_out, self.kernel_size)

        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('kernels', torch.from_numpy(kernels))

    def forward(self, x, inter_idx=None, inter_w=None):
        inter_idx, inter_w, xyz, feats, sample_idx = \
            L.inter_so3conv_grouping(x.xyz, x.feats, self.stride, self.n_neighbor,
                                  self.anchors, self.kernels,
                                  self.radius, self.sigma,
                                  inter_idx, inter_w, self.lazy_sample, pooling=self.pooling)


        # torch.set_printoptions(sci_mode=False)
        # print(feats[0,0,:,16])
        # print("-----------mean -----------------")
        # print(feats[0].mean(-2))
        # print("-----------std -----------------")
        # print(feats[0].std(-2))
        # import ipdb; ipdb.set_trace()
        feats = self.basic_conv(feats)

        return inter_idx, inter_w, sample_idx, SphericalPointCloud(xyz, feats, self.anchors)


class IntraSO3Conv(nn.Module):
    '''
    Note: only use intra conv when kanchor=60

    '''
    def __init__(self, dim_in, dim_out):
        super(IntraSO3Conv, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = L.get_anchors()
        # get so3 convolution index (precomputed 60x12 indexing)
        intra_idx = L.get_intra_idx()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = intra_idx.shape[1]
        self.basic_conv = BasicSO3Conv(dim_in, dim_out, self.kernel_size)
        self.register_buffer('anchors', torch.from_numpy(anchors))
        self.register_buffer('intra_idx', torch.from_numpy(intra_idx).long())

    def forward(self, x):
        feats = L.intra_so3conv_grouping(self.intra_idx, x.feats)
        feats = self.basic_conv(feats)
        return SphericalPointCloud(x.xyz, feats, self.anchors)


class PointnetSO3Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kanchor=60, drop_xyz=False):
        '''
        equivariant pointnet architecture for a better aggregation of spatial point features
        f (nb, nc, np, na) x xyz (nb, 3, np, na) -> (nb, ncout, na)
        '''
        super(PointnetSO3Conv, self).__init__()

        ### changing this part could make old models fail because of missing keys 'anchors'
        if kanchor == 1:
            self.anchors = None
        elif kanchor == 12:
        # if kanchor == 12:
            # get so3 anchors (12x3x3 rotation matrices)
            anchors = L.get_anchorsV12()   # 12*3*3
            self.register_buffer('anchors', torch.from_numpy(anchors))
        else:
            assert kanchor == 60, kanchor
            # get so3 anchors (60x3x3 rotation matrices)
            anchors = L.get_anchors(kanchor)
            self.register_buffer('anchors', torch.from_numpy(anchors))
        
        ### for quotient anchors, drop_xyz preserves equivariance
        self.drop_xyz = drop_xyz
        if drop_xyz:
            self.dim_in = dim_in
        else:
            self.dim_in = dim_in + 3
            
        self.dim_out = dim_out

        self.embed = nn.Conv2d(self.dim_in, self.dim_out,1)

    def forward(self, x):
        xyz = x.xyz
        feats = x.feats
        nb, nc, np, na = feats.shape

        # normalize xyz
        xyz = xyz - xyz.mean(2,keepdim=True)

        if na == 1:
            feats = torch.cat([x.feats, xyz[...,None]],1)
        else:
            if self.drop_xyz:
                feats = x.feats
            else:
                xyzr = torch.einsum('aji,bjn->bina',self.anchors,xyz)
                feats = torch.cat([x.feats, xyzr],1)

        feats = self.embed(feats)
        # bcpa -> bca
        feats = torch.max(feats,2)[0]
        return feats # nb, nc, na
