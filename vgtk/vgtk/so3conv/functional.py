import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm

# from utils_cuda import _neighbor_query, _spherical_conv
import vgtk
import vgtk.pc as pctk
import vgtk.cuda.zpconv as cuda_zpconv
import vgtk.cuda.gathering as gather
import vgtk.cuda.grouping as cuda_nn

import vgtk.spconv as zpconv


inter_so3conv_feat_grouping = zpconv.inter_zpconv_grouping_naive
batched_index_select = zpconv.batched_index_select

# pc: [nb,np,3] -> feature: [nb,1,np,na]
def get_occupancy_features(pc, n_anchor, use_center=False):
    nb, np, nd = pc.shape
    has_normals = nd == 6

    features = torch.zeros(nb, 1, np, n_anchor) + 1
    features = features.float().to(pc.device)

    if has_normals:
        ns = pc[:,:,3:]
        if n_anchor > 1:
            # anchors = torch.from_numpy(get_anchors())
            features_n = torch.einsum('bni,aij->bjna',ns.anchors)
        else:
            features_n = ns.transpose(1,2)[...,None].contiguous()
        features = torch.cat([features,features_n],1)

    if use_center:
        features[:,:,0,:] = 0.0

    return features


# (x,y,z) points derived from conic parameterization
def get_kernel_points_np(radius, aperature, kernel_size, multiplier=1):
    assert isinstance(kernel_size, int)
    rrange = np.linspace(0, radius, kernel_size, dtype=np.float32)
    kps = []

    for ridx, ri in enumerate(rrange):
        alpharange = zpconv.get_angular_kernel_points_np(aperature, ridx * multiplier + 1)
        for aidx, alpha in enumerate(alpharange):
            r_r = ri * np.tan(alpha)
            thetarange = np.linspace(0, 2 * np.pi, aidx * 2 + 1, endpoint=False, dtype=np.float32)
            xs = r_r * np.cos(thetarange)
            ys = r_r * np.sin(thetarange)
            zs = np.repeat(ri, aidx * 2 + 1)
            kps.append(np.vstack([xs,ys,zs]).T)

    kps = np.vstack(kps)
    return kps

def get_spherical_kernel_points_np(radius, kernel_size, multiplier=3):
    assert isinstance(kernel_size, int)
    rrange = np.linspace(0, radius, kernel_size, dtype=np.float32)
    kps = []

    for ridx, r_i in enumerate(rrange):
        asize = ridx * multiplier + 1
        bsize = ridx * multiplier + 1
        alpharange = np.linspace(0, 2*np.pi, asize, endpoint=False, dtype=np.float32)
        betarange = np.linspace(0, np.pi, bsize, endpoint=True, dtype=np.float32)

        xs = r_i * np.cos(alpharange[:, None]) * np.sin(betarange[None])
        ys = r_i * np.sin(alpharange[:, None]) * np.sin(betarange[None])

        zs = r_i * np.cos(betarange)[None].repeat(asize, axis=0)
        kps.append(np.vstack([xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)]).T)

    kps = np.vstack(kps)
    return kps

def get_sphereical_kernel_points_from_ply(radius, kernel_size):
    assert kernel_size <= 3 and kernel_size > 0
    mapping = {1:24, 2:30, 3:66}
    root = vgtk.__path__[0]
    anchors_path = os.path.join(root, 'data', 'anchors')
    ply_path = os.path.join(anchors_path, f'kpsphere{mapping[kernel_size]:d}.ply')
    ply = pctk.load_ply(ply_path).astype('float32')
    def normalize(pc, radius):
        r = np.sqrt((pc**2).sum(1).max())
        return pc*radius/r
    return normalize(ply, radius)


# initial_anchor_query(
#     at::Tensor centers, //[b, 3, nc]
#     at::Tensor xyz,  //[m, 3]
#     at::Tensor kernel_points, // [ks, na, 3]
#     const float radius, const float sigma)
def initial_anchor_query(frag, centers, kernels, r, sigma):
    return cuda_nn.initial_anchor_query(centers, frag, kernels, r, sigma)


def inter_so3conv_blurring(xyz, feats, n_neighbor, radius, stride,
                           inter_idx=None, lazy_sample=True, radius_expansion=1.0):
    """Sample a set of center points, and blur their feature with those of their neighboring points. 
    Return the blurred feature and coordinate of the center points. """
    if inter_idx is None:
        _, inter_idx, sample_idx, sample_xyz = zpconv.inter_zpconv_grouping_ball(xyz, stride, radius * radius_expansion, n_neighbor, lazy_sample)

    if stride == 1:
        return zpconv.inter_blurring_naive(inter_idx, feats), xyz
    else:
        return zpconv.inter_pooling_naive(inter_idx, sample_idx, feats), sample_xyz

def inter_so3conv_grouping(xyz, feats, stride, n_neighbor,
                          anchors, kernels, radius, sigma,
                          inter_idx=None, inter_w=None, lazy_sample=True,
                          radius_expansion=1.0, pooling=None, norot=False):
    '''Given input points and features, return the set of center points and features aggregated to the kernels centered at them. 
        xyz: [nb, 3, p1] coordinates
        feats: [nb, c_in, p1, na] features
        anchors: [na, 3, 3] rotation matrices
        kernels: [ks, 3] kernel points
        inter_idx: [nb, p2, nn] grouped points, where p2 = p1 / stride
        inter_w: [nb, p2, na, ks, nn] kernel weights:
                    Influences of each neighbor points on each kernel points
                    under the respective SO3 rotations
        return:
            new_feats: [nb, nc, nk, nb, na]
    '''

    if pooling is not None and stride > 1 and feats.shape[1] > 1:
        # Apply low pass blurring before strided conv
        if pooling == 'stride':
            # NOTE: if meanpool replaces stride, nn and radius needs to be matched with the next conv
            pool_stride = stride
            # TODO: REMOVE HARD CODING
            stride_nn = int(n_neighbor * pool_stride**0.5)
            stride = 1      # subsampling is taken care of by pool_stride, thus set stride to 1
        elif pooling == 'no-stride':
            pool_stride = 1
            stride_nn = n_neighbor
        else:
            raise NotImplementedError(f"Pooling mode {pooling} is not implemented!")

        feats, xyz = inter_so3conv_blurring(xyz, feats, stride_nn, radius, pool_stride, inter_idx, lazy_sample)
        inter_idx = None

    if inter_idx is None:
        grouped_xyz, inter_idx, sample_idx, new_xyz = zpconv.inter_zpconv_grouping_ball(xyz, stride,
                                                                         radius * radius_expansion, n_neighbor, lazy_sample)
        if not norot:
            inter_w = inter_so3conv_grouping_anchor(grouped_xyz, anchors, kernels, sigma)
        else:
            inter_w = inter_so3conv_grouping_anchor_norot(grouped_xyz, anchors, kernels, sigma)


        #####################DEBUGDEBUGDEBUGDEBUG####################################
        # print(xyz.shape)
        # xyz_sample = (xyz - xyz.mean(2, keepdim=True))[0]
        # gsample1 = xyz_sample[:,inter_idx[0,12].long()]
        # gsample2 = xyz_sample[:,inter_idx[0,25].long()]
        # gsample3 = xyz_sample[:,inter_idx[0,31].long()]
        # pctk.save_ply('data/gsample2.ply', gsample2.T.cpu().numpy(), c='r')
        # pctk.save_ply('data/gsample3.ply', gsample3.T.cpu().numpy(), c='r')
        # pctk.save_ply('data/xyz.ply', xyz_sample.T.cpu().numpy())

        # for bi in range(new_xyz.shape[0]):
        #     pctk.save_ply(f'vis/gsample{bi}.ply', new_xyz[bi].T.cpu().numpy())
        # # import ipdb; ipdb.set_trace()
        #############################################################################
    else:
        sample_idx = None
        new_xyz = xyz

    feats = zpconv.add_shadow_feature(feats)

    new_feats = inter_so3conv_feat_grouping(inter_idx, inter_w, feats) # [nb, c_in, ks, np, na]

    return inter_idx, inter_w, new_xyz, new_feats, sample_idx

def inter_so3conv_grouping_anchor_norot(grouped_xyz, anchors,
                                  kernels, sigma, interpolate='linear'):
    '''Calculate the weight between each kernel point and neighboring input point, at each location. 
    The dimension of rotational anchors is not here because the kernel points are symmetric to the rotations. 
        grouped_xyz: [b, 3, p2, nn]
        kernels: [nk, 3]
        sigma: float
        (TODO: remove anchors: [na, 3, 3]
        trace_idxv_ori: [12(na)*13(nk)])
        return: [b, p2, nk, nn]
    '''
    # distance: b, p2, nk, nn (save the na dimension due to symmetry)
    t_kernels = kernels.transpose(0,1)[None,:,None,:,None] # nk,3 -> 3,nk -> b,3,p2,nk,nn
    t_gxyz = grouped_xyz[...,None,:] # b,3,p2,nn -> b,3,p2,nk,nn

    if interpolate == 'linear':
        dists = torch.sum((t_gxyz - t_kernels)**2, dim=1)    # b,p2,nk,nn
        # dists = torch.sqrt(torch.sum((t_gxyz - t_rkernels)**2, dim=1))
        inter_w = F.relu(1.0 - dists/sigma, inplace=True)
    else:
        raise NotImplementedError("kernel function %s is not implemented!"%interpolate)

    # ### permute to augment b,p2,nk,nn to b,p2,na,nk,nn
    # trace_idxv_ori = trace_idxv_ori[None,None,...,None] # na,nk -> b,p2,na,nk,nn
    # inter_w = inter_w.unsqueeze(2).expand_as(trace_idxv_ori)  #  b,p2,nk,nn ->  b,p2,1,nk,nn -> b,p2,na,nk,nn
    # inter_w = torch.gather(inter_w, 3, trace_idxv_ori)  # b,p2,na,nk,nn

    return inter_w # b,p2,nk,nn

def inter_so3conv_grouping_anchor(grouped_xyz, anchors,
                                  kernels, sigma, interpolate='linear'):
    '''Calculate the weight between each kernel point and neighboring input point, at each location under each rotation anchors.
        grouped_xyz: [b, 3, p2, nn]
        anchors: [na, 3, 3]
        kernels: [nk, 3]
        sigma: float
        return:  [b, p2, na, ks, nn]
    '''

    # kernel rotations:  3, na, ks
    # [na,3,3] x [3,nk] -> [na,3,nk] -> [3,na,nk]
    rotated_kernels = torch.matmul(anchors, kernels.transpose(0,1)).permute(1,0,2).contiguous()

    # calculate influences: [3, na, ks] x [b, 3, p2, nn] -> [b, p2, na, ks, nn] weights
    t_rkernels = rotated_kernels[None, :, None, :, :, None]
    t_gxyz = grouped_xyz[...,None,None,:]

    if interpolate == 'linear':
        #  b, p2, na, ks, nn
        dists = torch.sum((t_gxyz - t_rkernels)**2, dim=1)
        # dists = torch.sqrt(torch.sum((t_gxyz - t_rkernels)**2, dim=1))
        inter_w = F.relu(1.0 - dists/sigma, inplace=True)
        # inter_w = F.relu(1.0 - dists / (3*sigma)**0.5, inplace=True)

        # torch.set_printoptions(precision=2, sci_mode=False)
        # print('---sigma----')
        # print(sigma)
        # print('---mean distance---')
        # print(dists.mean())
        # print(dists[0,10,0,6])
        # print('---weights---')
        # print(inter_w[0,10,0,6])
        # print('---summary---')
        # summary = torch.sum(inter_w[0,:,0] > 0.1, -1)
        # print(summary.float().mean(0))
        # import ipdb; ipdb.set_trace()
    else:
        raise NotImplementedError("kernel function %s is not implemented!"%interpolate)

    return inter_w


def intra_so3conv_grouping(intra_idx, feature):
    '''
        intra_idx: [na,pnn] so3 neighbors
        feature: [nb, c_in, np, na] features
    '''

    # group features -> [nb, c_in, pnn, np, na]

    nb, c_in, nq, na = feature.shape
    _, pnn = intra_idx.shape

    feature1 = feature.index_select(3, intra_idx.view(-1)).view(nb, c_in, nq, na, pnn)
    grouped_feat =  feature1.permute([0,1,4,2,3]).contiguous()

    # print(torch.sort(grouped_feat[0,0].mean(0)))
    # print(torch.sort(grouped_feat[1,0].mean(0)))
    # print(torch.sort(grouped_feat[0,0,0]))
    # print(torch.sort(grouped_feat[1,0,0]))
    # print(torch.sort(grouped_feat[0,0,1]))
    # print(torch.sort(grouped_feat[1,0,1]))

    # def find_rel(idx1, idx2):
    #     idx1 = idx1.cpu().numpy().squeeze()
    #     idx2 = idx2.cpu().numpy().squeeze()
    #     idx3 = np.zeros_like(idx1)
    #     for i in range(len(idx1)):
    #         idx3[idx2[i]] = idx1[i]
    #     return idx3

    # def comb_rel(idx1, idx2):
    #     idx1 = idx1.cpu().numpy().squeeze()
    #     idx2 = idx2.cpu().numpy().squeeze()
    #     idx3 = np.zeros_like(idx1)
    #     for i in range(len(idx1)):
    #         idx3[i] = idx1[idx2[i]]
    #     return idx3

    # rel01 = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[0,0,1])[1])
    # rel02 = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[1,0,0])[1])
    # rel13 = find_rel(torch.sort(grouped_feat[0,0,1])[1], torch.sort(grouped_feat[1,0,1])[1])
    # rel23 = find_rel(torch.sort(grouped_feat[1,0,0])[1], torch.sort(grouped_feat[1,0,1])[1])

    # rel_in = find_rel(torch.sort(feature[0,0,0])[1], torch.sort(feature[1,0,0])[1])
    # rel_out = find_rel(torch.sort(grouped_feat[0,0,0])[1], torch.sort(grouped_feat[1,0,0])[1])

    # import ipdb; ipdb.set_trace()

    return grouped_feat

# initialize so3 sampling
import vgtk.functional as fr
import os

GAMMA_SIZE = 3
ROOT = vgtk.__path__[0]
ANCHOR_PATH = os.path.join(ROOT, 'data', 'anchors/sphere12.ply')

Rs, R_idx, canonical_relative = fr.icosahedron_so3_trimesh(ANCHOR_PATH, GAMMA_SIZE)


def select_anchor(anchors, k):
    if k == 1:
        return anchors[29][None]
    elif k == 20:
        return anchors[::3]
    elif k == 40:
        return anchors.reshape(20,3,3,3)[:,:2].reshape(-1,3,3)
    else:
        return anchors


def get_anchors(k=60):
    return select_anchor(Rs,k)

def get_intra_idx():
    return R_idx

def get_canonical_relative():
    return canonical_relative

def get_relative_index():
    return fr.get_relativeR_index(Rs)


vs, v_adjs, v_level2s, v_opps, vRs = fr.icosahedron_trimesh_to_vertices(ANCHOR_PATH)    # 12*3, each vertex is of norm 1


def get_anchorsV():
    """return 60*3*3 matrix as rotation anchors determined by the symmetry of icosahedron vertices"""
    return vRs.copy()

def get_anchorsV12():
    """return 12*3*3 matrix as the section (representative rotation) of icosahedron vertices. 
    For each vertex on the sphere (icosahedron) (coset space S2 = SO(3)/SO(2)), 
    pick one rotation as its representation in SO(3), which is also called a section function (G/H -> G)"""
    return vRs.reshape(12, 5, 3, 3)[:,0].copy()    # 12*3*3

def get_icosahedron_vertices():
    return vs.copy(), v_adjs.copy(), v_level2s.copy(), v_opps.copy(), vRs.copy()

def get_relativeV_index():
    """return two 60(rotation anchors)*12(indices on s2) index matrices"""
    trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
    return trace_idx_ori.copy(), trace_idx_rot.copy()

def get_relativeV12_index():
    """return two 12(rotation anchors)*12(indices on s2) index matrices"""
    trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)  # 60(rotation anchors)*12(indices on s2), 60*12
    trace_idx_ori = trace_idx_ori.reshape(12, 5, 12)[:,0]   # 12(rotation anchors)*12(indices on s2)
    trace_idx_rot = trace_idx_rot.reshape(12, 5, 12)[:,0]   # 12(rotation anchors)*12(indices on s2)
    return trace_idx_ori.copy(), trace_idx_rot.copy()

def get_relativeVR_index(full=False):
    """return two 60(rotation anchors)*60(indices on anchor rotations) index matrices,
    or two 60(rotation anchors)*12(indices on the anchor rotations that are sections of icoshedron vertices) index matrices. 
    The latter case is different from get_relativeV_index(), because here the indices are in the range of 60. 
    """
    trace_idx_ori, trace_idx_rot = fr.get_relativeR_index(vRs)  
    # da     # find correspinding original element for each rotated (60,60)
    # bd     # find corresponding rotated element for each original
    trace_idx_rot = trace_idx_rot.transpose(0,1)    # db
    if full:
        return trace_idx_ori.copy(), trace_idx_rot.copy()

    trace_idx_ori = trace_idx_ori.reshape(-1, 12, 5)    # d,12,5
    trace_idx_rot = trace_idx_rot.reshape(-1, 12, 5)
    trace_idx_ori = trace_idx_ori[..., 0]   # d,12
    trace_idx_rot = trace_idx_rot[..., 0]   # d,12
    return trace_idx_ori.copy(), trace_idx_rot.copy()

# np.set_printoptions(threshold=np.inf)
# trace_idx_ori, trace_idx_rot = get_relativeV_index()
# print("trace_idxv_ori", trace_idx_ori, trace_idx_rot)

# trace_idx_ori, trace_idx_rot = get_relative_index()
# print("trace_idxr_ori_full", trace_idx_ori, trace_idx_rot)

# trace_idx_ori, trace_idx_rot = get_relativeVR_index()
# print("trace_idxr_ori", trace_idx_ori, trace_idx_rot)

# print("vs", vs)
# ### inspect that identity is included
# print("vRs", vRs)
# trace_idx_ori, trace_idx_rot = get_relativeVR_index(True)
# print("trace_idxr_ori_full", trace_idx_ori, trace_idx_rot)