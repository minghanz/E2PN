import math
import os
import numpy as np
import time
from collections import namedtuple
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import vgtk.spconv as zptk
import vgtk.so3conv as sptk

# [nb, np, 3] -> [nb, 3, np] x [nb, 1, np, na]
def preprocess_input(x, na, add_center=True):
    has_normals = x.shape[2] == 6
    # add a dummy center point at index zero
    if add_center and not has_normals:
        center = x.mean(1, keepdim=True)
        x = torch.cat((center,x),dim=1)[:,:-1]
    xyz = x[:,:,:3]
    return zptk.SphericalPointCloud(xyz.permute(0,2,1).contiguous(), sptk.get_occupancy_features(x, na, add_center), None)

def get_inter_kernel_size(band):
    return np.arange(band + 1).sum() + 1

def get_intra_kernel_size(band):
    return np.arange(band + 1).sum() + 1

# [b, c1, p, a] -> [b, c1, k, p, a] -> [b, c2, p, a]
class IntraSO3ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out,
                 norm=None, activation='relu', dropout_rate=0):

        super(IntraSO3ConvBlock, self).__init__()

        if norm is not None:
            norm = getattr(nn,norm)
            # norm = nn.InstanceNorm2d

        self.conv = sptk.IntraSO3Conv(dim_in, dim_out)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim__out)

        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        # [b, 3, p] x [b, c1, p]
        x = self.conv(x)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)

        # [b, 3, p] x [b, c2, p]
        return zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class PropagationBlock(nn.Module):
    def __init__(self, params, norm=None, activation='relu', dropout_rate=0):
        super(PropagationBlock, self).__init__()
        self.prop = sptk.KernelPropagation(**params)
        if norm is None:
            norm = nn.InstanceNorm2d #nn.BatchNorm2d
        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)
        self.norm = norm(params['dim_out'], affine=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, frag, clouds):
        x = self.prop(frag, clouds)
        feat = self.norm(x.feats)
        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return zptk.SphericalPointCloud(x.xyz, feat, x.anchors)

class S2ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, sigma, n_neighbor, multiplier, kanchor=12,
                 lazy_sample=None, norm=None, activation='relu', pooling='none', dropout_rate=0,
                 sym_kernel=True) -> None:
        """S2Convolution, normalization, relu, and dropout"""
        super().__init__()

        if lazy_sample is None:
            lazy_sample = True

        if norm is not None:
            norm = getattr(nn,norm)
            
        # if norm is None:
        #     norm = nn.InstanceNorm2d #nn.BatchNorm2d

        pooling_method = None if pooling == 'none' else pooling
        self.conv = sptk.S2Conv(dim_in, dim_out, kernel_size, stride,
                                      radius, sigma, n_neighbor, kanchor=kanchor,
                                      lazy_sample=lazy_sample, pooling=pooling_method, sym_kernel=sym_kernel)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)

        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x, inter_idx=None, inter_w=None):
        input_x = x
        inter_idx, inter_w, sample_idx, x = self.conv(x, inter_idx, inter_w)
        feat = self.norm(x.feats)   # bcpa
        # feat = x.feats

        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return inter_idx, inter_w, sample_idx, zptk.SphericalPointCloud(x.xyz, feat, x.anchors)

# [b, c1, p1, a] -> [b, c1, k, p2, a] -> [b, c2, p2, a]
class InterSO3ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride,
                 radius, sigma, n_neighbor, multiplier, kanchor=60,
                 lazy_sample=None, norm=None, activation='relu', pooling='none', dropout_rate=0):
        super(InterSO3ConvBlock, self).__init__()

        if lazy_sample is None:
            lazy_sample = True

        if norm is not None:
            norm = getattr(nn,norm)
            
        # if norm is None:
        #     norm = nn.InstanceNorm2d #nn.BatchNorm2d

        pooling_method = None if pooling == 'none' else pooling
        self.conv = sptk.InterSO3Conv(dim_in, dim_out, kernel_size, stride,
                                      radius, sigma, n_neighbor, kanchor=kanchor,
                                      lazy_sample=lazy_sample, pooling=pooling_method)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)

        if activation is None:
            self.relu = None
        else:
            self.relu = getattr(F, activation)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x, inter_idx=None, inter_w=None):
        input_x = x
        inter_idx, inter_w, sample_idx, x = self.conv(x, inter_idx, inter_w)
        feat = self.norm(x.feats)
        # feat = x.feats

        if self.relu is not None:
            feat = self.relu(feat)
        if self.training and self.dropout is not None:
            feat = self.dropout(feat)
        return inter_idx, inter_w, sample_idx, zptk.SphericalPointCloud(x.xyz, feat, x.anchors)


class BasicSO3ConvBlock(nn.Module):
    def __init__(self, params):
        super(BasicSO3ConvBlock, self).__init__()

        self.blocks = nn.ModuleList()
        self.layer_types = []
        for param in params:
            if param['type'] == 'intra_block':
                conv = IntraSO3ConvBlock(**param['args'])
            elif param['type'] == 'inter_block':
                conv = InterSO3ConvBlock(**param['args'])
            elif param['type'] == 'separable_block':
                conv = SeparableSO3ConvBlock(param['args'])
            elif param['type'] == 's2_block':
                conv = S2ConvBlock(**param['args'])
            elif param['type'] == 'separable_s2_block':
                conv = SeparableS2ConvBlock(param['args'])
            else:
                raise ValueError(f'No such type of SO3Conv {param["type"]}')
            self.layer_types.append(param['type'])
            self.blocks.append(conv)
        self.params = params

    def forward(self, x):
        inter_idx, inter_w = None, None
        for conv, param in zip(self.blocks, self.params):
            if param['type'] in ['inter', 'inter_block', 'separable_block', 's2_block', 'separable_s2_block']:
                inter_idx, inter_w, _, x = conv(x, inter_idx, inter_w)
                # import ipdb; ipdb.set_trace()

                if param['args']['stride'] > 1:
                    inter_idx, inter_w = None, None
            elif param['type'] in ['intra_block']:
                # Intra Convolution
                x = conv(x)
            else:
                raise ValueError(f'No such type of SO3Conv {param["type"]}')

        return x

    def get_anchor(self):
        if self.params[-1]['args']['kanchor'] == 12:
            # return torch.from_numpy(sptk.get_anchorsV12())
            raise NotImplementedError("Not clear whether the S2 vertices or SO(3) rotations needed here. ")
        else:
            return torch.from_numpy(sptk.get_anchors())

class SeparableS2ConvBlock(nn.Module):
    def __init__(self, params) -> None:
        """S2Conv and skip (1x1 conv) connection"""
        super().__init__()
        
        dim_in = params['dim_in']
        dim_out = params['dim_out']
        norm = getattr(nn,params['norm']) if 'norm' in params.keys() else None
        
        # self.use_intra = params['kanchor'] > 1

        self.s2_conv = S2ConvBlock(**params)
        
        self.stride = params['stride']

        # 1x1 conv for skip connection
        self.skip_conv = nn.Conv2d(dim_in, dim_out, 1)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)
        self.relu = getattr(F, params['activation'])

    def forward(self, x, inter_idx, inter_w):
        '''
            inter conv with skip connection
        '''
        skip_feature = x.feats
        inter_idx, inter_w, sample_idx, x = self.s2_conv(x, inter_idx, inter_w)
        
        if self.stride > 1:
            skip_feature = zptk.functional.batched_index_select(skip_feature, 2, sample_idx.long())
        skip_feature = self.skip_conv(skip_feature)
        skip_feature = self.relu(self.norm(skip_feature))
        # skip_feature = self.relu(skip_feature)
        x_out = zptk.SphericalPointCloud(x.xyz, x.feats + skip_feature, x.anchors)
        return inter_idx, inter_w, sample_idx, x_out

class SeparableSO3ConvBlock(nn.Module):
    def __init__(self, params):
        """InterSO3, IntraSO3, and skip (1x1 conv) connection"""
        super(SeparableSO3ConvBlock, self).__init__()

        dim_in = params['dim_in']
        dim_out = params['dim_out']
        norm = getattr(nn,params['norm']) if 'norm' in params.keys() else None
        
        self.use_intra = params['kanchor'] > 1

        self.inter_conv = InterSO3ConvBlock(**params)

        intra_args = {
            'dim_in': dim_out,
            'dim_out': dim_out,
            'dropout_rate': params['dropout_rate'],
            'activation': params['activation'],
        }

        if self.use_intra:
            self.intra_conv = IntraSO3ConvBlock(**intra_args)
        self.stride = params['stride']

        # 1x1 conv for skip connection
        self.skip_conv = nn.Conv2d(dim_in, dim_out, 1)
        self.norm = nn.InstanceNorm2d(dim_out, affine=False) if norm is None else norm(dim_out)
        self.relu = getattr(F, params['activation'])


    def forward(self, x, inter_idx, inter_w):
        '''
            inter, intra conv with skip connection
        '''
        skip_feature = x.feats
        inter_idx, inter_w, sample_idx, x = self.inter_conv(x, inter_idx, inter_w)

        if self.use_intra:
            x = self.intra_conv(x)
        if self.stride > 1:
            skip_feature = zptk.functional.batched_index_select(skip_feature, 2, sample_idx.long())
        skip_feature = self.skip_conv(skip_feature)
        skip_feature = self.relu(self.norm(skip_feature))
        # skip_feature = self.relu(skip_feature)
        x_out = zptk.SphericalPointCloud(x.xyz, x.feats + skip_feature, x.anchors)
        return inter_idx, inter_w, sample_idx, x_out

    def get_anchor(self):
        return torch.from_numpy(sptk.get_anchors())

class ClsOutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        super(ClsOutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        fc = params['fc']
        k = params['k']

        self.outDim = k

        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.BatchNorm2d(c))
            c_in = c
        # -----------------------------------------------

        # ------------------ intra conv -----------------
        if 'intra' in params.keys():
            self.intra = nn.ModuleList()
            self.skipconv = nn.ModuleList()
            for intraparams in params['intra']:
                conv = IntraSO3ConvBlock(**intraparams['args'])
                self.intra.append(conv)
                c_out = intraparams['args']['dim_out']

                # for skip convs
                self.skipconv.append(nn.Conv2d(c_in, c_out, 1))
                self.norm.append(nn.BatchNorm2d(c_out))
                c_in = c_out
        # -----------------------------------------------

        # ----------------- pooling ---------------------
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        # BxCxA -> Bx1xA or BxCxA attention weights
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, 1, 1)
        elif self.pooling_method == 'attention2':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, c_in, 1)
        # ------------------------------------------------

        self.fc1 = nn.ModuleList()
        for c in fc:
            self.fc1.append(nn.Linear(c_in, c))
            # self.norm.append(nn.BatchNorm1d(c))
            c_in = c

        self.fc2 = nn.Linear(c_in, self.outDim)

    def forward(self, feats, label=None):
        x_out = feats
        norm_cnt = 0
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(norm(x_out))
            norm_cnt += 1

        # mean pool at xyz
        out_feat = x_out
        x_out = x_out.mean(2, keepdim=True)

        # group convolution after mean pool
        if hasattr(self, 'intra'):
            x_in = zptk.SphericalPointCloud(None, x_out, None)
            for lid, conv in enumerate(self.intra):
                skip_feat = x_in.feats
                x_in = conv(x_in)

                # skip connection
                norm = self.norm[norm_cnt]
                skip_feat = self.skipconv[lid](skip_feat)
                skip_feat = F.relu(norm(skip_feat))
                x_in = zptk.SphericalPointCloud(None, skip_feat + x_in.feats, None)
                norm_cnt += 1
            x_out = x_in.feats


        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=3).mean(dim=2)
        elif self.pooling_method == 'debug':
            # for debug only
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.mean(2).max(-1)[0]
        ############## DEBUG ONLY ######################
        elif label is not None:
            def to_one_hot(label, num_class):
                '''
                label: [B,...]
                return [B,...,num_class]
                '''
                comp = torch.arange(num_class).long().to(label.device)
                for i in range(label.dim()):
                    comp = comp.unsqueeze(0)
                onehot = label.unsqueeze(-1) == comp
                return onehot.float()
            x_out = x_out.mean(2)
            label = label.squeeze()
            if label.dim() == 2:
                cdim = x_out.shape[1]
                label = label.repeat(1,5)[:,:cdim]
            confidence = to_one_hot(label, x_out.shape[2])
            if confidence.dim() < 3:
                confidence = confidence.unsqueeze(1)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        elif self.pooling_method.startswith('attention'):
            x_out = x_out.mean(2)
            out_feat = self.attention_layer(x_out)  # Bx1XA or BxCxA
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        # fc layers
        for linear in self.fc1:
            # norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            # x_out = F.relu(norm(x_out))
            x_out = F.relu(x_out)
            # norm_cnt += 1

        x_out = self.fc2(x_out)

        return x_out, out_feat.squeeze()

class AttRotClass(nn.Module):
    def __init__(self, dim_feat, dim_anchor, dim_cls, feat_all_anchors, anchor_ab_loss=False, fc_on_concat=False) -> None:
        """Permutation layer for classification task. \n
        Attention of the input feature of all anchors under all rotations (permutations) 
        with the (learned) template feature of all classes in all anchors.

        Return: 
        1. response of all classes under best rotation guess (for category classification), 
        2. response of all anchors and rotations under the best class guess (for rotation classification),
        3. permuted input feature under the best rotation guess under the best class guess (for retrieval). """
        super().__init__()
        self.dim_feat = dim_feat
        self.dim_anchor = dim_anchor
        self.dim_cls = dim_cls
        self.feat_all_anchors = feat_all_anchors
        self.anchor_ab_loss = anchor_ab_loss
        self.fc_on_concat = fc_on_concat

        W = torch.empty(self.dim_feat, self.dim_anchor, self.dim_cls)   #c,a,n_class
        # nn.init.xavier_normal_(W, gain=0.001)
        nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
        self.register_parameter('W', nn.Parameter(W))
        
        if self.fc_on_concat:
            self.fc = nn.Linear(self.dim_feat * self.dim_anchor, self.dim_feat)

    def forward(self, x, label=None, x0=None):
        """x: b,c,r,a
        return: [b,n], [b,r,a]"""
        nb = x.shape[0]
        att = torch.einsum("bcra,can->bran", x, self.W) #/ self.dim_feat # bran
        att_rn = torch.sigmoid(att).sum(2) # brn    # attention value for each rotation and class (sum over anchors)
        if label is None:
            att_n, att_n_ridx = att_rn.max(1)   # bn     # max attention value for each class, the rotation index that take the max att value for each class
            _, att_max_nidx = att_n.max(1)    # b          # the class index that take the max att value
        else:
            att_n = att_rn[torch.arange(nb), label.flatten(), :]    # bn
            _, att_max_nidx = att_n.max(1)    # b
            # att_max_nidx = torch.zeros_like(att_max_nidx) #!!!!!!

        if self.anchor_ab_loss:
            assert x0 is not None, "anchor_ab_loss requires x before permutation"
            att_ab = torch.einsum("bca,cdn->badn", x0, self.W)
            att_ab = att_ab[torch.arange(nb), :, :, att_max_nidx]   # bad
            att_anchor = att_ab
        else:
            att_bra = att[torch.arange(nb), :, :, att_max_nidx]    # bra
            assert att_bra.shape == x[:,0].shape, att_bra.shape
            att_anchor = att_bra

        if label is None:
            att_r_idx = att_n_ridx[torch.arange(nb), att_max_nidx]  # b
            x_at_r = x[torch.arange(nb), :, att_r_idx, :]       # bca
        else:
            x_at_r = x[torch.arange(nb), :, label.flatten(), :]       # bca
        if self.feat_all_anchors:
            return att_n, att_anchor, x_at_r.flatten(1)
        elif self.fc_on_concat:
            x_at_r = self.fc(x_at_r.flatten(1))
            return att_n, att_anchor, x_at_r
        else:
            return att_n, att_anchor, x_at_r[...,0]

class ClsOutBlockPointnet(nn.Module):
    def __init__(self, params, norm=None, debug=False):
        """outblock for classification task.
        1x1 convs, spatial pooling, rotational pooling (permutation layer)"""
        super(ClsOutBlockPointnet, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        fc = params['fc']
        k = params['k']
        na = params['kanchor']
        feat_all_anchors = params['feat_all_anchors']
        anchor_ab_loss = params['anchor_ab_loss']
        fc_on_concat = params['fc_on_concat']   # the output is only used in retrieval, not trained. 
        drop_xyz = params['drop_xyz']

        self.outDim = k

        self.linear = nn.ModuleList()
        self.norm = nn.ModuleList()

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            self.norm.append(nn.BatchNorm2d(c))
            c_in = c
        # -----------------------------------------------

        # ----------------- spatial pooling ---------------------
        # self.fc1 = nn.ModuleList()
        # for c in fc:
        #     self.fc1.append(nn.Linear(c_in, c))
        #     # self.norm.append(nn.BatchNorm1d(c))
        #     c_in = c
        self.pointnet = sptk.PointnetSO3Conv(c_in, c_in, na, drop_xyz)
        self.norm.append(nn.BatchNorm1d(c_in))

        # ----------------- rotational pooling ---------------------
        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        # BxCxA -> Bx1xA or BxCxA attention weights
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_in, 1, 1)
        elif self.pooling_method == 'permutation':
            self.att_rot_cat_later = AttRotClass(c_in, na, k, feat_all_anchors, anchor_ab_loss, fc_on_concat)
            assert na == 12, na
            trace_idx_ori, trace_idx_rot = sptk.get_relativeV_index()
            self.register_buffer("trace_idx_ori", torch.tensor(trace_idx_ori, dtype=torch.long))
            self.register_buffer("trace_idx_rot", torch.tensor(trace_idx_rot, dtype=torch.long))
            # self.trace_idx_ori = torch.nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.long),
            #                 requires_grad=False)   # 60*12 da
            # self.trace_idx_rot = torch.nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.long),
            #                 requires_grad=False)   # 60*12 db
        # ------------------------------------------------

        if self.pooling_method != 'permutation':
            self.fc2 = nn.Linear(c_in, self.outDim)

        self.debug = debug
        
    def forward(self, x, label=None):
        x_out = x.feats # bcpa

        if self.debug:
            return x_out[:,:40].mean(-1).mean(-1),None
        
        norm_cnt = 0
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(norm(x_out))
            # x_out = F.relu(x_out)
            norm_cnt += 1

        out_feat = x_out # bcpa
        x_in = zptk.SphericalPointCloud(x.xyz, out_feat, x.anchors)

        x_out = self.pointnet(x_in) # bca

        norm = self.norm[norm_cnt]
        norm_cnt += 1
        x_out = F.relu(norm(x_out))
        # x_out = F.relu(x_out)
        
        # mean pooling # bca -> bc
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=2)
        elif self.pooling_method == 'debug':
            # for debug only
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.max(2)[0]
        elif self.pooling_method.startswith('attention'):
            # the rotational attention is used to weight features from different rotation anchors
            # and fuse them together
            out_feat = self.attention_layer(x_out)  # Bx1XA or BxCxA
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
        elif self.pooling_method == 'permutation':
            x_out_permute = x_out[:,:, self.trace_idx_ori]  # b,c,r,a
            x_out, out_feat, x_feat = self.att_rot_cat_later(x_out_permute, label, x0=x_out) # [b,c_out], [b,r,a]
            return x_out, out_feat, x_feat
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        x_feat = x_out
        x_out = self.fc2(x_out) # b,c_out

        ### category prediction, rotation classs prediction, features for retrieval. 
        return x_out, out_feat.squeeze(), x_feat

class InvOutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        super(InvOutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']

        # TODO
        if 'intra' in params.keys():
            pass

        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        self.norm = nn.ModuleList()

        # Attention layer
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(mlp[-1], 1, 1)
            # self.attention_layer = nn.Conv1d(c_in, 1, 1)

        # 1x1 Conv layer
        self.linear = nn.ModuleList()
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            # self.linear.append(nn.Linear(c_in, c))
            self.norm.append(nn.InstanceNorm2d(c, affine=False))
            c_in = c

        # self.out_norm = nn.BatchNorm1d(c_in)


    def forward(self, feats):
        x_out = feats
        end = len(self.linear)

        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            if lid != end - 1:
                norm = self.norm[lid]
                x_out = F.relu(norm(x_out))

        out_feat = x_out.mean(2)

        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=3).mean(dim=2)
        elif self.pooling_method == 'debug':
            # for debug only
            x_out = x_out[..., 0].mean(2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.mean(2).max(-1)[0]
        elif self.pooling_method == 'attention':
            x_out = x_out.mean(2)
            out_feat = self.attention_layer(x_out)
            confidence = F.softmax(out_feat * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
            out_feat = confidence.squeeze()
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        # batch norm in the last layer?
        # x_out = self.out_norm(x_out)

        return F.normalize(x_out, p=2, dim=1), out_feat


class InvOutBlockPointnet(nn.Module):
    def __init__(self, params, norm=None):
        super(InvOutBlockPointnet, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        c_out = mlp[-1]

        na = params['kanchor']

        if 'pooling' not in params.keys():
            self.pooling_method = 'max'
        else:
            self.pooling_method = params['pooling']

        self.pointnet = sptk.PointnetSO3Conv(c_in,c_out,na)

        # Attention layer
        if self.pooling_method == 'attention':
            self.temperature = params['temperature']
            self.attention_layer = nn.Conv1d(c_out, 1, 1)


        # self.out_norm = nn.BatchNorm1d(c_out, affine=True)


    def forward(self, x):
        # nb, nc, np, na -> nb, nc, na
        x_out = self.pointnet(x)
        out_feat = x_out

        # mean pooling
        if self.pooling_method == 'mean':
            x_out = x_out.mean(dim=2)
        elif self.pooling_method == 'max':
            # max pooling
            x_out = x_out.max(2)[0]
        elif self.pooling_method == 'attention':
            attw = self.attention_layer(x_out)
            confidence = F.softmax(attw * self.temperature, dim=2)
            x_out = x_out * confidence
            x_out = x_out.sum(-1)
            confidence = confidence.squeeze()
        else:
            raise NotImplementedError(f"Pooling mode {self.pooling_method} is not implemented!")

        # batch norm in the last layer?
        # x_out = self.out_norm(x_out)
        return F.normalize(x_out, p=2, dim=1), F.normalize(out_feat, p=2, dim=1)

class InvOutBlockMVD(nn.Module):
    def __init__(self, params, norm=None):
        """outblock for se(3)-invariant descriptor learning"""
        super(InvOutBlockMVD, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        c_out = mlp[-1]
        na = params['kanchor']

        self.p_pool_to_cout = params['p_pool_to_cout']
        self.p_pool_first = params['p_pool_first']
        # self.permute = params['permute']
        self.permute_nl = params['permute_nl']
        self.permute_soft = params['permute_soft']

        # Attention layer
        self.temperature = params['temperature']

        if 'pooling' not in params.keys():
            self.pooling_method = 'attention'
        else:
            self.pooling_method = params['pooling']

        if self.pooling_method == 'attention':
            if self.p_pool_first:
                if self.p_pool_to_cout:
                    self.pointnet = sptk.PointnetSO3Conv(c_in,c_out,na)
                    self.attention_layer = nn.Sequential(nn.Conv1d(c_out, c_out, 1), \
                                                        nn.ReLU(inplace=True), \
                                                        nn.Conv1d(c_out,1,1))
                else:
                    self.pointnet = sptk.PointnetSO3Conv(c_in,c_in,na)
                    self.attention_layer = nn.Sequential(nn.Conv1d(c_in, c_in, 1), \
                                                        nn.ReLU(inplace=True), \
                                                        nn.Conv1d(c_in,1,1))
                    self.out_layer = nn.Linear(c_in, c_out)
            else:
                self.attention_layer = nn.Sequential(nn.Conv2d(c_in, c_in, 1), \
                                                        nn.ReLU(inplace=True), \
                                                        nn.Conv2d(c_in,c_in,1))
                if na != 1:
                    self.attention_layer2 = nn.Sequential(nn.Conv1d(c_in, c_in, 1), \
                                                            nn.ReLU(inplace=True), \
                                                            nn.Conv1d(c_in,1,1))
                self.pointnet = sptk.PointnetSO3Conv(c_in,c_out,1)
                
        elif self.pooling_method == 'permutation':
            self.pointnet = sptk.PointnetSO3Conv(c_in,c_in,na)
            # self.pn_linear = nn.Linear(c_in, c_in)
            self.attention_layer = nn.Conv1d(c_in*na, 1, 1)    #, bias=False
            if self.permute_nl:
                self.out_layer = nn.Sequential(nn.Linear(c_in*na, c_out*2), \
                                                    nn.ReLU(inplace=True), \
                                                    nn.Linear(c_out*2,c_out))
            else:
                self.out_layer = nn.Linear(c_in*na, c_out)

            trace_idx_ori, trace_idx_rot = sptk.get_relativeV_index()
            self.register_buffer("trace_idx_ori", torch.tensor(trace_idx_ori.swapaxes(0,1), dtype=torch.long))  # 12*60
            self.register_buffer("trace_idx_rot", torch.tensor(trace_idx_rot.swapaxes(0,1), dtype=torch.long))

        else:
            raise ValueError(f'pooling_method {self.pooling_method} not recognized')

        # self.out_norm = nn.BatchNorm1d(c_out, affine=True)

    def forward_permute(self, x):
        
        nb, nc, np, na = x.feats.shape
        
        x_feat = self._pooling(x)   # bcpa -> bca
        # x_feat = self.pn_linear(x_feat)     # bca
        x_feat = x_feat[:,:,self.trace_idx_ori].flatten(1,2) #   b[ca]r

        x_attn = self.attention_layer(x_feat).squeeze(1) # br
        if self.permute_soft:
            attn_sfm = F.softmax(x_attn, dim=1)           # br
            x_feat_max_r = (x_feat * attn_sfm.unsqueeze(1)).sum(-1) # b[ca]
        else:
            # x_attn = F.normalize(x_attn, dim=1)     # should not be normalized, commented out 11/8/22
            _, max_r_idx = torch.max(x_attn, dim=1)    # b
            x_feat_max_r = x_feat[torch.arange(nb), :, max_r_idx]   # b[ca]

        x_out = self.out_layer(x_feat_max_r)        # b c_out

        return x_out, x_attn

    def forward_pfirst(self, x):
        ### first do spatial pooling, then do attention on anchors

        nb, nc, np, na = x.feats.shape
        
        if self.p_pool_to_cout:
            x_feat = self.pointnet(x)   # bcpa -> b c_out a
        else:
            x_feat = self._pooling(x)   # bcpa -> bca

        attn = self.attention_layer(x_feat)     # bca -> b1a
        attn_sfm = F.softmax(attn, dim=2)           # b1a
        attn = attn.squeeze(1)                      # ba, unnormalized
        x_out = (x_feat * attn_sfm).sum(-1)         # bca, b1a -> bc

        if not self.p_pool_to_cout:
            x_out = self.out_layer(x_out)           # b, c -> b c_out

        return F.normalize(x_out, p=2, dim=1), attn

    def forward(self, x):
        if self.pooling_method == 'attention':
            if self.p_pool_first:
                return self.forward_pfirst(x)
            else:
                return self.forward_afirst(x)

        elif self.pooling_method == 'permutation':
            return self.forward_permute(x)

        else:
            raise ValueError(f'pooling_method {self.pooling_method} not recognized')

    def forward_afirst(self, x):
        # nb, nc, np, na -> nb, nc, na

        # attention first
        nb, nc, np, na = x.feats.shape

        if na == 1:
            attn = self.attention_layer(x.feats)
            attn = F.softmax(attn, dim=3)           # nb,nc,np,na
            attn_final = attn

            # nb, nc, np, 1
            x_out = (x.feats * attn).sum(-1, keepdim=True)
        else:
            attn = self.attention_layer(x.feats)
            attn_sfm = F.softmax(attn, dim=3)           # nb,nc,np,na

            attn_pool = torch.max(attn,2)[0]    # bca
            attn_final = self.attention_layer2(attn_pool) # b1a
            attn_final = attn_final.squeeze(1)  # ba

            # nb, nc, np, 1
            x_out = (x.feats * attn_sfm).sum(-1, keepdim=True)
        x_in = zptk.SphericalPointCloud(x.xyz, x_out, None)

        # nb, nc
        x_out = self.pointnet(x_in).view(nb, -1)

        return F.normalize(x_out, p=2, dim=1), attn_final

    def _pooling(self, x):
        # [nb, nc, na]
        x_out = self.pointnet(x)
        x_out = F.relu(x_out)

        return x_out

# outblock for rotation regression model
class SO3OutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        super(SO3OutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        # fc = params['fc']
        # k = params['k']
        # self.outDim = k

        self.linear = nn.ModuleList()
        # self.norm = nn.ModuleList()
        self.temperature = params['temperature']
        self.representation = params['representation']
        self.attention_layer = nn.Conv2d(mlp[-1], 1, (1,1))

        # out channel equals 4 for quaternion representation, 6 for ortho representation
        self.regressor_layer = nn.Conv2d(mlp[-1],4,(1,1))

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, 1))
            # self.norm.append(nn.BatchNorm2d(c))
            c_in = c

    def forward(self, feats):
        x_out = feats
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            # norm = self.norm[norm_cnt]
            x_out = linear(x_out)
            x_out = F.relu(x_out)

        # mean pool at xyz ->  BxCxA
        x_out = x_out.mean(2)

        # attention weight
        attention_wts = self.attention_layer(x_out)  # Bx1XA
        confidence = F.softmax(attention_wts * self.temperature, dim=2).view(x_out.shape[0], x_out.shape[2])
        # regressor
        y = self.regressor_layer(x_out) # Bx6xA
        return confidence, y

class RelSO3OutBlockR(nn.Module):
    def __init__(self, params, norm=None):
        """outblock for relative rotation regression"""
        super(RelSO3OutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']
        na = params['kanchor']
        self.rot_ref_tgt = params['rot_ref_tgt']
        self.topk = params['topk']

        self.check_equiv = params['check_equiv']

        if 'pooling' not in params.keys():
            self.pooling_method = 'attention'
        else:
            self.pooling_method = params['pooling']

        if self.pooling_method == 'permutation':
            assert na == 12, f"na = {na} not supported"
            trace_idx_ori, trace_idx_rot = sptk.get_relativeV_index()
            self.register_buffer("trace_idx_ori", torch.tensor(trace_idx_ori, dtype=torch.long))
            self.register_buffer("trace_idx_rot", torch.tensor(trace_idx_rot, dtype=torch.long))
            # self.trace_idx_ori = torch.nn.Parameter(torch.tensor(trace_idx_ori, dtype=torch.long), requires_grad=False)   # 60*12 da
            # self.trace_idx_rot = torch.nn.Parameter(torch.tensor(trace_idx_rot, dtype=torch.long), requires_grad=False)   # 60*12 db
        elif self.pooling_method == 'attention':
            assert na <= 60, f'na={na} not supported'
        else:
            raise ValueError(f'pooling_method {self.pooling_method} not supported')
            

        self.pointnet = sptk.PointnetSO3Conv(c_in, c_in, na)
        if self.pooling_method == 'permutation':
            c_in = c_in * 3
        else:
            c_in = c_in * 2

        self.linear = nn.ModuleList()

        self.temperature = params['temperature']
        rp = params['representation']

        if rp == 'quat':
            self.out_channel = 4
        elif rp == 'ortho6d':
            self.out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%rp)

        self.attention_layer = nn.Conv2d(mlp[-1], 1, (1,1))

        # out channel equals 4 for quaternion representation, 6 for ortho representation
        if self.pooling_method == 'attention':
            self.regressor_layer = nn.Conv2d(mlp[-1],self.out_channel,(1,1))
        elif self.pooling_method == 'permutation':
            self.regressor_layer = nn.Conv2d(mlp[-1]*na,self.out_channel,(1,1))
            self.regressor_layer_mid = nn.Conv2d(mlp[-1]*na,self.out_channel,(1,1))
            # self.regressor_layer = nn.Sequential(
            #     nn.Conv2d(mlp[-1]*na,mlp[-1]*5,(1,1)),
            #     nn.ReLU(),
            #     nn.Conv2d(mlp[-1]*5,self.out_channel,(1,1)),
            # )   # tmp!!!!
        else:
            raise ValueError("self.pooling_method {} not recognized".format(self.pooling_method))

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, (1,1)))
            c_in = c


    def forward(self, f1, f2, x1, x2):
        # nb, nc, np, na -> nb, nc, na
        # f1, x1: src; f2, x2: tgt; src = R * tgt
        sp1 = zptk.SphericalPointCloud(x1, f1, None)
        sp2 = zptk.SphericalPointCloud(x2, f2, None)

        f1 = self._pooling(sp1)
        f2 = self._pooling(sp2)

        if self.pooling_method == 'attention':
            return self.forward_single(f1, f2)
        elif self.pooling_method == 'permutation':
            return self.forward_concat(f1, f2)
        else:
            raise ValueError("self.pooling_method {} not recognized".format(self.pooling_method))


        # return: [nb, na, na], [nb, n_out, na, na]
        # return confidence, y

    def forward_concat(self, f1, f2):
        """f1, f2: nb,nc,na
        x1 = R x2"""
        nb = f1.shape[0]
        na = f1.shape[2]

        if self.rot_ref_tgt:
            f1_permute = f1[:,:,self.trace_idx_ori]
            # x_out = torch.cat([f1_permute, f2.unsqueeze(2).expand_as(f1_permute)], 1)  # b, c*2, r, a
            # x_out = f1_permute - f2.unsqueeze(2)  # b, c, r, a # tmp!!!!
            x_out = torch.cat([f1.unsqueeze(2).expand_as(f1_permute), f1_permute, f2.unsqueeze(2).expand_as(f1_permute)], 1)  # b, c*2, r, a
        else:
            f2_permute = f2[:,:,self.trace_idx_rot]     # b,c,r,a
            # innerp = torch.einsum("bcra,bca->br", f2_permute, f1)

            # x_out = torch.cat([f2_permute, f1.unsqueeze(2).expand_as(f2_permute)], 1)  # b, c*2, r, a # tmp!!!!
            x_out = torch.cat([f1.unsqueeze(2).expand_as(f2_permute), f2_permute, f2.unsqueeze(2).expand_as(f2_permute)], 1)  # b, c*2, r, a

        # fc layers with relu
        for linear in self.linear:
            x_out = linear(x_out)
            x_out = F.relu(x_out)

        attention_wts = self.attention_layer(x_out).view(nb, 60, na)    # logits b,r,a

        confidence = torch.sigmoid(attention_wts)
        confidence_r = confidence.sum(-1)   # b, r
        if self.topk == 1:
            max_conf, max_r = torch.max(confidence_r, 1, keepdim=True)  # b, 1

            max_r = max_r[:,None,:,None].expand(-1, x_out.shape[1], -1, na)     # b, c, 1, a
            x_out_max_r = torch.gather(x_out, 2, max_r).squeeze(2)  # b, c, a
            x_out_max_r = x_out_max_r.reshape(nb, -1, 1, 1)
            y = self.regressor_layer(x_out_max_r).reshape(nb, -1)   # b, c_out
        else:
            # confidence_softmax = F.softmax(confidence_r)
            top_conf, top_idx = torch.topk(confidence_r, self.topk, 1)    # b, topk
            top_idx = top_idx[:,None,:,None].expand(nb, x_out.shape[1], self.topk, na)   # b, c, topk, a
            x_out_top_conf = torch.gather(x_out, 2, top_idx)    # b, c, topk, a
            x_out_top_conf = x_out_top_conf.transpose(2,3).reshape(nb, -1, self.topk, 1)  # b, (c, a), topk
            y = self.regressor_layer_mid(x_out_top_conf)   # b, cout, topk, 1
            y = y.transpose(1,2).reshape(nb, self.topk, self.out_channel)   # b, topk, cout
            y_top = self.regressor_layer(x_out_top_conf[:,:,[0]]).reshape(nb, self.out_channel).unsqueeze(1)   # b, 1, cout
            y = torch.cat([y, y_top], 1)    # b, (topk+1), cout

            # top_mask = top_conf > 0.1
            # top_mask_row = top_mask.sum(1, keepdim=True).expand_as(top_mask)
            # top_mask_max = torch.ones_like(top_mask)
            # # top_mask_max[:,0] = True
            # top_mask_final = torch.where(top_mask_row, top_mask, top_mask_max)
            # top_conf = torch.where(top_mask_final, top_conf, torch.zeros_like(top_conf))
            # top_conf = F.normalize(top_conf, p=1, dim=1)

            # top_conf = top_conf[:,None,:].expand(-1, self.out_channel, -1)   # b, cout, topk
            # y = (ys * top_conf).sum(2)


        # return: binary classification logits [b,r,a], residual rotation [b, c_out]
        # return attention_wts, y
        if self.check_equiv:
            return attention_wts, y, f1_permute, f1, f2
        else:
            return attention_wts, y

    def forward_single(self, f1, f2):
        # return: [nb, na, na], [nb, n_out, na, na]
        nb = f1.shape[0]
        na = f1.shape[2]

        # expand and concat into metric space (nb, nc*2, na_tgt, na_src)
        f2_expand = f2.unsqueeze(-1).expand(-1,-1,-1,na).contiguous()
        f1_expand = f1.unsqueeze(-2).expand(-1,-1,na,-1).contiguous()

        x_out = torch.cat((f1_expand,f2_expand),1)

        # fc layers with relu
        for linear in self.linear:
            x_out = linear(x_out)
            x_out = F.relu(x_out)

        attention_wts = self.attention_layer(x_out).view(nb, na, na)
        confidence = F.softmax(attention_wts * self.temperature, dim=1)
        y = self.regressor_layer(x_out)
        return confidence, y

    def _pooling(self, x):
        # [nb, nc, na]
        x_out = self.pointnet(x)
        x_out = F.relu(x_out)

        return x_out
        # return feats.mean(2)
