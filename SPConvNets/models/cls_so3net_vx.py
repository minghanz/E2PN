import math
import os
from turtle import forward
from black import out
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from collections import OrderedDict
import json
from escnn import gspaces
from escnn import nn as enn

class ClsSO3VoxConvModel(nn.Module):
    def __init__(self, params=None):
        super(ClsSO3VoxConvModel, self).__init__()
        self.scale = params.model.scale
        self.freq = params.model.freq   # 2 or 3
        if params.model.group == 'SO3':
            self.init_SO3()
        elif params.model.group == 'I':
            self.init_I()
        elif params.model.group == 'S2':
            raise NotImplementedError('g.quotient_representation((False, 5)) not implemented')
            self.init_S2()
        else:
            raise NotImplementedError(f'group {params.model.group} not supported')

    # def init_S2(self):
    #     self.r3_act = gspaces.icoOnR3()
    #     g = self.r3_act.fibergroup
    #     reg_repr = g.quotient_representation((False, 5))    # not implemented


    def init_I(self):
        self.r3_act = gspaces.icoOnR3()
        g = self.r3_act.fibergroup
        reg_repr = g.regular_representation

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r3_act, [self.r3_act.trivial_repr])
        self.input_type = in_type
        # convolution 0
        out_type = enn.FieldType(self.r3_act, 2*self.scale*[self.r3_act.irrep(0)] + 2*self.scale*[self.r3_act.irrep(1)] + 2*self.scale*[self.r3_act.irrep(2)] + 3*self.scale*[self.r3_act.irrep(3)] + 3*self.scale*[self.r3_act.irrep(4)] )
        self.conv0 = enn.R3Conv(in_type, out_type, kernel_size=5, padding=2)
        print(f'conv 0: {in_type.size} {out_type.size}')

        in_type = out_type
        activation1 = enn.ELU(enn.FieldType(self.r3_act, 4*self.scale*[reg_repr]), inplace=True)
        out_type = activation1.in_type
        next_type = enn.FieldType(self.r3_act, 4*self.scale*[self.r3_act.irrep(0)] + 4*self.scale*[self.r3_act.irrep(1)] + 4*self.scale*[self.r3_act.irrep(2)] + 6*self.scale*[self.r3_act.irrep(3)] + 6*self.scale*[self.r3_act.irrep(4)])
        self.block1 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation1,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 39-240-78
        self.skip1 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 1: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        activation2 = enn.ELU(enn.FieldType(self.r3_act, 8*self.scale*[reg_repr]), inplace=True)
        out_type = activation2.in_type
        next_type = enn.FieldType(self.r3_act, 4*self.scale*[reg_repr])
        self.block2 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation2,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 78-480-240
        self.skip2 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 2: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        activation3 = enn.ELU(enn.FieldType(self.r3_act, 8*self.scale*[reg_repr]), inplace=True)
        out_type = activation3.in_type
        next_type = enn.FieldType(self.r3_act, 8*self.scale*[reg_repr])
        self.block3 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation3,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 240-480-480
        self.skip3 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 3: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        activation4 = enn.ELU(enn.FieldType(self.r3_act, 16*self.scale*[reg_repr]), inplace=True)
        out_type = activation4.in_type
        next_type = enn.FieldType(self.r3_act, 5*self.scale*[reg_repr]+1*self.scale*[self.r3_act.irrep(0)] + 1*self.scale*[self.r3_act.irrep(1)] + 1*self.scale*[self.r3_act.irrep(2)] + 1*self.scale*[self.r3_act.irrep(3)])
        self.block4 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation4,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 480-960-312
        self.skip4 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 4: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        out_type = enn.FieldType(self.r3_act, [reg_repr] * 128)
        self.conv5 = enn.R3Conv(in_type, out_type, 3)
        self.inv_layer = enn.GroupPooling(out_type)
        next_type = self.inv_layer.out_type
        print(f'inv: {in_type.size} {out_type.size} {next_type.size}')

        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.fc3 = nn.Linear(128, 40)

    def init_SO3(self):
        self.r3_act = gspaces.rot3dOnR3(self.freq)
        g = self.r3_act.fibergroup
        reg_repr = g.bl_regular_representation(self.freq)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r3_act, [self.r3_act.trivial_repr])
        self.input_type = in_type
        # convolution 0 
        out_type = enn.FieldType(self.r3_act, 5*self.scale*[self.r3_act.irrep(0)] + 3*self.scale*[self.r3_act.irrep(1)] + 5*self.scale*[self.r3_act.irrep(2)] )

        self.conv0 = enn.R3Conv(in_type, out_type, kernel_size=5, padding=2)
        print(f'conv 0: {in_type.size} {out_type.size}')
        
        in_type = out_type
        if self.freq == 2:
            activation1 = enn.FourierELU(self.r3_act, 7*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation1 = enn.FourierELU(self.r3_act, 3*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)
        out_type = activation1.in_type
        if self.freq == 2:
            next_type = enn.FieldType(self.r3_act, 10*self.scale*[self.r3_act.irrep(0)] + 6*self.scale*[self.r3_act.irrep(1)] + 10*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = enn.FieldType(self.r3_act, [reg_repr])

        self.block1 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation1,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 39-240-78
        self.skip1 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 1: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        if self.freq == 2:
            activation2 = enn.FourierELU(self.r3_act, 14*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation2 = enn.FourierELU(self.r3_act, 6*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)

        out_type = activation2.in_type
        if self.freq == 2:
            next_type = enn.FieldType(self.r3_act, 26*self.scale*[self.r3_act.irrep(0)] + 28*self.scale*[self.r3_act.irrep(1)] + 26*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = enn.FieldType(self.r3_act, 3*[reg_repr])

        self.block2 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation2,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 78-480-240
        self.skip2 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 2: {in_type.size} {out_type.size} {next_type.size}')
        
        in_type = next_type
        if self.freq == 2:
            activation3 = enn.FourierELU(self.r3_act, 14*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation3 = enn.FourierELU(self.r3_act, 6*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)

        out_type = activation3.in_type
        if self.freq == 2:
            next_type = enn.FieldType(self.r3_act, 53*self.scale*[self.r3_act.irrep(0)] + 54*self.scale*[self.r3_act.irrep(1)] + 53*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = enn.FieldType(self.r3_act, 6*[reg_repr])

        self.block3 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation3,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 240-480-480
        self.skip3 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 3: {in_type.size} {out_type.size} {next_type.size}')
        
        in_type = next_type
        if self.freq == 2:
            activation4 = enn.FourierELU(self.r3_act, 27*self.scale, irreps=[(f,) for f in range(3)], type='rand', N=60, inplace=True)
        else:
            activation4 = enn.FourierELU(self.r3_act, 11*self.scale, irreps=[(f,) for f in range(4)], type='thomson', N=120, inplace=True)

        out_type = activation4.in_type
        if self.freq == 2:
            next_type = enn.FieldType(self.r3_act, 35*self.scale*[self.r3_act.irrep(0)] + 34*self.scale*[self.r3_act.irrep(1)] + 35*self.scale*[self.r3_act.irrep(2)])
        else:
            next_type = enn.FieldType(self.r3_act, 4*[reg_repr])

        self.block4 = enn.SequentialModule(
            enn.R3Conv(in_type, out_type, 3, 1),
            enn.IIDBatchNorm3d(out_type),
            activation4,
            enn.R3Conv(out_type, next_type, 3, 1, stride=2)
        )   # 480-960-312
        self.skip4 = enn.SequentialModule(
            enn.R3Conv(in_type, next_type, 1),
            enn.PointwiseAvgPool3D(next_type, 2, 2, 1),
        )
        print(f'block 4: {in_type.size} {out_type.size} {next_type.size}')

        in_type = next_type
        out_type = enn.FieldType(self.r3_act, [reg_repr] * 128)
        self.conv5 = enn.R3Conv(in_type, out_type, 3)
        self.inv_layer = enn.NormPool(out_type)
        next_type = self.inv_layer.out_type
        print(f'inv: {in_type.size} {out_type.size} {next_type.size}')

        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.fc3 = nn.Linear(128, 40)

    def forward(self, input):
        bdim = input.shape[0]
        x = self.input_type(input)
        x = self.conv0(x)
        # print(f"conv0 {x.shape}")

        x1 = self.block1(x)
        x2 = self.skip1(x)
        # print(f"block1 {x1.shape} skip1 {x2.shape}")
        x = x1 + x2

        x1 = self.block2(x)
        x2 = self.skip2(x)
        # print(f"block3 {x1.shape} skip2 {x2.shape}")
        x = x1 + x2
        
        x1 = self.block3(x)
        x2 = self.skip3(x)
        # print(f"block3 {x1.shape} skip3 {x2.shape}")
        x = x1 + x2

        x1 = self.block4(x)
        x2 = self.skip4(x)
        # print(f"block4 {x1.shape} skip4 {x2.shape}")
        x = x1 + x2
        x = self.conv5(x)
        # print(f"conv5 {x.shape}")

        x = self.inv_layer(x)
        # print(f"inv_layer {x.shape}")
        
        x = x.tensor.reshape(bdim, -1)
        x_feat = x
        # print(f"torch tensor {x.shape}")
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x, x_feat

if __name__ == '__main__':
    a = ClsSO3VoxConvModel()
    print('tested')