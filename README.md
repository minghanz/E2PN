
# E2PN: Efficient SE(3)-Equivariant Point Network

This repository contains the code (in PyTorch) for [E2PN: Efficient SE(3)-Equivariant Point Network](https://arxiv.org/abs/2206.05398). The implementation is developed based on an open-sourced existing work: [Equivariant Point Network for 3D Point Cloud Analysis](https://github.com/nintendops/EPN_PointCloud) (EPN). 


## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Experiments](#experiments)
4. [Contacts](#contacts)

## Introduction

E2PN is a SE(3)-equivariant network architecture designed for deep point cloud analysis. It mainly includes a SE(3)-equivariant convolution on $S^2\times \mathbb{R}^3$ which is a homogeneous space of SE(3), and a permutation layer to recover SE(3) information from the features in the homogeneous space. 

<!-- ![](https://github.com/nintendops/EPN_PointCloud/blob/main/media/spconv.png) -->



## Usage

The code has been tested on Python3.7, PyTorch 1.7.1 and CUDA (10.1). The module and additional dependencies can be set up with 
```
cd vgtk
python setup.py build_ext
```

## Experiments

**Datasets**

The rotated Modelnet40 point cloud dataset is generated from the [Aligned Modelnet40 subset](https://github.com/lmb-freiburg/orion) and can be downloaded using this [link](https://drive.google.com/file/d/1xRoYjz2KCwkyIPf21E-WKIZkjLYabPgJ/view?usp=sharing).

The original 3DMatch training and evaluation dataset can be found [here](https://3dmatch.cs.princeton.edu/#keypoint-matching-benchmark). We followed [this repo](https://github.com/craigleili/3DLocalMultiViewDesc) to preprocess rgb frames into fused fragments and extract matching keypoints for training. The preprocessed data ready for training can be downloaded [here](https://drive.google.com/file/d/1ME42RjtrNJNz1zSTBrO2NtG89fsOkQLv/view?usp=sharing) (146GB). We also prepared to preprocessed 3DMatch evaluation dataset [here](https://drive.google.com/file/d/14ZGJZHuQLhg87En4C5po6bgTFn4tns4R/view?usp=sharing) (40GB), where local patches around testing keypoints have been precomputed.

<!-- **Pretrained Model**

Pretrained model can be downloaded using this [link](https://drive.google.com/file/d/1vy9FRGWQsuVi4nf--YIqg_8yHFiWWJhh/view?usp=sharing) -->

**Training**

The following lines can be used for the training of each experiment

***E2PN mode***
```
# modelnet classification
python run_modelnet.py experiment --experiment-id cls_s2 --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
model --flag permutation --kanchor 12 --feat-all-anchors

# modelnet shape alignment
python run_modelnet_rotation.py experiment --experiment-id rot_s2 --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
model --flag permutation --kanchor 12

# 3DMatch shape registration
python run_3dmatch.py experiment --experiment-id inv_s2 --model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_TRAIN \
train --npt 16 model --kanchor 12
```
***EPN mode***
```
# modelnet classification
python run_modelnet.py experiment --experiment-id cls --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 \
train_loss --cls-attention-loss

# modelnet shape alignment
python run_modelnet_rotation.py experiment --experiment-id rot --model-dir PATH_TO_LOG -d PATH_TO_MODELNET40

# 3DMatch shape registration
python run_3dmatch.py experiment --experiment-id inv --model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_TRAIN \
train --npt 16
```

**Evaluation**

The following lines can be used for the evaluation of each experiment

***E2PN mode***
```
# modelnet classification
python run_modelnet.py experiment --experiment-id cls_s2 --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 -r PATH_TO_CKPT \
model --flag permutation --kanchor 12 --feat-all-anchors

# modelnet shape alignment
python run_modelnet_rotation.py experiment --experiment-id rot_s2 --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 -r PATH_TO_CKPT \
model --flag permutation --kanchor 12

# 3DMatch shape registration
python run_3dmatch.py experiment --experiment-id inv_s2 --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_EVAL -r PATH_TO_CKPT \
model --kanchor 12
```
***EPN mode***
```
# modelnet classification
python run_modelnet.py experiment --experiment-id cls --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 -r PATH_TO_CKPT \
model train_loss --cls-attention-loss

# modelnet shape alignment
python run_modelnet_rotation.py experiment --experiment-id rot --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_MODELNET40 -r PATH_TO_CKPT

# 3DMatch shape registration
python run_3dmatch.py experiment --experiment-id inv --run-mode eval \
--model-dir PATH_TO_LOG -d PATH_TO_3DMATCH_EVAL -r PATH_TO_CKPT
```


## Citation
If you find our project useful in your research, please consider citing:

```
@article{zhu20222,
  title={E $\^{} 2$ PN: Efficient SE (3)-Equivariant Point Network},
  author={Zhu, Minghan and Ghaffari, Maani and Clark, William A and Peng, Huei},
  journal={arXiv preprint arXiv:2206.05398},
  year={2022}
}
```
