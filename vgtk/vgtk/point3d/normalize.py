
import math
import numpy as np
import torch


'''
Point cloud transform:
    pc: 
        torch: [b, 3, p]
        np: [(b, )3, p]
'''


# translation normalization
def centralize(pc):
    return pc - pc.mean(dim=2, keepdim=True)

def centralize_np(pc, batch=False):
    axis = 2 if batch else 1
    return pc - pc.mean(axis=axis, keepdims=True)


def normalize(pc):
    """Centralize and normalize to a unit ball. Take a batched pytorch tensor. """
    pc = centralize(pc)
    var = pc.pow(2).sum(dim=1, keepdim=True).sqrt()
    return pc / var.max(dim=2, keepdim=True)

def normalize_np(pc, batch=False):
    """Centralize and normalize to a unit ball. Take a numpy array. """
    pc = centralize_np(pc, batch)
    axis = 1 if batch else 0
    var = np.sqrt((pc**2).sum(axis=axis, keepdims=True))
    return pc / var.max(axis=axis+1, keepdims=True)