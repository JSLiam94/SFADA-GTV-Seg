import numpy as np
from medpy.metric import hd95 as hd95_medpy
from torch import Tensor
import torch
import torch.nn as nn


def dice(output, target, eps=1e-5):
    eps=1e-5
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    '''
    Calculate three Dice coefficients for different classes:
    - Dice1 (ET): label4 (replaced with 3)
    - Dice2 (TC): labels 1 and 3
    - Dice3 (WT): labels 1, 2, and 3

    Parameters:
    - output: (b, num_class, d, h, w)
    - target: (b, d, h, w)
    '''
    output = torch.argmax(output, dim=1)
    target = target.long()
    if np.any(output):
        dice1 = 0 
    dice1 = dice((output == 3).long(), (target == 3).long())
    dice2 = dice(((output == 1) | (output == 3)).long(), ((target == 1) | (target == 3)).long())
    dice3 = dice((output != 0).long(), (target != 0).long())

    return dice1, dice2, dice3

def Dice(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N





def cal_hd95(output: Tensor, target: Tensor, spacing=None):
    output = torch.argmax(output, dim=1)
    target = target.float()

    hd95_ec = compute_hd95((output == 3).float(), (target == 3).float())
    hd95_co = compute_hd95(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    hd95_wt = compute_hd95((output != 0).float(), (target != 0).float())

    return hd95_ec, hd95_co, hd95_wt

def compute_hd95(pred, gt, spacing=None):
    #pred = pred.bool().cpu().numpy()
    #gt = gt.bool().cpu().numpy()

    try:
        hd = hd95_medpy(pred, gt, voxelspacing=spacing)
    except:
        hd = 373.1287 if np.any(gt) else 0.0
        #print(hd)

    return hd