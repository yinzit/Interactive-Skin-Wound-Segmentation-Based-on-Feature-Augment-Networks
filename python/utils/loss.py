import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from math import exp
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        intersect = (input*target).sum()
        union = torch.sum(input) + torch.sum(target)
        Dice = (2*intersect+self.smooth)/(union+self.smooth)
        dice_loss = 1-Dice
        return dice_loss


class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=4, smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(0, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num)
        return dice_loss


class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice))**self.gamma
        dice_loss = Dice/(self.class_num - 1)
        return dice_loss


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        # pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# 加权bceloss
class EdgeLoss(nn.Module):
    def __init__(self, edge_weight=1):
        super(EdgeLoss, self).__init__()
        self.edge_weight = edge_weight

    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def forward(self, inputs, targets):
        edgein = inputs
        edgemask = targets
        loss = self.edge_weight * 20 * self.bce2d(edgein, edgemask)

        return loss