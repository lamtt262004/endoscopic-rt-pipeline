"""
Loss functions for polyp segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import dice_coef_fn


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        dice = dice_coef_fn(y_pred, y_true)
        return 1 - dice


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss"""
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bceloss(pred_, target_)


class BceDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    def __init__(self, wb=0.5, wd=0.5):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    """BCE-Dice Loss with multi-scale ground truth supervision"""
    def __init__(self, wb=0.5, wd=0.5):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = (
            self.bcedice(gt_pre5, target) * 0.1 +
            self.bcedice(gt_pre4, target) * 0.2 +
            self.bcedice(gt_pre3, target) * 0.3 +
            self.bcedice(gt_pre2, target) * 0.4 +
            self.bcedice(gt_pre1, target) * 0.5
        )
        return bcediceloss + gt_loss


def tversky(y_true, y_pred, smooth=1e-5, alpha=0.7):
    """
    Tversky Index metric
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        smooth: Smoothing factor
        alpha: Hyperparameter for Tversky index
    
    Returns:
        Tversky index score
    """
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    """Tversky Loss (1 - Tversky Index)"""
    return 1 - tversky(y_true, y_pred)


def dice_tversky_loss(pred, target, bce_weight=0.5):
    """Combined Dice and Tversky Loss"""
    dice = DiceLoss()(pred, target)
    tv = tversky_loss(target, pred)
    loss = dice * bce_weight + tv * (1 - bce_weight)
    return loss
