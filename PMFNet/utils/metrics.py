"""
Utility functions and metrics for model evaluation
"""
import numpy as np
import torch
import torch.nn.functional as F


def iou_score_fn(output, target):
    """
    Intersection over Union (IoU) metric
    
    Args:
        output: Model predictions
        target: Ground truth labels
    
    Returns:
        IoU score
    """
    smooth = 1e-5

    output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef_fn(output, target):
    """
    Dice Coefficient metric
    
    Args:
        output: Model predictions
        target: Ground truth labels
    
    Returns:
        Dice coefficient score
    """
    smooth = 1

    output = output.view(-1)
    target = target.view(-1)
    intersection = (output * target).sum()

    return (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
