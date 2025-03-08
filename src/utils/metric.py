"""
Purpose:
trainer.py is responsible for the training phase of your deep learning model. It implements the training loop, where the model learns from the training data by optimizing its parameters (e.g., weights) using a loss function, optimizer, and gradient descent.

Key Responsibilities:
- Initialize the model, optimizer, and loss function.
- Iterate over training data batches in epochs.
- Forward pass (compute predictions), calculate loss, backward pass (compute gradients), and update model parameters.
- Log training metrics (e.g., loss) and save checkpoints (e.g., best model weights).
- Handle training-specific configurations (e.g., learning rate, batch size) from configs/training.yaml.

Context: This is the core learning process, typically run once or iteratively during development to improve model performance on training data.


"""

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


def compute_dice_loss(logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Compute the Dice loss between the predicted logits and the ground truth masks."""
    dice = DiceLoss(mode="binary", from_logits=True).to(logits.device)
    loss: torch.Tensor = dice(logits, gt_masks)
    return loss


def compute_bce_loss(logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Compute the BCE loss between the predicted logits and the ground truth masks."""
    bce: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to(logits.device)
    loss: torch.Tensor = bce(logits, gt_masks)
    return loss


def compute_loss(logits: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Compute the total loss between the predicted logits and the ground truth masks."""
    dice_loss: torch.Tensor = compute_dice_loss(logits, gt_masks)
    bce_loss: torch.Tensor = compute_bce_loss(logits, gt_masks)
    loss: torch.Tensor = dice_loss + bce_loss
    return loss
