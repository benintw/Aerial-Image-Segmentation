"""
Purpose: evaluator.py is responsible for the evaluation or validation phase, assessing the model’s performance on a separate validation or test dataset. It does not update model parameters but computes metrics to gauge generalization and quality.

Key Responsibilities:
- Load a pre-trained model (e.g., from a checkpoint).
- Iterate over validation/test data batches without gradient computation (using torch.no_grad()).
- Compute evaluation metrics (e.g., IoU, Dice score for segmentation) and log results.
- Generate visualizations (e.g., segmentation masks) if needed.
- Handle evaluation-specific configurations (e.g., batch size, metrics) from configs/validation.yaml.

Context: This is run after training or periodically during training to monitor performance on unseen data, ensuring the model generalizes well.

"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.metric import compute_dice_loss, compute_bce_loss, compute_loss


class Evaluator:

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()

    def validate(self, val_dataloader: DataLoader) -> float:
        """
        Iterate over validation data batches without gradient computation (using torch.no_grad()).
        Compute validation loss.
        """
        total_loss = 0.0

        with torch.inference_mode():
            for batch in tqdm(val_dataloader, desc="Validating ..."):
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = compute_loss(logits, masks)

                total_loss += loss.item()

        return total_loss / len(val_dataloader)
