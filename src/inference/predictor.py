"""
Purpose: predictor.py is responsible for the inference phase, applying the trained model to new data to generate predictions (e.g., segmentation masks for aerial images).

Key Responsibilities:
- Load a pre-trained model (e.g., from a checkpoint like checkpoints/best_model.pt).
- Process new input data (e.g., test images) without gradient computation (using torch.no_grad()).
- Generate predictions (e.g., binary segmentation masks).
- Save predictions to disk (e.g., as PNG images) and optionally visualize them.
- Handle inference-specific configurations (e.g., input directory, output directory) from configs/inference.yaml.

Context: This runs after training and validation, typically for deployment, testing on new data, or generating results for submission (e.g., Kaggle, client deliverables). It’s triggered by make predict.

"""

# src/inference/predictor.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from src.utils.visualization import visualize_segmentation

# NOTE: Predictor should predict on a single map image or a batch of map images
# TODO: Add a function to predict on a single map image
# TODO: Add a function to predict on a batch of map images


class Predictor:
    def __init__(self, model: torch.nn.Module, device: torch.device, config: dict):
        """
        Initialize the Predictor for inference.

        Args:
            model (torch.nn.Module): Pre-trained model for inference.
            device (torch.device): Device to run inference on (e.g., 'cuda', 'cpu').
            config (dict): Configuration dictionary with inference settings.
        """
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()
        self.output_dir = Path(config.get("output_dir", "predictions"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, data_loader: DataLoader) -> None:
        """
        Generate predictions on the provided data loader and save them.

        Args:
            data_loader (DataLoader): DataLoader containing test images.
        """
        with torch.inference_mode():
            for batch_idx, (images, image_paths) in enumerate(
                tqdm(data_loader, desc="Predicting")
            ):
                images = images.to(self.device)
                logits = self.model(images)
                preds = (logits > 0.5).float()  # Convert logits to binary predictions

                # Save predictions and visualize
                for i, (pred, image_path) in enumerate(zip(preds, image_paths)):
                    output_path = self.output_dir / f"pred_{batch_idx}_{i}.png"
                    visualize_segmentation(
                        image=images[i].cpu(),
                        pred=pred.cpu(),
                        target=None,  # No ground truth during inference
                        output_path=output_path,
                    )
                    print(f"Saved prediction at {output_path}")
