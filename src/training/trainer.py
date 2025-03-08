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
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.utils.device import get_device
from src.data.data_loader import get_dataloaders
from src.models.model import RoadSegModel
from src.training.evaluator import Evaluator
from src.utils.metric import compute_dice_loss, compute_bce_loss, compute_loss
from src.utils.visualization import plot_training_history


class Trainer:

    def __init__(self, config: dict, device_name: str) -> None:
        self.config = config
        self.device = get_device(device_name)
        print(f"Using device: {self.device}")

        self.setup_seeds()

        self.train_dataloader, self.val_dataloader = get_dataloaders(config)
        self.model = self._create_model(config)
        self.model.to(self.device)
        self.evaluator = Evaluator(self.model, self.device, config)
        self.optimizer = self._create_optimizer(self.model.parameters(), config)

        self.grad_clip = config.get("grad_clip", 1.0)
        self.history: dict[str, list[float | np.ndarray]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def setup_seeds(self) -> None:
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["random_seed"])
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config["random_seed"])

    def _create_optimizer(self, model_params, config):
        optimizer_config = config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "Adam").lower()
        optimizer_params = optimizer_config.get(
            "params", {"lr": config["learning_rate"]}
        )

        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if optimizer_type not in optimizers:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_type}. Use one of {list(optimizers.keys())}"
            )
        return optimizers[optimizer_type](model_params, **optimizer_params)

    def _create_model(self, config):
        model_name = config.get("model", "RoadSegModel")
        models = {
            "roadsegmodel": RoadSegModel,
        }
        if model_name.lower() not in models:
            raise ValueError(
                f"Unsupported model: {model_name}. Use one of {list(models.keys())}"
            )
        return models[model_name.lower()](config)

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_dataloader, desc="Training"):
            images, masks = batch
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)

            loss = compute_loss(logits, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def validate(self) -> float:
        """Validate the model on the validation set using the Evaluator"""
        avg_loss = self.evaluator.validate(self.val_dataloader)
        return avg_loss

    def train(self):
        best_val_loss: float = float("inf")

        epochs_pbar = tqdm(
            range(self.config["epochs"]),
            desc="Training",
        )

        for epoch in epochs_pbar:
            print(f"Epoch {epoch + 1} of {self.config['epochs']:03d}")

            # Epoch level losses (ie. per epoch losses)
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            if patience_counter >= self.config["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Update progress bar description
            epochs_pbar.set_description(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_training_history(
            self.history,
            save_path=Path(self.config["logging"]["log_dir"])
            / f"training_history_{timestamp}.png",
        )

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        save_path = Path(self.config["save_dir"]) / self.config["save_name"]

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
