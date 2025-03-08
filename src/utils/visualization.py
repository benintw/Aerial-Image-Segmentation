import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torchvision

import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
from pathlib import Path
import numpy as np


def show_image(image, mask, pred_image=None):

    if pred_image == None:

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.set_title("IMAGE")
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap="gray")

        ax2.set_title("GROUND TRUTH")
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap="gray")

    elif pred_image != None:

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        ax1.set_title("IMAGE")
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap="gray")

        ax2.set_title("GROUND TRUTH")
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap="gray")

        ax3.set_title("MODEL OUTPUT")
        ax3.imshow(pred_image.permute(1, 2, 0).squeeze(), cmap="gray")


def visualize_dataset(dataset: Dataset) -> None:
    """Plots the first 25 images and masks from a dataset."""

    images, masks = next(iter(DataLoader(dataset, batch_size=25, shuffle=True)))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Images")
    plt.imshow(
        torchvision.utils.make_grid(
            images[:25], nrow=5, padding=2, normalize=True
        ).permute(1, 2, 0),
        cmap="magma",
    )
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Masks")
    plt.imshow(
        torchvision.utils.make_grid(
            masks[:25], nrow=5, padding=2, normalize=True
        ).permute(1, 2, 0),
        cmap="magma",
    )
    plt.show()


def visualize_batch_from_dataloader(dataloader: DataLoader) -> None:
    """Visualize a batch of images and masks from a dataloader."""
    images, masks = next(iter(dataloader))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Images")
    plt.imshow(
        torchvision.utils.make_grid(
            images[:8], nrow=4, normalize=True, padding=2
        ).permute(1, 2, 0),
        cmap="hot",
    )

    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Masks")
    plt.imshow(
        torchvision.utils.make_grid(
            masks[:8], nrow=4, normalize=True, padding=2
        ).permute(1, 2, 0),
        cmap="hot",
    )

    plt.show()


"""
Visualization utilities for training and prediction results.

This module provides functions for visualizing:
- Training metrics (loss, accuracy)
- Model predictions
- Dataset samples with annotations
"""


def plot_training_history(history: dict, save_path: Path | str | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Validation")
    ax1.set_title("Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Validation")
    ax2.set_title("Accuracy History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_segmentation(image, pred, target=None, output_path=None):
    """
    Visualize the input image, predicted mask, and (optionally) ground truth mask.
    """
    fig, axes = plt.subplots(1, 2 if target is None else 3, figsize=(15, 5))
    axes[0].imshow(image.cpu().numpy().transpose(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    axes[1].imshow(pred.cpu().numpy(), cmap="gray")
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    if target is not None:
        axes[2].imshow(target.cpu().numpy(), cmap="gray")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
