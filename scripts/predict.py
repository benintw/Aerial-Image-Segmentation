"""
Purpose: predict.py serves as the command-line entry point for the make predict command, orchestrating the inference process by setting up the necessary components and calling Predictor.predict().

Key Responsibilities:
- Parse command-line arguments (e.g., --config, --device).
- Load the configuration from configs/inference.yaml.
- Initialize the model, data loader, and Predictor.
- Run the prediction process and log results.

Context: This script integrates with your Makefile and scripts/predict.sh, providing a user-friendly interface to run inference.

"""

import argparse
import torch
from src.inference.predictor import Predictor
from src.utils.config import load_configs
from src.models.model import RoadSegModel
from src.data.data_loader import get_inference_loader
from pathlib import Path


def _create_model(config):
    model_name = config.get("model", "RoadSegModel").lower()
    models = {
        "roadsegmodel": RoadSegModel,
        # Add other models as needed (e.g., "unet": UNet)
    }
    if model_name not in models:
        raise ValueError(
            f"Unsupported model: {model_name}. Use one of {list(models.keys())}"
        )
    return models[model_name](config)


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate predictions with the segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to the inference config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run inference on",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_configs(args.config)

    # Set up device
    device = torch.device(args.device)

    # Initialize model and load checkpoint
    model = _create_model(config).to(device)
    checkpoint_path = Path(config["save_dir"]) / config["save_name"]
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:

        raise
    except Exception as e:

        raise

    # 2025-Mar-08:
    # TODO: Add a function to predict on a single map image
    # TODO: Add a function to predict on a batch of map images

    # Load inference data
    data_loader = get_inference_loader(config)

    # Initialize predictor
    predictor = Predictor(model, device, config)

    # Run prediction
    predictor.predict(data_loader)


if __name__ == "__main__":
    main()
