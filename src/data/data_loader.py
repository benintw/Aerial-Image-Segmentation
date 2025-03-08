import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from icecream import ic
from typing import Any
import os

from src.data.dataset import AerialImageDataset, get_transforms
from src.utils.visualization import visualize_batch_from_dataloader


def get_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    df = pd.read_csv(config["csv_path"])

    # Load split CSVs
    train_df = pd.read_csv(os.path.join(config["split_dir"], "train_split.csv"))
    val_df = pd.read_csv(os.path.join(config["split_dir"], "val_split.csv"))
    train_transform, val_transform = get_transforms(config)

    # train_df, val_df = train_test_split(
    #     df,
    #     test_size=config["train_test_split"],
    #     random_state=config["random_seed"],
    # )

    train_dataset = AerialImageDataset(train_df, train_transform)
    val_dataset = AerialImageDataset(val_df, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        # persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        # persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
    )

    return train_loader, val_loader


def get_inference_loader(config: dict[str, Any]) -> DataLoader:
    """Create and return a DataLoader for inference data."""
    required_keys = [
        "input_dir",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")

    image_paths = glob.glob(os.path.join(config["input_dir"], "*.png"))
    if not image_paths:
        raise ValueError(f"No images found in {config['input_dir']}")

    dataset = InferenceDataset(image_paths, transform=config.get("transform"))
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
    )


def get_test_loader(config: dict[str, Any]) -> DataLoader:
    """Create and return a DataLoader for the test dataset."""
    required_keys = [
        "split_dir",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")

    test_df = pd.read_csv(os.path.join(config["split_dir"], "test_split.csv"))
    _, test_transform = get_transforms(config)
    test_dataset = AerialImageDataset(test_df, test_transform)
    return DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
    )


def test_data_loaders(train_loader: DataLoader, val_loader: DataLoader) -> None:
    """Validate data loader outputs and print debug information."""
    ic(len(train_loader), len(val_loader))

    # Get first batch
    batch = next(iter(train_loader))
    images, masks = batch[0], batch[1]

    # Check shapes and value ranges
    ic(images.shape, masks.shape)
    ic(images.min(), images.max())
    ic(masks.min(), masks.max())


def main() -> None:
    with open("configs/dataset.yaml") as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = get_dataloaders(config)

    try:
        test_data_loaders(train_loader, val_loader)
    finally:
        # Clean up DataLoader workers
        train_loader._iterator = None
        val_loader._iterator = None

    visualize_batch_from_dataloader(train_loader)


if __name__ == "__main__":
    main()
