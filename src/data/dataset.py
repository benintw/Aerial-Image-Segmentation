import cv2
import torch
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from torch.utils.data import Dataset
import albumentations as A

from icecream import ic
from typing import Any

from src.utils.visualization import visualize_dataset


class AerialImageDataset(Dataset):
    def __init__(self, df, transform: A.Compose | None = None) -> None:
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        row = self.df.iloc[idx]

        img_path = "data/{}".format(row["images"])
        mask_path = "data/{}".format(row["masks"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image is None at idx: {idx}")
        if mask is None:
            raise ValueError(f"Mask is None at idx: {idx}")

        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        img_tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        mask_tensor = torch.round(torch.from_numpy(mask).permute(2, 0, 1) / 255.0)

        return img_tensor, mask_tensor


class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path


def get_transforms(config: dict[str, Any]) -> tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            A.Resize(height=config["image_size"], width=config["image_size"]),
            A.HorizontalFlip(p=config.get("horizontal_flip", 0.5)),
            A.VerticalFlip(p=config.get("vertical_flip", 0.5)),
        ],
        is_check_shapes=False,
    )
    val_transform = A.Compose(
        [
            A.Resize(height=config["image_size"], width=config["image_size"]),
        ],
        is_check_shapes=False,
    )
    return train_transform, val_transform


def main() -> None:

    with open("configs/dataset.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["csv_path"])
    ic(df.head())
    ic(df.shape)

    idx = 0

    img_path = "data/{}".format(df.iloc[idx]["images"])
    mask_path = "data/{}".format(df.iloc[idx]["masks"])

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    ic(img.shape, mask.shape)

    train_transform, val_transform = get_transforms(config)

    train_dataset = AerialImageDataset(df, train_transform)
    val_dataset = AerialImageDataset(df, val_transform)

    ic(len(train_dataset), len(val_dataset))

    visualize_dataset(train_dataset)


if __name__ == "__main__":
    main()
