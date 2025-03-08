# scripts/split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path


def split_dataset(
    csv_path: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> None:
    """
    Split the dataset into training, validation, and test sets, saving them as separate CSVs.

    Args:
        csv_path (str): Path to the input train.csv.
        output_dir (str): Directory to save the split CSVs.
        train_ratio (float): Proportion of data for training (e.g., 0.7).
        val_ratio (float): Proportion of data for validation (e.g., 0.15).
        random_seed (int): Random seed for reproducibility.
    """
    # Validate ratios
    test_ratio = 1.0 - train_ratio - val_ratio
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError(
            "Train, validation, and test ratios must be between 0 and 1 and sum to 1"
        )

    # Load the dataset
    df = pd.read_csv(csv_path)

    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=random_seed, shuffle=True
    )

    # Second split: train vs val (adjust val_ratio for the remaining data)
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    # Add a 'split' column to the original dataframe (optional)
    df["split"] = ""
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    # Save the split dataframes
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "train_with_splits.csv", index=False)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    print(f"Saved split CSVs to {output_dir}:")
    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/train.csv",
        help="Path to the input train.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Directory to save split CSVs",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Proportion of data for training"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    split_dataset(
        args.csv_path,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.random_seed,
    )
