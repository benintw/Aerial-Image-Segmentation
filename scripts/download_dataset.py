def download_dataset():
    """
    Downloads the road segmentation dataset files (images/, masks/, and train.csv)
    from GitHub and saves them to the data/Road_seg_dataset/ directory.
    """
    import os
    import requests
    from pathlib import Path
    import shutil

    # Create base directory
    base_dir = Path("data/")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Base GitHub raw content URL
    base_url = "https://raw.githubusercontent.com/parth1620/Road_seg_dataset/master"

    # Download train.csv
    csv_url = f"{base_url}/train.csv"
    csv_path = base_dir / "train.csv"

    print("Downloading train.csv...")
    response = requests.get(csv_url)
    if response.status_code == 200:
        with open(csv_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download train.csv: {response.status_code}")
        return

    # Download images and masks directories
    for directory in ["images", "masks"]:
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)

        print(f"\nDownloading {directory}...")

        # Read train.csv to get file names
        with open(csv_path, "r") as f:
            # Skip header
            next(f)
            for line in f:
                # Extract filename from the appropriate column
                if directory == "images":
                    filename = line.split(",")[0].split("/")[-1]
                else:
                    filename = line.split(",")[1].strip().split("/")[-1]

                file_url = f"{base_url}/{directory}/{filename}"
                file_path = dir_path / filename

                if not file_path.exists():  # Skip if file already exists
                    try:
                        response = requests.get(file_url)
                        if response.status_code == 200:
                            with open(file_path, "wb") as f:
                                f.write(response.content)
                            # print(f"Downloaded {filename}")
                        else:
                            print(
                                f"Failed to download {filename}: {response.status_code}"
                            )
                    except Exception as e:
                        print(f"Error downloading {filename}: {e}")

    print("\nDownload completed!")


if __name__ == "__main__":
    download_dataset()
