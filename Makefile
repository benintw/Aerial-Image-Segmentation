# Default config file paths
TRAIN_CONFIG ?= configs/training.yaml
VAL_CONFIG ?= configs/validation.yaml
PRED_CONFIG ?= configs/inference.yaml




# Directory for split data
SPLIT_DIR = data/splits

# Download dataset
download_dataset:
	@echo "Downloading dataset..."
	@./scripts/download_dataset.sh
	@echo "Dataset downloaded successfully."

# Check if split data exists
check_splits:
	@if [ ! -f $(SPLIT_DIR)/train_split.csv ]; then \
		echo "Error: Split CSVs not found. Run 'make split_data' first."; \
		exit 1; \
	fi

# Split data command
split_data:
	@echo "Splitting dataset..."
	@./scripts/split_data.sh
	@echo "Dataset split completed."

# Training command
train: check_splits
	@echo "Starting training..."
	@./scripts/train.sh 
	@echo "Training completed. Check logs/train_*.log for details."

# Validation command
validate: check_splits
	@echo "Starting validation..."
	@./scripts/validate.sh
	@echo "Validation completed. Check logs/validate_*.log for details."

# Prediction command
predict: check_splits
	@echo "Starting prediction..."
	@./scripts/predict.sh
	@echo "Prediction completed. Check predictions/ for outputs."

# Run all steps in sequence
all: download_dataset split_data train validate predict
	@echo "All steps completed successfully."

# Clean generated files (add commands to remove outputs if needed)
clean:
	rm -rf outputs/*
	rm -rf __pycache__/*

check:
	poetry check

.PHONY: train validate predict all clean check split_data download_dataset
