import os
import torch
import tomllib
from torch.utils.data import DataLoader
from modules.model.classifier import CoralSegmentationModel
from modules.data.dataset import CoralSegmentationDataset


def main():
    print("[DEBUG] Starting debug script...")

    # ✅ Check GPU availability

    print("\n[DEBUG] PyTorch CUDA Info:")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    print(f"  - CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  - Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Current CUDA Device: {torch.cuda.current_device()}")
        device = torch.device("cuda")
        print(f"[DEBUG] Using device: {device}")
        
    else:
        print("[ERROR] CUDA is not available. Training will fail.")

    # ✅ Load configuration
    try:
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)
        print("[DEBUG] Config loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return

    # ✅ Test model initialization
    try:
        model = CoralSegmentationModel(config).to(device)
        print("[DEBUG] Model initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return

    # ✅ Test dataset loading
    try:
        train_dataset = CoralSegmentationDataset(
            image_dir=config["training"]["TrainImageDir"],
            mask_dir=config["training"]["TrainMaskDir"]
        )
        print(
            f"[DEBUG] Train dataset loaded successfully. Size: {len(train_dataset)}")
        print(
            f"[DEBUG] Train image directory: {config['training']['TrainImageDir']}")
        print(
            f"[DEBUG] Train mask directory: {config['training']['TrainMaskDir']}")
    except Exception as e:
        print(f"[ERROR] Train dataset loading failed: {e}")
        return

    try:
        val_dataset = CoralSegmentationDataset(
            image_dir=config["training"]["ValImageDir"],
            mask_dir=config["training"]["ValMaskDir"]
        )
        print(
            f"[DEBUG] Validation dataset loaded successfully. Size: {len(val_dataset)}")
        print(
            f"[DEBUG] Validation image directory: {config['training']['ValImageDir']}")
        print(
            f"[DEBUG] Validation mask directory: {config['training']['ValMaskDir']}")
    except Exception as e:
        print(f"[ERROR] Validation dataset loading failed: {e}")
        return

    # ✅ Test DataLoader
    try:
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        print("[DEBUG] Train DataLoader created successfully.")
    except Exception as e:
        print(f"[ERROR] Train DataLoader creation failed: {e}")
        return

    try:
        images, masks = next(iter(train_loader))
        print("[DEBUG] Successfully loaded a batch from train_loader.")
    except Exception as e:
        print(f"[ERROR] Failed to load batch from train_loader: {e}")
        return


    # ✅ Test model forward pass
    try:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        print("[DEBUG] Model forward pass successful.")
    except Exception as e:
        print(f"[ERROR] Model forward pass failed: {e}")
        return

    print("[DEBUG] Debugging complete. Everything seems to be working.")


if __name__ == "__main__":
    main()
