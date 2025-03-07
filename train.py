import os
import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # For interpolation
from torch.utils.data import DataLoader

from modules.model.classifier import CoralSegmentationModel
from modules.data.dataset import CoralSegmentationDataset


# ✅ Dice Loss for better segmentation results
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        num = 2 * (preds * targets).sum()
        den = preds.sum() + targets.sum() + self.smooth
        return 1 - num / den  # Dice Loss


def train(config):
    if config["training"]["DoTraining"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Use full config for model initialization
        print("Initializing model...")
        model = CoralSegmentationModel(config).to(device)
        print("Model initialized.")

        # Use training-specific keys for DataLoader and optimizer settings
        train_dataset = CoralSegmentationDataset(
            image_dir=config["training"]["TrainImageDir"],
            mask_dir=config["training"]["TrainMaskDir"]
        )
        val_dataset = CoralSegmentationDataset(
            image_dir=config["training"]["ValImageDir"],
            mask_dir=config["training"]["ValMaskDir"]
        )

        print("Datasets created.")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=config["training"]["BatchSize"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["training"]["BatchSize"], shuffle=False
        )

        print("DataLoaders created.")

        dice_loss = DiceLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["LearningRate"]
        )
        print("Optimizer created.")

        num_epochs = config["training"]["NumEpochs"]
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            print(f"\nEpoch {epoch+1} started.")

            for batch_idx, (images, masks) in enumerate(train_loader):
                print(
                    f"  Batch {batch_idx+1}/{len(train_loader)}: loading data")
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                # Upsample outputs to match target mask size
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                print(f"    Batch {batch_idx+1} loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(train_loader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")

        # Optional inference snippet...
        model.eval()
        print("\nRunning inference example on the validation set...")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                print(f"  Processing validation batch {batch_idx+1}")
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                print(f"    Predictions shape: {preds.shape}")
                break

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/coral_segmentation.pth")
        print("Model saved successfully!")


# ✅ Full training pipeline (Preprocessing + Training)
def full_train(config):
    train(config)
    print("Done")


# ✅ Load config and start training
def main():
    print("Loading configuration...")
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    print("Configuration loaded.")
    full_train(config)


if __name__ == "__main__":
    main()
