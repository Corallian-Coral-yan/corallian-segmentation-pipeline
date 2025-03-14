import os
import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # For interpolation
from torch.utils.data import DataLoader

from modules.model.classifier import CoralSegmentationModel
from modules.data.dataset import CoralSegmentationDataset

# Dice Loss for better segmentation results
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        num = 2 * (preds * targets).sum(dim=(1, 2, 3))
        den = preds.sum(dim=(1, 2, 3)) + \
            targets.sum(dim=(1, 2, 3)) + self.smooth
        return 1 - (num / den).mean()  # Averaging over batch


def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + \
        target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


def compute_dice(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + 1e-6) / \
        (pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6)
    return dice.mean().item()


def train(config):
    if not config["training"]["DoTraining"]:
        print("Training is disabled in config.")
        return

    # Set device (force CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize model and move to GPU
    print("Initializing model...")
    model = CoralSegmentationModel(config).to(device)
    print("Model initialized.")

    # Load datasets
    train_dataset = CoralSegmentationDataset(
        image_dir=config["training"]["TrainImageDir"],
        mask_dir=config["training"]["TrainMaskDir"]
    )
    val_dataset = CoralSegmentationDataset(
        image_dir=config["training"]["ValImageDir"],
        mask_dir=config["training"]["ValMaskDir"]
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DataLoaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["BatchSize"], shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["BatchSize"], shuffle=False, num_workers=8, pin_memory=True
    )

    print("DataLoaders created.")

    # Define loss functions and optimizer
    dice_loss = DiceLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["training"]["LearningRate"])

    print("Optimizer created.")

    num_epochs = config["training"]["NumEpochs"]
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch+1} started.")

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)  # Move to GPU
            optimizer.zero_grad()

            outputs = model(images)
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f"    Batch {batch_idx+1} loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.4f}")

    # Evaluation mode for validation
    model.eval()
    total_iou, total_dice = 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            # Move outputs and masks to CPU before computing metrics
            outputs_np = torch.sigmoid(outputs).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            total_iou += compute_iou(torch.tensor(outputs_np),
                                     torch.tensor(masks_np))
            total_dice += compute_dice(torch.tensor(outputs_np),
                                       torch.tensor(masks_np))
            num_batches += 1

    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    print(f"Validation IoU: {avg_iou:.4f}, Dice Score: {avg_dice:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/coral_segmentation.pth")
    print("Model saved successfully!")


# Full training pipeline
def full_train(config):
    train(config)
    print("Done")


# Load config and start training
def main():
    print(f"Loading configuration...")
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    print("Configuration loaded.")
    full_train(config)


if __name__ == "__main__":
    main()