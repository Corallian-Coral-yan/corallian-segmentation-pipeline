import os
import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.model.classifier import CoralSegmentationModel  # ✅ Use segmentation model
from modules.preprocessing.image_cropper import ImageCropper
# ✅ Use segmentation dataset
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


# ✅ Preprocessing pipeline (Image Cropping)
def preprocess(config):
    if config["UsePreprocessing"] and config["UseCachedPreprocessing"]:
        index_filepath = os.path.join(config["OutputRoot"], "index.csv")
        if os.path.isfile(index_filepath):
            print(
                f"Using cached image crop index file at {index_filepath}, image cropping skipped")
            return

    if config["UsePreprocessing"]:
        cropper = ImageCropper(
            config["OutputRoot"],
            config["BaseInputDir"],
            dirs=config["Dirs"],
            recurse=config["InputRecurse"]
        )
        cropper.begin_cropping()


# ✅ Training pipeline
def train(config):
    if config["DoTraining"]:
        # Load model
        model = CoralSegmentationModel(config).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Load dataset
        train_dataset = CoralSegmentationDataset(image_dir=config["training"]["TrainImageDir"],
                                                 mask_dir=config["training"]["TrainMaskDir"])
        val_dataset = CoralSegmentationDataset(image_dir=config["training"]["ValImageDir"],
                                               mask_dir=config["training"]["ValMaskDir"])

        train_loader = DataLoader(
            train_dataset, batch_size=config["training"]["BatchSize"], shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=config["training"]["BatchSize"], shuffle=False)

        # ✅ Use Dice + BCE loss for segmentation
        dice_loss = DiceLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["LearningRate"])

        # ✅ Training loop
        num_epochs = config["training"]["NumEpochs"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # ✅ Apply both BCE and Dice loss
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # ✅ Save trained model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/coral_segmentation.pth")
        print("Model saved successfully!")


# ✅ Full training pipeline (Preprocessing + Training)
def full_train(config):
    preprocess(config["preprocessing"])
    train(config["training"])
    print("Done")


# ✅ Load config and start training
def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    full_train(config)


if __name__ == "__main__":
    main()
