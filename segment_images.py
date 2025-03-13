import torch
import toml
import os
import cv2
import numpy as np
from torchvision import transforms
from modules.model.classifier import CoralSegmentationModel  # Import your model

# Load config
CONFIG_PATH = "config.toml"
config = toml.load(CONFIG_PATH)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model():
    if not config["ModelFilepath"]:
        raise ValueError(
            "ModelFilepath is empty. Please set it in config.toml.")

    # Initialize model with config
    model = CoralSegmentationModel(config)

    # Load trained weights
    model.load_state_dict(torch.load(
        config["ModelFilepath"], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Perform segmentation
def segment_image(model, image_path, output_folder):
    image_tensor = preprocess_image(image_path).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)

    # Convert to binary mask
    mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # Assuming single-class
    mask = (mask > 0.5).astype(np.uint8) * 255  # Thresholding

    # Save result
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"seg_{filename}")
    cv2.imwrite(output_path, mask)
    print(f"Saved segmented image: {output_path}")

# Process folder


def segment_folder():
    input_folder = config["inference"]["ImageDir"]
    output_folder = config["inference"]["OutputDir"]

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    model = load_model()

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            segment_image(model, image_path, output_folder)


if __name__ == "__main__":
    segment_folder()
