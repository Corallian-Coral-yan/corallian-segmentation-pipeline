import torch
import tomllib  # Built-in TOML parser (Python 3.11+)
import os
import cv2
import numpy as np
from torchvision import transforms
from modules.model.classifier import CoralSegmentationModel  # Import your model

# Load config
CONFIG_PATH = "config.toml"

with open(CONFIG_PATH, "rb") as f:  # Open in binary mode (required by tomllib)
    config = tomllib.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
def load_model():
    model_path = config["ModelFilepath"]

    if not model_path:
        raise ValueError(
            "ModelFilepath is empty. Please set it in config.toml.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = CoralSegmentationModel(config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"Model loaded from: {model_path}")
    # Debugging output
    print(f"Model is on: {next(model.parameters()).device}")
    return model


# Image preprocessing
def preprocess_image(image_path):
    # Read in original format
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    img_size = config["inference"]["ImageSize"]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Add batch dimension and move to device
    return transform(image).unsqueeze(0).to(DEVICE)
    print(f"Image tensor is on: {tensor_image.device}")  # Debugging output
    return tensor_image


# Perform segmentation
def segment_image(model, image_path, output_path):
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        # Ensure input is on the right device
        output = model(image_tensor.to(DEVICE))

    # Convert output to binary mask
    mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary (0 or 255)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask.astype(np.uint8))
    print(f"Saved segmented image: {output_path}")


# Process the root folder recursively
def segment_folder():
    input_root = config["inference"]["InputRoot"]
    output_root = config["inference"]["OutputRoot"]

    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    model = load_model()

    for root, _, files in os.walk(input_root):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(root, filename)

                # Preserve relative folder structure
                relative_path = os.path.relpath(root, input_root)
                output_folder = os.path.join(output_root, relative_path)
                output_path = os.path.join(output_folder, f"seg_{filename}")

                segment_image(model, input_path, output_path)


if __name__ == "__main__":
    segment_folder()
