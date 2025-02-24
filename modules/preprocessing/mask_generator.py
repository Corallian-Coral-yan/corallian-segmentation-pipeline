import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import os


class MaskGenerator:
    """
    Generates segmentation masks using DeepLabV3+.
    """

    def __init__(self, image_dir, output_dir):
        """
        Args:
        - image_dir (str): Path to the directory containing coral images.
        - output_dir (str): Path to save generated masks.
        """
        self.image_dir = image_dir
        self.output_dir = output_dir

        # Load pretrained DeepLabV3 model
        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=True).eval()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image(self, image_path):
        """
        Process and generate a segmentation mask for a single image.
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)

        # Perform segmentation
        with torch.no_grad():
            output = self.model(image_tensor)["out"][0]
        mask = output.argmax(0).byte().cpu().numpy()

        return image, mask

    def generate_masks(self):
        """
        Generate segmentation masks for all images in the directory.
        """
        image_filenames = sorted(os.listdir(self.image_dir))

        for img_file in image_filenames:
            image_path = os.path.join(self.image_dir, img_file)
            image, mask = self.process_image(image_path)

            # Save the mask as an image
            # Convert mask to binary (0 or 255)
            mask_image = Image.fromarray(mask * 255)
            mask_output_path = os.path.join(
                self.output_dir, f"{os.path.splitext(img_file)[0]}_mask.png")
            mask_image.save(mask_output_path)

            print(f"Saved mask: {mask_output_path}")

            # (Optional) Visualize
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original Coral Image")

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Generated Mask")

            plt.show()


# ✅ Run mask generation
if __name__ == "__main__":
    image_directory = "datasets/train_images/"  # Replace with your dataset path
    mask_output_directory = "datasets/train_masks/"

    generator = MaskGenerator(image_directory, mask_output_directory)
    generator.generate_masks()
