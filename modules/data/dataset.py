import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CoralSegmentationDataset(Dataset):
    """
    Custom dataset class for coral segmentation.
    Loads images and their corresponding masks for training.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
        - image_dir (str): Path to the directory containing images.
        - mask_dir (str): Path to the directory containing masks.
        - transform (callable, optional): Optional transform to be applied on images.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get sorted list of images and masks to ensure correct pairing
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        """Returns total number of samples."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Loads and returns an image-mask pair."""
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load image and mask
        image = Image.open(img_path).convert(
            "RGB")  # Convert image to 3-channel RGB
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Convert image and mask to tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # Normalize image (standard practice in deep learning)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])(image)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).float()

        return image, mask
