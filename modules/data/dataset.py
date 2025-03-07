import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CoralSegmentationDataset(Dataset):
    """
    Custom dataset class for coral segmentation.
    Loads raw coral images and their corresponding binary masks.
    """

    def __init__(self, image_dir, mask_dir, transform=None, target_size=(512, 512)):
        """
        Args:
            image_dir (str): Path to the directory containing raw coral images.
            mask_dir (str): Path to the directory containing binary masks.
            transform (callable, optional): Additional transform to be applied on images.
            target_size (tuple): Desired output size (width, height) for images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size

        # Sorted filenames to ensure pairing
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load raw image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both image and mask to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0).float()  # binarize mask

        # Normalize the image
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)

        if self.transform:
            image = self.transform(image)

        return image, mask
