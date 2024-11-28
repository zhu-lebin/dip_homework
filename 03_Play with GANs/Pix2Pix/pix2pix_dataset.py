import os
from torch.utils.data import Dataset
import cv2
import torch

class pix2pixDataset(Dataset):
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (string): Path to the dataset folder containing images.
        """
        # Read all jpg image file paths from the dataset directory
        self.image_filenames = [
            os.path.join(dataset_dir, fname)
            for fname in os.listdir(dataset_dir)
            if fname.endswith('.jpg')
        ]

        if len(self.image_filenames) == 0:
            raise ValueError(f"No .jpg files found in the directory: {dataset_dir}")
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)

        if img_color_semantic is None:
            raise ValueError(f"Failed to read image: {img_name}")

        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0

        # Split into RGB and semantic parts
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]

        return image_rgb, image_semantic