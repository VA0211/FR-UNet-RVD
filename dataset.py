import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation


class VesselDataset(Dataset):
    def __init__(self, data_path, mode):
        """
        Custom Dataset for Vessel Segmentation.
        :param data_path: Root path to the dataset (Spatial directory).
        :param mode: Mode of the dataset (training, validation, or testing).
        """
        self.mode = mode
        self.img_path = os.path.join(data_path, "images", mode)
        self.mask_path = os.path.join(data_path, "masks", mode)

        # List image and mask files
        self.img_files = sorted(os.listdir(self.img_path))
        self.mask_files = sorted(os.listdir(self.mask_path))

        # Define data augmentation/transforms for training
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ]) if mode == "train" else None

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding ground truth mask.
        :param idx: Index of the data item.
        :return: Tuple of (image, mask), both as tensors.
        """
        # Load image and mask
        img_file = self.img_files[idx]
        mask_file = self.mask_files[idx]

        img = self._load_image(os.path.join(self.img_path, img_file))
        mask = self._load_image(os.path.join(self.mask_path, mask_file))

        # Apply data augmentation if in training mode
        if self.mode == "train" and self.transforms is not None:
            seed = torch.seed()  # Ensures consistent transforms for image and mask
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            mask = self.transforms(mask)

        return img, mask

    def _load_image(self, file_path):
        """
        Load an image or mask as a PyTorch tensor.
        :param file_path: Path to the image/mask file.
        :return: Tensor representation of the image/mask.
        """
        with open(file_path, mode='rb') as file:
            data = pickle.load(file)
        return torch.from_numpy(data).float()

    def __len__(self):
        """
        Total number of samples in the dataset.
        """
        return len(self.img_files)
