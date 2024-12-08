import os
import argparse
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files


def data_process(data_path, patch_size, stride, mode):
    """
    Processes images and binary masks into patches or prepares them for testing.
    :param data_path: Path to the dataset (should contain train/val/test directories).
    :param patch_size: The size of the image patch for partitioning.
    :param stride: Stride size for partitioning.
    :param mode: Mode of operation ('training', 'validation', or 'testing').
    """
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)

    img_dir = os.path.join(data_path, "images", mode)
    mask_dir = os.path.join(data_path, "masks", mode)

    # List image and mask files
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    img_list = []
    gt_list = []

    # Load images and masks
    for img_file, mask_file in zip(img_files, mask_files):
        img = Image.open(os.path.join(img_dir, img_file))
        mask = Image.open(os.path.join(mask_dir, mask_file))

        # Convert images and masks to grayscale (1 channel)
        img = Grayscale(1)(img)
        mask = Grayscale(1)(mask)

        img_list.append(ToTensor()(img))
        gt_list.append(ToTensor()(mask))

    # Normalize images
    img_list = normalization(img_list)

    # Generate patches or save full-size images
    if mode == "training" or mode == "validation":
        img_patches = get_patch(img_list, patch_size, stride)
        mask_patches = get_patch(gt_list, patch_size, stride)

        save_patch(img_patches, save_path, "img_patch")
        save_patch(mask_patches, save_path, "mask_patch")
    elif mode == "testing":
        save_each_image(img_list, save_path, "img")
        save_each_image(gt_list, save_path, "mask")


def get_patch(imgs_list, patch_size, stride):
    """
    Generate patches from images.
    :param imgs_list: List of images (tensor format).
    :param patch_size: Size of the patches.
    :param stride: Stride for generating patches.
    :return: List of image patches.
    """
    patches = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride

    for img in imgs_list:
        img = F.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        img = img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        img = img.permute(1, 2, 0, 3, 4).contiguous().view(-1, 1, patch_size, patch_size)
        patches.extend(img)

    return patches


def save_patch(imgs_list, path, file_prefix):
    """
    Save image patches to disk as pickle files.
    """
    for i, img in enumerate(imgs_list):
        with open(os.path.join(path, f"{file_prefix}_{i}.pkl"), "wb") as f:
            pickle.dump(img.numpy(), f)
        print(f"Saved: {file_prefix}_{i}.pkl")


def save_each_image(imgs_list, path, file_prefix):
    """
    Save each image as a pickle file.
    """
    for i, img in enumerate(imgs_list):
        with open(os.path.join(path, f"{file_prefix}_{i}.pkl"), "wb") as f:
            pickle.dump(img.numpy(), f)
        print(f"Saved: {file_prefix}_{i}.pkl")


def normalization(imgs_list):
    """
    Normalize a list of images using their mean and standard deviation.
    """
    imgs = torch.cat(imgs_list, dim=0)
    mean = imgs.mean()
    std = imgs.std()
    normalized_list = [(Normalize([mean], [std])(img) - img.min()) / (img.max() - img.min()) for img in imgs_list]
    return normalized_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_path", type=str, required=True, help="Path to dataset (with train/val/test structure).")
    parser.add_argument("-ps", "--patch_size", type=int, default=48, help="Patch size for image partitioning.")
    parser.add_argument("-s", "--stride", type=int, default=6, help="Stride for image partitioning.")
    parser.add_argument("-m", "--mode", type=str, required=True, choices=["training", "validation", "testing"], help="Mode: train, val, or test.")
    args = parser.parse_args()

    data_process(args.data_path, args.patch_size, args.stride, args.mode)
