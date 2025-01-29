import os
import requests
import zipfile
import glob
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# URL of the COCO dataset
url = "http://images.cocodataset.org/zips/test2014.zip"
dataset_dir = "./test-denoising/coco_test2014"
zip_path = "./test-denoising/test2014.zip"

# Create destination directory
os.makedirs(dataset_dir, exist_ok=True)

# Download
if not os.path.exists(zip_path):
    print("Downloading the file...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print("Download completed.")
else:
    print("The file already exists.")

# Decompression
if not os.listdir(dataset_dir):  # Check if the directory is empty
    print("Extracting the file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Extraction completed.")
else:
    print("Files have already been extracted.")


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(
            output_size, (int, tuple)
        ), "Output size must be an int or tuple"
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2, "Output size tuple must have two elements"
            self.output_size = output_size

    def __call__(self, sample):
        image, groundtruth = sample
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # Padding if image is smaller than crop size
        if h < new_h:
            pad_h = new_h - h + 1
            image = np.pad(image, ((pad_h, 0), (0, 0), (0, 0)), mode="reflect")
            groundtruth = np.pad(
                groundtruth, ((pad_h, 0), (0, 0), (0, 0)), mode="reflect"
            )
        if w < new_w:
            pad_w = new_w - w + 1
            image = np.pad(image, ((0, 0), (pad_w, 0), (0, 0)), mode="reflect")
            groundtruth = np.pad(
                groundtruth, ((0, 0), (pad_w, 0), (0, 0)), mode="reflect"
            )

        h, w = image.shape[:2]
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top : top + new_h, left : left + new_w]
        groundtruth = groundtruth[top : top + new_h, left : left + new_w]
        return image, groundtruth


class MyImage(Dataset):
    def __init__(self, ind_init, ind_final, transform=None):
        super().__init__()
        # Use a relative path to the dataset directory
        image_path = os.path.join(dataset_dir, "test2014", "*.jpg")
        self.seg_path = [
            seq
            for i, seq in enumerate(glob.iglob(image_path))
            if ind_init <= i < ind_final
        ]
        self.leng = len(self.seg_path)
        self.transform = transform

    def __len__(self):
        return self.leng

    def __getitem__(self, index):
        img = imageio.imread(self.seg_path[index])
        x = img

        if self.transform:
            img, x = self.transform((img, x))

        img = np.rollaxis(img, 2)
        x = np.rollaxis(x, 2)
        img = torch.from_numpy(img).float() / 255.0
        x = torch.from_numpy(x).long() / 255.0

        # Add noise
        img += (0.1**0.5) * torch.randn(img.shape)
        img = torch.clip(img, 0, 1)
        return img, x
