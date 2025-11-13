import collections
import glob
import logging
import math
import os
import warnings

import numpy as np
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None
    open_image = False

    def set_custom_transforms(self, transforms):
        if transforms:
            raise NotImplementedError("Not implemented for modified transforms.")


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)


class iCIFAR100(iCIFAR10):
    base_dataset = datasets.cifar.CIFAR100
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    class_order = [  # Taken from original iCaRL implementation:
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]


class ImageNet100(DataHandler):
    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    imagenet_size = 100
    open_image = True
    suffix = ""
    metadata_path = None

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)

    def base_dataset(self, data_path, train=True, download=False):
        if download:
            warnings.warn(
                "ImageNet incremental dataset cannot download itself,"
                " please see the instructions in the README."
            )

        split = "train" if train else "val"

        print("Loading metadata of ImageNet_{} ({} split).".format(self.imagenet_size, split))
        metadata_path = os.path.join(
            data_path if self.metadata_path is None else self.metadata_path,
            "{}_{}{}.txt".format(split, self.imagenet_size, self.suffix)
        )

        self.data, self.targets = [], []
        #print(f"\n\nreading paths from file - {metadata_path}\n\nwhich is appended to {data_path}\n\n")
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")


                self.data.append(os.path.join(data_path, path))
                self.targets.append(int(target))

        self.data = np.array(self.data)

        #print(f"example - {self.data[0]}")
        for i in self.data:
            if not os.path.exists(i):
                print(i)
                raise Exception(f"Image {i} not available in the dataset folder")
        return self



class ImageNet1000(ImageNet100):
    imagenet_size = 1000
