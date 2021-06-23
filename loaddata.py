"""
Data Loader implementation, specifically designed for in-house datasets. Code will be designed to reflect flexibility in
custom dataloaders for new data.

Created by Matthew Keaton 2/18/21
"""

import numpy as np
from torch.utils.data import Dataset
from torch import as_tensor
import os
from tifffile import imread
from tqdm import tqdm
import cv2
from cellpose_src.utils import diameters

from transforms import random_horizontal_flip, random_rotate \
    # ResizeImage

import matplotlib.pyplot as plt


class StandardizedTiffData(Dataset):
    """
    Dataset subclass for loading in any tiff data, where the dataset follows the following format:
        - /data
            - vol1.tiff
            ...
            - voln.tiff
        - /labels
            - lbl1.tiff
            ...
            - lbln.tiff
    Data and labels are expected to be named in such a way that when sorted in ascending order,
    the ith element of data corresponds to the ith label
    """

    # Currently, set to load in volumes upfront (via __init__()) rather than one at a time (via __getitem__())
    def __init__(self, data_dir, do_3D=False, d_transform=None, l_transform=None, augmentations=None,
                 default_meds=(32, 32)):
        self.d_transform = d_transform
        self.l_transform = l_transform
        self.augmentations = augmentations
        self.default_x = default_meds[0]
        self.default_y = default_meds[1]
        print('Loading Dataset Data Volumes...')
        self.d_list = sorted([data_dir + os.sep + 'data' + os.sep + f for f in os.listdir(os.path.join(data_dir, 'data'))
                              if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])
        self.data = []
        if do_3D:
            for d_file in tqdm(self.d_list):
            # for d_file in self.d_list:
                self.data.extend(list(imread(d_file).astype('float')))
        else:
            for d_file in tqdm(self.d_list):
            # for d_file in self.d_list:
                self.data.append(imread(d_file).astype('float'))
        print('\nLoading Dataset Labels...')
        self.l_list = sorted([data_dir + os.sep + 'labels' + os.sep + f for f in os.listdir(os.path.join(data_dir, 'labels'))
                              if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])
        self.labels = []
        if do_3D:
            # for l_file in tqdm(self.l_list):
            for l_file in self.l_list:
                self.labels.extend(list(imread(l_file).astype('uint8')))
        else:
            # for l_file in tqdm(self.l_list):
            for l_file in self.l_list:
                self.labels.append(imread(l_file).astype('uint8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # resize = ResizeImage()
        X = self.data[index]
        y = self.labels[index]
        label_file = self.l_list[index]
        if self.d_transform:
            X = self.d_transform(as_tensor(X))
        if self.l_transform:
            y = self.l_transform(as_tensor(y))
            y = as_tensor(y)

        med, cts = diameters(y)
        rescale_x, rescale_y = self.default_x / med, self.default_y / med
        X = np.transpose(X.numpy(), (1, 2, 0))
        X = cv2.resize(X, (int(X.shape[1] * rescale_x), int(X.shape[0] * rescale_y)),
                        interpolation=cv2.INTER_LINEAR)[np.newaxis, :]
        y = np.transpose(y.numpy(), (1, 2, 0))
        y = cv2.resize(y, (int(y.shape[1] * rescale_x), int(y.shape[0] * rescale_y)),
                       interpolation=cv2.INTER_NEAREST)[np.newaxis, :]

        return X, y[0], label_file
