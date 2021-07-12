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

from transforms import random_horizontal_flip, random_rotate

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
    def __init__(self, data_dir, do_3D=False, from_3D=False, d_transform=None, l_transform=None, size_model=None):
        """"
        Args:
            data_dir: root directory of the dataset, containing 'data' and 'labels' folders
            do_3D: whether or not to train 3D cellpose model (requires that from_3d is true)
            from_3D: whether input samples are 2D images (False) or 3D volumes (True)
            d_transform: composed transformation to be applied to input data
            l_transform: composed transformation to be applied to input labels
            size_model: input size_model to be used for diameter prediction/resizing
        """
        self.d_transform = d_transform
        self.l_transform = l_transform
        # if size_model != None:
        #     print('Add size_model loading code here')
        #     # TODO: Add this later
        self.d_list = sorted([data_dir + os.sep + 'data' + os.sep + f for f in os.listdir(os.path.join(data_dir, 'data'))
                              if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])
        self.data = []
        if do_3D:  # and from_3D
            for d_file in tqdm(self.d_list, desc='Loading Dataset Data Volumes...'):
            # for d_file in self.d_list:
                self.data.extend(list(imread(d_file).astype('float')))
        else:
            if from_3D:
                for d_file in tqdm(self.d_list, desc='Loading Dataset Data Volumes...'):
                    raw_vol = imread(d_file).astype('float')
                    # raw_vol = raw_vol.astype('float')
             
                    self.data.append(raw_vol[len(raw_vol)//2])
                print('handle this')
            else:
                for d_file in tqdm(self.d_list, desc='Loading Dataset Data Volumes...'):
                # for d_file in self.d_list:
                    self.data.append(imread(d_file).astype('float'))
        self.l_list = sorted([data_dir + os.sep + 'labels' + os.sep + f for f in os.listdir(os.path.join(data_dir, 'labels'))
                              if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])
        self.labels = []
        if do_3D:  # and from_3D
            for l_file in tqdm(self.l_list, desc='Loading Dataset Labels...'):
            # for l_file in self.l_list:
                self.labels.extend(list(imread(l_file).astype('int16')))
        else:
            if from_3D:
                # TODO: handle this
                print('do this')
            else:
                for l_file in tqdm(self.l_list, desc='Loading Dataset Labels...'):
                # for l_file in self.l_list:
                    self.labels.append(imread(l_file).astype('int16'))

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

        return X, y, label_file
