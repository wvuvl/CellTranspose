"""
Data Loader implementation, specifically designed for in-house datasets. Code will be designed to reflect flexibility in
custom dataloaders for new data.

Created by Matthew Keaton 2/18/21
"""
from torch.utils.data import Dataset
from torch import as_tensor, tensor, cat, unsqueeze
import os
from tifffile import imread
from tqdm.contrib import tzip

from transforms import Reformat, Normalize1stTo99th, Resize, random_horizontal_flip,\
    random_rotate, LabelsToFlows, generate_patches

import matplotlib.pyplot as plt


class CellPoseData(Dataset):
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

    # Currently, set to load in volumes upfront to cpu memory (via __init__())
    # rather than one at a time via gpu memory (via __getitem__())
    def __init__(self, split_name, data_dir, do_3D=False, from_3D=False, evaluate=False,
                 resize: Resize = None):
        """"
        Args:
            data_dir: root directory/directories of the dataset, containing 'data' and 'labels' folders
            do_3D: whether or not to train 3D cellpose model (requires that from_3d is true)
            from_3D: whether input samples are 2D images (False) or 3D volumes (True)
            d_transform: composed transformation to be applied to input data
            l_transform: composed transformation to be applied to input labels
            size_model: input size_model to be used for diameter prediction/resizing
        """
        reformat = Reformat()
        normalize = Normalize1stTo99th()
        self.evaluate = evaluate
        if isinstance(data_dir, list):  # TODO: Determine how to not treat input as list (if necessary)
            self.d_list = []
            self.l_list = []
            for dir_i in data_dir:
                self.d_list = self.d_list + sorted([dir_i + os.sep + 'data' + os.sep + f for f in
                                                    os.listdir(os.path.join(dir_i, 'data')) if f.lower()
                                                   .endswith('.tiff') or f.lower().endswith('.tif')])
                self.l_list = self.l_list + sorted([dir_i + os.sep + 'labels' + os.sep + f for f in
                                                    os.listdir(os.path.join(dir_i, 'labels')) if f.lower()
                                                   .endswith('.tiff') or f.lower().endswith('.tif')])
        else:
            self.d_list = sorted([data_dir + os.sep + 'data' + os.sep + f for f in os.listdir(os.path.join(
                data_dir, 'data')) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])
            self.l_list = sorted([data_dir + os.sep + 'labels' + os.sep + f for f in os.listdir(os.path.join(
                data_dir, 'labels')) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])
        self.data = []
        self.labels = []
        self.original_dims = []
        if do_3D:  # and from_3D
            for d_file, l_file in tzip(self.d_list, self.l_list, desc='Loading {} Dataset...'.format(split_name)):
                new_data = as_tensor(list(imread(d_file)).astype('float'))
                new_data = reformat(new_data)
                new_data = normalize(new_data)
                new_label = as_tensor(list(imread(l_file)).astype('int16'))
                new_label = reformat(new_label)
                new_data, new_label, original_dim = resize(new_data, new_label)
                self.data.extend(new_data)
                self.labels.extend(new_label)
                self.original_dims.append(original_dim)

        else:
            if from_3D:
                for d_file, l_file in tzip(self.d_list, self.l_list, desc='Loading {} Dataset...'.format(split_name)):
                    raw_vol = imread(d_file).astype('float')
                    new_data = reformat(as_tensor(raw_vol[len(raw_vol)//2]))
                    new_data = normalize(new_data)
                    raw_vol = imread(l_file).astype('int16')
                    new_label = reformat(as_tensor(raw_vol[len(raw_vol) // 2]))
                    new_data, new_label, original_dim = resize(new_data, new_label)
                    self.data.append(new_data)
                    self.labels.append(new_label)
                    self.original_dims.append(original_dim)
            else:
                for d_file, l_file in tzip(self.d_list, self.l_list, desc='Loading {} Dataset...'.format(split_name)):
                    new_data = as_tensor(imread(d_file).astype('float'))
                    new_data = reformat(new_data)
                    new_data = normalize(new_data)
                    new_label = as_tensor(imread(l_file).astype('int16'))
                    new_label = reformat(new_label)
                    new_data, new_label, original_dim = resize(new_data, new_label)
                    self.data.append(new_data)
                    self.labels.append(new_label)
                    self.original_dims.append(original_dim)
        self.data_samples = self.data
        self.label_samples = self.labels

    def __len__(self):
        return len(self.data_samples)

    def reprocess_on_epoch(self, default_meds):
        self.data_samples = tensor([])
        self.label_samples = tensor([])
        for (data, labels) in zip(self.data, self.labels):
            data, labels = random_horizontal_flip(data, labels)
            data, labels = random_rotate(data, labels)
            labels = as_tensor([LabelsToFlows()(labels[i].numpy()) for i in range(len(labels))])
            data, labels = generate_patches(unsqueeze(data, 0), labels)  # TODO: See if unsqueeze can be put in generate patches (check with eval and validation)
            self.data_samples = cat((self.data_samples, data))
            self.label_samples = cat((self.label_samples, labels))

    def pre_generate_patches(self):
        self.data_samples = tensor([])
        self.label_samples = tensor([])
        new_l_list = []
        new_original_dims = []
        for (data, labels, label_fname, original_dim) in zip(self.data, self.labels, self.l_list, self.original_dims):
            data, labels = generate_patches(unsqueeze(data, 0), labels, eval=True)
            self.data_samples = cat((self.data_samples, data))
            self.label_samples = cat((self.label_samples, labels))
            for _ in range(len(data)):
                new_l_list.append(label_fname)
                new_original_dims.append(original_dim)
        self.l_list = new_l_list
        self.original_dims = new_original_dims

    def __getitem__(self, index):
        if self.evaluate:
            return self.data_samples[index], self.label_samples[index], self.l_list[index], self.original_dims[index]
        else:
            return self.data_samples[index], self.label_samples[index]
