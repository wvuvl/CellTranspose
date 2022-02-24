"""
Data Loader implementation, specifically designed for in-house datasets. Code will be designed to reflect flexibility in
custom dataloaders for new data.
"""
import numpy as np
from torch.utils.data import Dataset
from torch import empty, as_tensor, tensor, cat, unsqueeze, float32
import torch.nn.functional as F
import os
import math
from tqdm import tqdm
from tqdm.contrib import tzip
import tifffile
import cv2
import random
from transforms import reformat, normalize1stto99th, Resize, random_horizontal_flip, labels_to_flows, generate_patches

import matplotlib.pyplot as plt


class CellTransposeData(Dataset):
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
    def __init__(self, split_name, data_dirs, n_chan, pf_dirs=None, do_3D=False, from_3D=False, evaluate=False, batch_size = 1,
                 resize: Resize = None):
        """"
        Args:
            split_name: name corresponding to the split (i.e. train, validation, test, target)
            n_chan: Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images)
            data_dirs: root directory/directories of the dataset, containing 'data' and 'labels' folders
            pf_dirs: root directory/directories of pre-calculated flows, if they exist
            do_3D: whether or not to train 3D CellTranspose model (requires that from_3d is true)
            from_3D: whether input samples are 2D images (False) or 3D volumes (True)
            evaluate: if set to true, returns additional information when calling __getitem__()
            resize: Resize object containing parameters by which to resize input samples accordingly
        """
        self.split_name = split_name
        self.evaluate = evaluate
        if isinstance(data_dirs, list):  # TODO: Determine how to not treat input as list (if necessary)
            self.d_list = []
            self.l_list = []
            for dir_i in data_dirs:
                self.d_list = self.d_list + sorted([dir_i + os.sep + 'data' + os.sep + f for f in
                                                    os.listdir(os.path.join(dir_i, 'data')) if f.lower()
                                                   .endswith('.tiff') or f.lower().endswith('.tif')
                                                    or f.lower().endswith('.png')])
                self.l_list = self.l_list + sorted([dir_i + os.sep + 'labels' + os.sep + f for f in
                                                    os.listdir(os.path.join(dir_i, 'labels')) if f.lower()
                                                   .endswith('.tiff') or f.lower().endswith('.tif')
                                                    or f.lower().endswith('.png')])
        else:
            self.d_list = sorted([data_dirs + os.sep + 'data' + os.sep + f for f in os.listdir(os.path.join(
                data_dirs, 'data')) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
                                  or f.lower().endswith('.png')])
            self.l_list = sorted([data_dirs + os.sep + 'labels' + os.sep + f for f in os.listdir(os.path.join(
                data_dirs, 'labels')) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
                                  or f.lower().endswith('.png')])
        if pf_dirs is not None:
            if isinstance(pf_dirs, list):  # TODO: Determine how to not treat input as list (if necessary)
                self.pf_list = []
                for dir_i in pf_dirs:
                    self.pf_list = self.pf_list + sorted([dir_i + os.sep + 'labels' + os.sep + f for f in
                                                          os.listdir(os.path.join(dir_i, 'labels')) if f.lower()
                                                         .endswith('.tiff') or f.lower().endswith('.tif')])
            else:
                self.pf_list = sorted([pf_dirs + os.sep + 'labels' + os.sep + f for f in os.listdir(os.path.join(
                    pf_dirs, 'labels')) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')])
        self.data = []
        self.labels = []
        self.original_dims = []
        if do_3D:  # and from_3D
            for ind in tqdm(range(len(self.d_list)), desc='Loading {} Dataset...'.format(split_name)):
                ext = os.path.splitext(self.d_list[ind])[-1]
                if ext == '.tif' or ext == '.tiff':
                    new_data = as_tensor(list(tifffile.imread(self.d_list[ind]).astype('float')))
                    new_label = as_tensor(list(tifffile.imread(self.l_list[ind])).astype('int16'))
                else:
                    new_data = as_tensor(list(cv2.imread(self.d_list[ind], -1).astype('float')))
                    new_label = as_tensor(list(cv2.imread(self.l_list[ind], -1).astype('int16')))
                new_data = reformat(new_data, n_chan, do_3D=True)
                new_data = normalize1stto99th(new_data)
                new_label = reformat(new_label, do_3D=True)
                if pf_dirs is not None:
                    new_pf = as_tensor(list(tifffile.imread(self.pf_list[ind])))
                    new_pf = reformat(new_pf, is_pf=True, do_3D=True)  # TODO: May/may not need updated to reflect do_3D loading
                else:
                    new_pf = None
                if resize is not None:
                    new_data, new_label, new_pf, original_dim = resize(new_data, new_label, new_pf)
                    self.original_dims.append(original_dim)
                self.data.extend(new_data)
                self.labels.extend(new_label)

        else:
            if from_3D:
                # Note: majority of bottleneck caused by reading and normalizing data
                for ind in tqdm(range(len(self.d_list)), desc='Loading {} Dataset...'.format(split_name)):
                    ext = os.path.splitext(self.d_list[ind])[-1]
                    if ext == '.tif' or ext == '.tiff':
                        raw_data_vol = tifffile.imread(self.d_list[ind]).astype('float')
                        raw_label_vol = tifffile.imread(self.l_list[ind]).astype('int16')
                    else:
                        raw_data_vol = cv2.imread(self.d_list[ind], -1).astype('float')
                        raw_label_vol = cv2.imread(self.l_list[ind], -1).astype('int16')
                    raw_data_vol = [reformat(as_tensor(raw_data_vol[i]), n_chan) for i in range(len(raw_data_vol))]
                    raw_data_vol = [normalize1stto99th(raw_data_vol[i]) for i in range(len(raw_data_vol))]
                    raw_label_vol = [reformat(as_tensor(raw_label_vol[i])) for i in range(len(raw_label_vol))]
                    if pf_dirs is not None:  # Not currently handled
                        print('Add this later')
                        # if resize is not None:
                        #     *do_resize_here*
                    else:
                        if resize is not None:
                            new_data = []
                            new_label = []
                            original_dim = []
                            for i in range(len(raw_data_vol)):
                                nd, nl, _, od = resize(raw_data_vol[i], raw_label_vol[i])
                                new_data.append(nd)
                                new_label.append(nl)
                                original_dim.append(od)
                            self.original_dims.extend(original_dim)
                    self.data.extend(new_data)
                    self.labels.extend(new_label)

            else:
                for ind in tqdm(range(len(self.d_list)), desc='Loading {} Dataset...'.format(split_name)):
                    ext = os.path.splitext(self.d_list[ind])[-1]
                    if ext == '.tif' or ext == '.tiff':
                        new_data = as_tensor(tifffile.imread(self.d_list[ind]).astype('float'))
                        new_label = as_tensor(tifffile.imread(self.l_list[ind]).astype('int16'))
                    else:
                        new_data = as_tensor(cv2.imread(self.d_list[ind], -1).astype('float'))
                        new_label = as_tensor(cv2.imread(self.l_list[ind], -1).astype('int16'))
                    new_data = reformat(new_data, n_chan)
                    new_data = normalize1stto99th(new_data)
                    new_label = reformat(new_label)
                    if pf_dirs is not None:
                        new_pf = tifffile.imread(self.pf_list[ind])
                        new_pf = reformat(new_pf, is_pf=True)
                    else:
                        new_pf = None
                    if resize is not None:
                        new_data, new_label, original_dim = resize(new_data, new_label, new_pf)
                        self.original_dims.append(original_dim)
                    self.data.append(new_data)
                    self.labels.append(new_label)

        if self.split_name.lower() == 'target' and len(self.data) < batch_size:
            ds = self.data
            ls = self.labels
            for _ in range(1, math.ceil(batch_size / len(self.data))):
                self.data = self.data+ds
                self.labels = self.labels+ls
        
        self.data_samples = self.data
        self.label_samples = self.labels
    def __len__(self):
        return len(self.data) #return len(self.data_samples)


class TrainCellTransposeData(CellTransposeData):
    def __init__(self, split_name, data_dirs, n_chan, pf_dirs=None, do_3D=False, from_3D=False, evaluate=False,
                 crop_size=(96, 96), has_flows=False, batch_size = 1, resize: Resize=None):
        self.resize = resize
        self.crop_size = crop_size
        self.has_flows = has_flows
        super().__init__(split_name, data_dirs, n_chan, pf_dirs=pf_dirs, do_3D=do_3D,
                         from_3D=from_3D, evaluate=evaluate, batch_size=batch_size, resize=None)

    def train_generate_rand_crop(self, data, label=None, crop=(96, 96), lbl_flows=False):
        if data.shape[3] < crop[0]: 
            pad_x = math.ceil((crop[0]-data.shape[3])/2)
            data = F.pad(data, (pad_x, pad_x))
            label = F.pad(label, (pad_x, pad_x))

        if data.shape[2] < crop[1]:
            pad_y = math.ceil((crop[1]-data.shape[2])/2)
            data = F.pad(data, (0, 0, pad_y, pad_y))
            label = F.pad(label, (0, 0, pad_y, pad_y))

        x_max = data.shape[3] - crop[0]
        y_max = data.shape[2] - crop[1]
        
        patch_data = empty((data.shape[0] * 1 * 1, data.shape[1], crop[0], crop[1]))
        if lbl_flows:
            patch_label = empty((1 * 1, 3, crop[0], crop[1]))
        else:
            patch_label = empty((label.shape[0] * 1 * 1, crop[0], crop[1]))

        i = random.randint(0, x_max)
        j = random.randint(0, y_max)

        d_patch = data[0, :, j:j + crop[1], i:i + crop[0]]
        
        patch_data[0] = d_patch

        if lbl_flows:
            l_patch = label[:, j:j + crop[1], i:i + crop[0]]
        else:
            l_patch = label[0, j:j + crop[1], i:i + crop[0]]
        patch_label[0] = l_patch

        return patch_data, patch_label
    
    # Augmentations and tiling applied to input data (for training and adaptation) -
    # separated from DataLoader to allow for possibility of running only once or once per epoch
    # NOTE: ltf takes ~50% of time; generating patches and concatenating takes nearly as long
    # TODO: Save generated training data? Massively increase time to train
    def process_training_data(self, index, crop_size, has_flows=False):
        #self.data_samples = tensor([])
        #self.label_samples = tensor([])
        samples_generated = []
        data, labels = self.data[index], self.labels[index]
        
        # for (data, labels) in tzip(self.data, self.labels, desc='Processing {} Dataset...'.format(self.split_name)):
        try:
            data, labels, dim = self.resize(data, labels, random_scale=random.uniform(0.75, 1.25))
            data, labels = random_horizontal_flip(data, labels)
            # data, labels = random_rotate(data, labels)
            
            # print(data.shape," ",labels.shape)
            data, labels = self.train_generate_rand_crop(unsqueeze(data, 0), labels, crop=crop_size, lbl_flows=has_flows)

            if labels.ndim == 3:
                labels = as_tensor(np.array([labels_to_flows(labels[i].numpy()) for i in range(len(labels))]), dtype=float32)
           
            return data[0], labels[0]
        except RuntimeError:
            print('Caught Size Mismatch.')
            samples_generated.append(-1)

    def __getitem__(self, index):
        return self.process_training_data(index, self.crop_size, has_flows=self.has_flows)
        # self.data[index], self.labels[index] #return self.data_samples[index], self.label_samples[index]


class ValTestCellTransposeData(CellTransposeData):
    def __init__(self, split_name, data_dirs, n_chan, pf_dirs=None, do_3D=False, from_3D=False, evaluate=False,
                 resize: Resize = None):
        super().__init__(split_name, data_dirs, n_chan, pf_dirs=pf_dirs, do_3D=do_3D, from_3D=from_3D,
                         evaluate=evaluate, resize=resize)

    # Generates patches for validation dataset - only happens once
    def pre_generate_validation_patches(self, patch_size, min_overlap):
        self.data_samples = tensor([])
        self.label_samples = tensor([])
        new_l_list = []
        new_original_dims = []
        for (data, labels, label_fname, original_dim) in tzip(self.data, self.labels, self.l_list, self.original_dims,
                                                              desc='Processing Validation Dataset...'):
            if data.shape[1] >= patch_size[0] and data.shape[2] >= patch_size[1]:
                data, labels = generate_patches(unsqueeze(data, 0), labels, patch=patch_size,
                                                min_overlap=min_overlap, lbl_flows=False)
                labels = as_tensor(np.array([labels_to_flows(labels[i].numpy()) for i in range(len(labels))]))
                # data, labels = remove_empty_label_patches(data, labels)
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
