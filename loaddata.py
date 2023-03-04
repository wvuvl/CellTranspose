"""
Data Loader implementation, specifically designed for in-house datasets. Code will be designed to reflect flexibility in
custom data loaders for new data.
"""
import torch
from torch.utils.data import Dataset
from torch import Tensor, empty, as_tensor, tensor, cat, unsqueeze, float32, zeros
import torch.nn.functional as F
import os
import math
from tqdm import tqdm
from tqdm.contrib import tzip
import tifffile
import cv2
import random
import numpy as np
from transforms import reformat, normalize1stto99th, Resize, random_horizontal_flip, labels_to_flows, generate_patches, gaussian_blurr
import copy
from skimage.segmentation import find_boundaries

class CellTransposeData(Dataset):
    """
    Dataset subclass for loading in any tiff data, serving as a superclass to each dataset type for CellTranspose.
    The dataset is expected to possess the following structure:
        - /data
            - vol1.tiff
            ...
            - voln.tiff
        - /labels
            - lbl1.tiff
            ...
            - lbln.tiff
    *** NOTE: Data and labels are expected to be named in such a way that when sorted in ascending order,
    the ith element of data corresponds to the ith label
    """

    def __init__(self, split_name, data_dirs, n_chan, pf_dirs=None, do_3D=False, from_3D=False,
                 evaluate=False, batch_size=1, resize: Resize = None):
        """
        Parameters
        ------------------------------------------------------------------------------------------------

            split_name: name corresponding to the split (i.e. train, validation, test, target)
            data_dirs: root directory/directories of the dataset, containing 'data' and 'labels' folders
            n_chan: Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images)
            pf_dirs: root directory/directories of pre-calculated flows, if they exist
            do_3D: whether or not to train 3D CellTranspose model (requires that from_3d is true)
            from_3D: whether input samples are 2D images (False) or 3D volumes (True)
            evaluate: if set to true, returns additional information when calling __getitem__()
            batch_size: default 1
            resize: Resize object containing parameters by which to resize input samples accordingly
        """
        self.do_3D = do_3D
        self.from_3D = from_3D
        self.split_name = split_name
        self.evaluate = evaluate
        self.d_list_3D = []
        self.l_list_3D = []
        self.d_list = []
        self.l_list = []

        for dir_i in data_dirs:
            self.d_list = self.d_list + sorted([dir_i + os.sep + 'data' + os.sep + f for f in
                                                os.listdir(os.path.join(dir_i, 'data')) if f.lower()
                                               .endswith('.tiff') or f.lower().endswith('.tif')
                                                or f.lower().endswith('.png')])
            
            if os.path.exists(os.path.join(dir_i,'labels')):
                self.l_list = self.l_list + sorted([dir_i + os.sep + 'labels' + os.sep + f for f in
                                                    os.listdir(os.path.join(dir_i, 'labels')) if f.lower()
                                                .endswith('.tiff') or f.lower().endswith('.tif')
                                                    or f.lower().endswith('.png')])
        if pf_dirs is not None:
            self.pf_list = []
            for dir_i in pf_dirs:
                self.pf_list = self.pf_list + sorted([dir_i + os.sep + 'labels' + os.sep + f for f in
                                                      os.listdir(os.path.join(dir_i, 'labels')) if f.lower()
                                                     .endswith('.tiff') or f.lower().endswith('.tif')])
        self.data = []
        self.labels = []
        self.original_dims = []
        self.lbl_len = len(self.l_list)
        
        if self.lbl_len == 0: 
            assert self.evaluate == True,\
                '>>> Folder containing labelled images does not exist, cannot continue without it for training OR validation purposes...'
            print('>>> Folder containing labelled images does not exist, continueing without it for evaluation purposes...')
            
        if from_3D:
            print('>>> Utilizing 2D model for evaluation on 3D volumes...')
        else:
            for ind in tqdm(range(len(self.d_list)), desc='Loading {} Dataset...'.format(split_name)):
                ext = os.path.splitext(self.d_list[ind])[-1]
                if ext == '.tif' or ext == '.tiff':
                    new_data = as_tensor(tifffile.imread(self.d_list[ind]).astype('float'))
                    if self.lbl_len != 0: 
                        new_label = as_tensor(tifffile.imread(self.l_list[ind]).astype('int16'))
                else:
                    new_data = as_tensor(cv2.imread(self.d_list[ind], -1).astype('float'))
                    if self.lbl_len != 0: 
                        new_label = as_tensor(cv2.imread(self.l_list[ind], -1).astype('int16'))
                
                if self.lbl_len != 0 and len(torch.unique(new_label)) > 1:
                    new_data = reformat(new_data, n_chan)
                    new_data = normalize1stto99th(new_data)
                    new_label = reformat(new_label)
                   
                    if pf_dirs is not None:
                        new_pf = tifffile.imread(self.pf_list[ind])
                        new_pf = new_pf.reshape(1, new_pf.shape[0], new_pf.shape[1], new_pf.shape[2])
                    else:
                        new_pf = None
                    if resize is not None:
                        new_data, new_label, original_dim, _ = resize(new_data, new_label, new_pf)
                        self.original_dims.append(original_dim)
                    
                    self.data.append(new_data)
                    self.labels.append(new_label)
                
                elif self.lbl_len == 0:
                    new_data = reformat(new_data, n_chan)
                    new_data = normalize1stto99th(new_data)
                    new_label = []
                    
                    if pf_dirs is not None:
                        new_pf = tifffile.imread(self.pf_list[ind])
                        new_pf = new_pf.reshape(1, new_pf.shape[0], new_pf.shape[1], new_pf.shape[2])
                    else:
                        new_pf = None
                    
                    if resize is not None:
                        new_data, new_label, original_dim, _ = resize(new_data, new_label, new_pf)
                        self.original_dims.append(original_dim)
                    
                    self.data.append(new_data)
                    self.labels.append(new_label)
            
                else:
                    print(f"Found no labels - Skipping this file: {self.d_list[ind]}")
                    
        self.target_data_samples = self.data
        self.target_label_samples = self.labels
        
        if self.split_name.lower() == 'target' and len(self.data) < batch_size and not from_3D and not do_3D:
            for _ in range(1, math.ceil(batch_size / len(self.data))):
                self.data = self.data + self.target_data_samples
                if self.lbl_len != 0: 
                    self.labels = self.labels + self.target_label_samples

        self.data_samples = self.data
        self.label_samples = self.labels

    def __len__(self):
        return len(self.d_list) if (self.do_3D or self.from_3D) else len(self.data)

class TrainCellTransposeData(CellTransposeData):
    def __init__(self, split_name, data_dirs, n_chan, pf_dirs=None, do_3D=False, from_3D=False, evaluate=False,
                 crop_size=(112, 112), has_flows=False, batch_size=1, resize: Resize = None,
                 preprocessed_data=None, proc_every_epoch=True, result_dir=None):
        self.resize = resize
        self.crop_size = crop_size
        self.has_flows = has_flows
        self.from_3D = from_3D
        self.preprocessed_data = preprocessed_data
        self.do_every_epoch = proc_every_epoch
        
        if self.preprocessed_data is None:
            super().__init__(split_name, data_dirs, n_chan, pf_dirs=pf_dirs, do_3D=do_3D,
                             from_3D=from_3D, evaluate=evaluate, batch_size=batch_size, resize=None)
        
        if self.preprocessed_data is not None:
            print('Training preprocessed data provided...')
            self.data = as_tensor(np.load(os.path.join(self.preprocessed_data, 'train_preprocessed_data.npy')))
            self.labels = as_tensor(np.load(os.path.join(self.preprocessed_data, 'train_preprocessed_labels.npy')))
        elif self.do_every_epoch is False and self.preprocessed_data is None:
            
            data_samples = tensor([])
            label_samples = tensor([])
            for i in tqdm(range(len(self.data)), desc='Preprocessing training data once only...'):
                try:
                    data, labels, dim, _ = self.resize(self.data[i], self.labels[i],random_scale=random.uniform(0.75, 1.25))
                    data, labels = random_horizontal_flip(data, labels)
                    data, labels = train_generate_rand_crop(unsqueeze(data, 0), labels,
                                                            crop=crop_size, lbl_flows=has_flows)
                    if labels.ndim == 3:
                        labels = as_tensor(np.array([labels_to_flows(labels[i].numpy()) for i in range(len(labels))]),
                                           dtype=float32)
                    data_samples = cat((data_samples, data))
                    label_samples = cat((label_samples, labels))
                    
                except RuntimeError:
                    print('Caught Size Mismatch.')
            self.data = data_samples
            self.labels = label_samples
            if result_dir is not None: 
                np.save(os.path.join(result_dir, 'train_preprocessed_data.npy'), self.data.cpu().detach().numpy())
                np.save(os.path.join(result_dir, 'train_preprocessed_labels.npy'), self.labels.cpu().detach().numpy())

    # Augmentations and tiling applied to input data (for training and adaptation) -
    # separated from DataLoader to allow for possibility of running only once or once per epoch
    def process_training_data(self, index, crop_size, has_flows=False):
        samples_generated = []
        data, labels = self.data[index], self.labels[index]
        try:
            data, labels, dim, _ = self.resize(data, labels,random_scale=random.uniform(0.75, 1.25))
            data, labels = random_horizontal_flip(data, labels)
            data, labels = train_generate_rand_crop(unsqueeze(data, 0), labels, crop=crop_size, lbl_flows=has_flows)
            if labels.ndim == 3:
                labels = as_tensor(np.array([labels_to_flows(labels[i].numpy())
                                             for i in range(len(labels))]), dtype=float32)
            return data[0], labels[0]
        except RuntimeError:
            print('Caught Size Mismatch.')
            samples_generated.append(-1)

    def __getitem__(self, index):
        if self.preprocessed_data is None and self.do_every_epoch:
            return self.process_training_data(index, self.crop_size, has_flows=self.has_flows)
        else:
            return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    
class TrainCellTransposeData_with_contrast(CellTransposeData):
    def __init__(self, split_name, data_dirs, n_chan, pf_dirs=None, do_3D=False, from_3D=False, evaluate=False,
                 crop_size=(112, 112), has_flows=False, batch_size=1, resize: Resize = None,
                 preprocessed_data=None, proc_every_epoch=True, result_dir=None):
        self.resize = resize
        self.crop_size = crop_size
        self.has_flows = has_flows
        self.from_3D = from_3D
        self.preprocessed_data = preprocessed_data
        self.do_every_epoch = proc_every_epoch
        
        if self.preprocessed_data is None:
            super().__init__(split_name, data_dirs, n_chan, pf_dirs=pf_dirs, do_3D=do_3D,
                             from_3D=from_3D, evaluate=evaluate, batch_size=batch_size, resize=None)
        
        if self.preprocessed_data is not None:
            print('Training preprocessed data provided...')
            self.data = as_tensor(np.load(os.path.join(self.preprocessed_data, 'train_preprocessed_data.npy')))
            self.labels = as_tensor(np.load(os.path.join(self.preprocessed_data, 'train_preprocessed_labels.npy')))
        
        elif self.do_every_epoch is False and self.preprocessed_data is None:
            
            data_samples = tensor([])
            label_samples = tensor([])
            all_org_within_lbls = tensor([])
            all_org_boundaries_lbls = tensor([])
            for i in tqdm(range(len(self.data)), desc='Preprocessing training data once only...'):
                try:
                    data, labels, dim, _ = self.resize(self.data[i], self.labels[i],random_scale=random.uniform(0.75, 1.25))
                    data, labels = random_horizontal_flip(data, labels)
                    
                    data_crop_1, labels_crop_1 = train_generate_rand_crop(unsqueeze(data, 0), labels,
                                                            crop=crop_size, lbl_flows=has_flows)
                    
                    data_crop_2 = gaussian_blurr(copy.deepcopy(data_crop_1))
                    
                    # data_crop_1 = gaussian_blurr(data_crop_1)
                    
                    original_lbl = copy.deepcopy(labels_crop_1)
                    
                    org_within_lbls = []
                    org_boundaries_lbls = []
                    for i in range(len(labels_crop_1)):
                        msk = labels_crop_1[i].numpy()
                        msk_within = copy.deepcopy(msk)
                        msk_boundary = copy.deepcopy(msk)
                        boundary = find_boundaries(msk, mode="thick").astype(np.uint16)
                        indices = np.where(boundary==1)
                        diff_indices = np.where(boundary!=1)
                        msk_within[indices] = 0
                        msk_boundary[diff_indices] = 0
                        org_within_lbls.append(msk_within)
                        org_boundaries_lbls.append(msk_boundary)
                    
                    org_within_lbls = as_tensor(np.array(org_within_lbls), dtype=float32)
                    org_boundaries_lbls = as_tensor(np.array(org_boundaries_lbls), dtype=float32)
            
                    if labels.ndim == 3:
                        labels_crop_1 = as_tensor(np.array([labels_to_flows(labels_crop_1[i].numpy()) for i in range(len(labels_crop_1))]),
                                           dtype=float32)
                    
                    data = cat((data_crop_1, data_crop_2))
                    labels = cat((labels_crop_1, labels_crop_1))
                    
                    data_samples = cat((data_samples, data.unsqueeze(0)))
                    label_samples = cat((label_samples, labels.unsqueeze(0)))
                    all_org_within_lbls = cat((all_org_within_lbls, org_within_lbls.unsqueeze(0)))
                    all_org_boundaries_lbls = cat((all_org_boundaries_lbls, org_boundaries_lbls.unsqueeze(0)))
                except RuntimeError:
                    print('Caught Size Mismatch.')
            self.data = data_samples
            self.labels = label_samples
            self.all_org_within_lbls = all_org_within_lbls
            self.all_org_boundaries_lbls = all_org_boundaries_lbls
            if result_dir is not None: 
                np.save(os.path.join(result_dir, 'train_preprocessed_data.npy'), self.data.cpu().detach().numpy())
                np.save(os.path.join(result_dir, 'train_preprocessed_labels.npy'), self.labels.cpu().detach().numpy())

    # Augmentations and tiling applied to input data (for training and adaptation) -
    # separated from DataLoader to allow for possibility of running only once or once per epoch
    def process_training_data(self, index, crop_size, has_flows=False):
        samples_generated = []
        data, labels = self.data[index], self.labels[index]
        try:
            data, labels, dim, _ = self.resize(data, labels ,random_scale=random.uniform(0.75, 1.25))
            data, labels = random_horizontal_flip(data, labels)
            data_crop_1, labels_crop_1 = train_generate_rand_crop(unsqueeze(data, 0), labels,
                                                            crop=crop_size, lbl_flows=has_flows)
            
            
            data_crop_2 = gaussian_blurr(copy.deepcopy(data_crop_1))
            
            # data_crop_1 = gaussian_blurr(data_crop_1)
            
            original_lbl = copy.deepcopy(labels_crop_1)
            
            org_within_lbls = []
            org_boundaries_lbls = []
            for i in range(len(labels_crop_1)):
                msk = labels_crop_1[i].numpy()
                msk_within = copy.deepcopy(msk)
                msk_boundary = copy.deepcopy(msk)
                boundary = find_boundaries(msk, mode="thick").astype(np.uint16)
                indices = np.where(boundary==1)
                diff_indices = np.where(boundary!=1)
                msk_within[indices] = 0
                msk_boundary[diff_indices] = 0
                org_within_lbls.append(msk_within)
                org_boundaries_lbls.append(msk_boundary)
            
            org_within_lbls = as_tensor(np.array(org_within_lbls), dtype=float32)
            org_boundaries_lbls = as_tensor(np.array(org_boundaries_lbls), dtype=float32)
            
            if labels.ndim == 3:
                labels_crop_1 = as_tensor(np.array([labels_to_flows(labels_crop_1[i].numpy()) for i in range(len(labels_crop_1))]),
                                    dtype=float32)
            
            return data_crop_1[0], data_crop_2[0], labels_crop_1[0], labels_crop_1[0], org_within_lbls, org_boundaries_lbls
        
        except RuntimeError:
            print('Caught Size Mismatch.')
            samples_generated.append(-1)

    def __getitem__(self, index):
        if self.preprocessed_data is None and self.do_every_epoch:
            return self.process_training_data(index, self.crop_size, has_flows=self.has_flows)
        else:
            return self.data[index][0], self.data[index][1], self.labels[index][0], self.labels[index][1], self.all_org_within_lbls[index], self.all_org_boundaries_lbls[index]

    def __len__(self):
        return len(self.data)


def train_generate_rand_crop(data, label=None, crop=(112, 112), lbl_flows=False):
    
    less_labels = True
    while less_labels == True:
        # print("while from train generate rand crop")  
        if data.shape[3] < crop[0]:
            pad_x = math.ceil((crop[0] - data.shape[3]) / 2)
            data = F.pad(data, (pad_x, pad_x))
            label = F.pad(label, (pad_x, pad_x))

        if data.shape[2] < crop[1]:
            pad_y = math.ceil((crop[1] - data.shape[2]) / 2)
            data = F.pad(data, (0, 0, pad_y, pad_y))
            label = F.pad(label, (0, 0, pad_y, pad_y))

        x_max = data.shape[3] - crop[0]
        y_max = data.shape[2] - crop[1]
        patch_data = zeros((data.shape[0] * 1 * 1, data.shape[1], crop[0], crop[1]))
        if lbl_flows:
            patch_label = zeros((1 * 1, 3, crop[0], crop[1]))
        else:
            patch_label = zeros((label.shape[0] * 1 * 1, crop[0], crop[1]))
        i = random.randint(0, x_max)
        j = random.randint(0, y_max)
        d_patch = data[0, :, j:j + crop[1], i:i + crop[0]]
        patch_data[0] = d_patch

        if lbl_flows:
            l_patch = label[:, j:j + crop[1], i:i + crop[0]]
        else:
            l_patch = label[0, j:j + crop[1], i:i + crop[0]]
        patch_label[0] = l_patch

        # if len(torch.unique(patch_label)) < 5: 
        #     print("# of cells encoutered less than 5, recalculating the crop ", torch.unique(patch_label))
        # else:        
        #     less_labels = False

        
        if len(torch.unique(patch_label)) > 1: less_labels = False
        # else: print("# of cells encoutered less than 1, recalculating the crop ", torch.unique(patch_label))
        
        if  True in torch.isnan(patch_label) or True in  torch.isnan(patch_data):
            # less_labels = False 
            print("found nan value")
        
    return patch_data, patch_label


class EvalCellTransposeData(CellTransposeData):
    def __init__(self, split_name, data_dirs, n_chan, pf_dirs=None, do_3D=False, from_3D=False,
                 evaluate=False, resize: Resize = None, ):
        self.from_3D = from_3D
        super().__init__(split_name, data_dirs, n_chan, pf_dirs=pf_dirs, do_3D=do_3D, from_3D=from_3D,
                         evaluate=evaluate, resize=resize)

    # Generates patches for validation dataset - only happens once
    def pre_generate_validation_patches(self, patch_size, min_overlap):
        self.data_samples = tensor([])
        self.label_samples = tensor([])
        new_d_list = []
        new_original_dims = []
        for (data, labels, data_fname, original_dim) in tzip(self.data, self.labels, self.d_list, self.original_dims,
                                                              desc='Processing Validation Dataset...'):
            

            

            if data.shape[1] >= patch_size[0] and data.shape[2] >= patch_size[1]:
                if len(labels) != 0: 
                    data, labels = generate_patches(unsqueeze(data, 0), labels, patch=patch_size,
                                                min_overlap=min_overlap, lbl_flows=False)
                    labels = as_tensor(np.array([labels_to_flows(labels[i].numpy()) for i in range(len(labels))]))
                else:
                     data = generate_patches(unsqueeze(data, 0), patch=patch_size,
                                                min_overlap=min_overlap, lbl_flows=False)

                self.data_samples = cat((self.data_samples, data))
                self.label_samples = cat((self.label_samples, labels))

                for _ in range(len(data)):
                    new_d_list.append(data_fname)
                    new_original_dims.append(original_dim)
        self.d_list = new_d_list
        self.original_dims = new_original_dims

    def __getitem__(self, index):
        if len(self.label_samples) == 0:
            label = []
        else:
            label = self.label_samples[index]
        if self.evaluate and not self.from_3D:
            return self.data_samples[index], label, self.d_list[index], self.original_dims[index]
        else:
            return self.data_samples[index], label


# final version of 3D validation dataloader
class EvalCellTransposeData3D(CellTransposeData):
    def __init__(self, split_name, data_dirs, n_chan, do_3D=False,
                 from_3D=False, evaluate=False, resize: Resize = None, anisotropy = (1.0, 1.0, 1.0)):
        self.resize = resize
        self.n_chan = n_chan
        self.anisotropy = anisotropy
        
        super().__init__(split_name, data_dirs, n_chan, do_3D=do_3D, from_3D=from_3D, evaluate=evaluate, resize=resize)
    
    def process_eval_3D(self, index):
        ext = os.path.splitext(self.d_list[index])[-1]

        if ext == '.tif' or ext == '.tiff':
            raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
        else:
            raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')

        axis = ('Z', 'Y', 'X')
        plane = ('YX', 'ZX', 'ZY')
        TP = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        data_vol = []
        original_dim = []
        
        print(f">>> Image path: {self.d_list[index]}")
        for ind in range(len(plane)):
            new_data = raw_data_vol.transpose(TP[ind])
            print(f">>> Processing 3D data on {new_data.shape[0]} {plane[ind]} planes in {axis[ind]} direction...")
            new_data_vol = []
            new_origin_dim = []
            for i in range(len(new_data)):
                d = reformat(as_tensor(new_data[i]), self.n_chan)
                data = normalize1stto99th(d)
                label = []
                if self.resize is not None:
                    data, label, dim, diam = self.resize(data, label, anisotropy=self.anisotropy[ind])
                else:
                    dim = (data[0], data[1])
                new_data_vol.append(data)
                new_origin_dim.append(dim)
            data_vol.append(new_data_vol)
            original_dim.append(new_origin_dim)
        return data_vol, self.d_list[index], plane, original_dim, diam

    def __getitem__(self, index):
        return self.process_eval_3D(index)
