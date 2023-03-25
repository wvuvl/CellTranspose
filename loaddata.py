"""
Data Loader implementation, specifically designed for in-house datasets. Code will be designed to reflect flexibility in
custom data loaders for new data.
"""
from torch.utils.data import Dataset
import os
import math
from tqdm import trange
import tifffile
import cv2
import numpy as np
from transforms import reformat, normalize1stto99th, train_generate_rand_crop, labels_to_flows, resize_image, padding_2D, calc_median_dim, random_rotate_and_resize
from cellpose_src import transforms, utils

    
class TrainCellTransposeData(Dataset):
    def __init__(self, data_dirs, n_chan, crop_size=112, has_flows=False, batch_size=1, preprocessed_data=None, proc_every_epoch=True, 
                 result_dir=None, median_diam=30, target_median_diam=None, target=False, rescale=True, min_train_masks=1):
        
        

        self.crop_size = crop_size
        self.has_flows = has_flows
        self.preprocessed_data = preprocessed_data
        self.do_every_epoch = proc_every_epoch
        self.n_chan = n_chan
        self.d_list = []
        self.l_list = []
        self.min_train_masks = min_train_masks
        self.median_diam = median_diam
        self.rescale = rescale
        
        for dir_i in data_dirs:
            assert os.path.exists(os.path.join(dir_i,'labels')), f"Training folder {os.path.join(dir_i,'labels')} does not exists, it is needed for training."
            self.d_list = self.d_list + sorted([dir_i + os.sep + 'data' + os.sep + f for f in
                                                os.listdir(os.path.join(dir_i, 'data')) if f.lower()
                                            .endswith('.tiff') or f.lower().endswith('.tif')
                                                or f.lower().endswith('.png')])
            
            
            self.l_list = self.l_list + sorted([dir_i + os.sep + 'labels' + os.sep + f for f in
                                                os.listdir(os.path.join(dir_i, 'labels')) if f.lower()
                                            .endswith('.tiff') or f.lower().endswith('.tif')
                                                or f.lower().endswith('.png')])
            
        
        if self.preprocessed_data is not None:
            print('Training preprocessed data provided...')
            self.data = np.load(os.path.join(self.preprocessed_data, 'train_preprocessed_data.npy'))
            self.labels = np.load(os.path.join(self.preprocessed_data, 'train_preprocessed_labels.npy'))
            
        else:
            self.data = []
            self.labels = []
            for index in trange(len(self.d_list), desc='Loading training data...'):
                ext = os.path.splitext(self.d_list[index])[-1]
                if ext == '.tif' or ext == '.tiff':
                    raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
                else:
                    raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')
                
                curr_data = normalize1stto99th(reformat(raw_data_vol, self.n_chan))
                # curr_data = resize_image(curr_data, rsz=self.resize_measure)
                curr_data, _, _, _ = padding_2D(curr_data, self.crop_size)
                self.data.append(curr_data) 
                
                ext = os.path.splitext(self.l_list[index])[-1]
                if ext == '.tif' or ext == '.tiff':
                    raw_lbl_vol = tifffile.imread(self.l_list[index]).astype('float')
                else:
                    raw_lbl_vol = cv2.imread(self.l_list[index], -1).astype('float')
                    
                curr_lbl = reformat(raw_lbl_vol)    
                # curr_lbl = resize_image(curr_lbl, rsz=self.resize_measure, interpolation=cv2.INTER_NEAREST)
                curr_lbl, _, _, _ = padding_2D(curr_lbl, self.crop_size)
                curr_lbl = curr_lbl if has_flows else labels_to_flows(curr_lbl[0])
                self.labels.append(curr_lbl) 
                     

        # cellpose source
        nmasks = np.array([label.max() for label in self.labels])
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            print(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            self.data = [self.data[i] for i in ikeep]
            self.labels = [self.labels[i] for i in ikeep]
        
                   
        if target:
            if target_median_diam is None:
                target_median_diam = calc_median_dim(self.labels)
            
            print(f"Calculated median diams of the target: {target_median_diam}, \nWill get resized to equalize {median_diam}")
            self.resize_measure = float(median_diam/target_median_diam)
        
            self.target_data_samples = self.data
            self.target_label_samples = self.labels
            for _ in range(1, math.ceil(batch_size / len(self.data))):
                self.data = self.data + self.target_data_samples
                self.labels = self.labels + self.target_label_samples

        # average cell diameter
        self.diam_train = np.array([utils.diameters(self.labels[i][0])[0] for i in range(len(self.labels))])
        self.diam_train_mean = self.diam_train[self.diam_train > 0].mean() if not target else median_diam
        
        if self.rescale:
            self.diam_train[self.diam_train<5] = 5.
            self.scale_range = 0.5
            print(f'Median diameter {self.diam_train_mean}')
        else:
            self.scale_range = 1.0    
        self.resize_array = [float(curr_diam/self.diam_train_mean) if self.rescale else 1.0 for curr_diam in self.diam_train]     
        
        if not self.do_every_epoch:
            data_samples = []
            label_samples = []
            for index in trange(len(self.data), desc='Preprocessing training data once only...'):
                data, labels = self.process_training_data(index, self.crop_size, has_flows=self.has_flows)
                data_samples.append(data)
                label_samples.append(labels)   
            self.data = data_samples
            self.labels = label_samples

            if result_dir is not None: 
                np.save(os.path.join(result_dir, 'train_preprocessed_data.npy'), self.data)
                np.save(os.path.join(result_dir, 'train_preprocessed_labels.npy'), self.labels)
                    
    def process_training_data(self, index, crop_size, has_flows=False):
        samples_generated = []
        data, labels = self.data[index], self.labels[index]
        
                
        # # random horizontal flip
        # if np.random.rand() > .5:
        #     data = np.fliplr(data).copy()
        #     labels = np.fliplr(labels).copy()
        # data, labels = train_generate_rand_crop(data, labels, crop=crop_size)
        # labels = labels if has_flows else labels_to_flows(labels[0]) # labels[0] because it has one channel in the front idx 0
        
        data, labels = random_rotate_and_resize(data, Y=labels, rescale=self.resize_array[index], scale_range=self.scale_range, xy=(crop_size,crop_size))
        return data, labels
        

    def __getitem__(self, index):
        if self.preprocessed_data is None and self.do_every_epoch:
            return self.process_training_data(index, self.crop_size, has_flows=self.has_flows)
        else:
            return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class ValCellTransposeData(Dataset):
    def __init__(self, data_dirs, n_chan, patch_size=112, resize_measure=1.0, min_overlap=0.1, augment=False):
        
        
    
        self.resize_measure = resize_measure
        self.n_chan = n_chan
        self.d_list = []
        self.l_list = []
        
        for dir_i in data_dirs:
            assert os.path.exists(os.path.join(dir_i,'labels')), f"Validation folder {os.path.join(dir_i,'labels')} does not exists, it is needed for validation."
            self.d_list = self.d_list + sorted([dir_i + os.sep + 'data' + os.sep + f for f in
                                                os.listdir(os.path.join(dir_i, 'data')) if f.lower()
                                            .endswith('.tiff') or f.lower().endswith('.tif')
                                                or f.lower().endswith('.png')])
            
            
            self.l_list = self.l_list + sorted([dir_i + os.sep + 'labels' + os.sep + f for f in
                                                os.listdir(os.path.join(dir_i, 'labels')) if f.lower()
                                            .endswith('.tiff') or f.lower().endswith('.tif')
                                                or f.lower().endswith('.png')])
        
        self.data = np.array([])
        self.labels = np.array([])
        for index in trange(len(self.d_list), desc='Loading and Processing Validation Dataset...'):
            ext = os.path.splitext(self.d_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
            else:
                raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')    
            curr_data = normalize1stto99th(reformat(raw_data_vol, self.n_chan))
            curr_data = resize_image(curr_data, rsz=resize_measure)
            curr_data, _, _, _ = padding_2D(curr_data, patch_size)
            IMG, _, _, _, _ = transforms.make_tiles(curr_data, bsize=patch_size, augment=augment, tile_overlap=min_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
            self.data = IMG if len(self.data) == 0 else np.concatenate((self.data, IMG))
            
            ext = os.path.splitext(self.l_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_lbl_vol = tifffile.imread(self.l_list[index]).astype('float')
            else:
                raw_lbl_vol = cv2.imread(self.l_list[index], -1).astype('float')
            
            curr_lbl = reformat(raw_lbl_vol)
            # lable interpolation changed to nearest neighbour from linear
            curr_lbl = resize_image(curr_lbl, rsz=resize_measure, interpolation=cv2.INTER_NEAREST)
            curr_lbl = labels_to_flows(curr_lbl[0]) # curr_lbl[0] because it has one channel in the front idx 0
            curr_lbl, _, _, _ = padding_2D(curr_lbl, patch_size)
            LBL, _, _, _, _ = transforms.make_tiles(curr_lbl, bsize=patch_size, augment=augment, tile_overlap=min_overlap)
            ny, nx, nchan, ly, lx = LBL.shape
            LBL = np.reshape(LBL, (ny*nx, nchan, ly, lx))
            self.labels = LBL if len(self.labels) == 0 else np.concatenate((self.labels, LBL))
        
        # should work since the shape of the data will be the same
        self.data = np.asarray(self.data)
        self.labels = np.asarray(self.labels)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):   
        return self.data[index], self.labels[index]
    

class EvalCellTransposeData(Dataset):
    def __init__(self, data_dirs, n_chan, resize_measure=1.0):
        
        self.resize_measure = resize_measure
        self.n_chan = n_chan
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
        self.data = []
        self.labels = []
        for index in range(len(self.d_list)):
            ext = os.path.splitext(self.d_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
            else:
                raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')
            self.data.append(normalize1stto99th(reformat(raw_data_vol, self.n_chan))) 
            
            if os.path.exists(os.path.join(dir_i,'labels')):
                ext = os.path.splitext(self.l_list[index])[-1]
                if ext == '.tif' or ext == '.tiff':
                    raw_lbl_vol = tifffile.imread(self.l_list[index]).astype('float')
                else:
                    raw_lbl_vol = cv2.imread(self.l_list[index], -1).astype('float')
                self.labels.append(raw_lbl_vol) 

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):   
        return self.data[index], self.d_list[index], self.resize_measure

# Updated more efficient version    
class EvalCellTransposeData3D(Dataset):
    def __init__(self, data_dirs, n_chan, resize_measure=1.0):
        self.resize_measure = resize_measure
        self.n_chan = n_chan
        self.d_list = []
        
        for dir_i in data_dirs:
            self.d_list = self.d_list + sorted([dir_i + os.sep + 'data' + os.sep + f for f in
                                                os.listdir(os.path.join(dir_i, 'data')) if f.lower()
                                            .endswith('.tiff') or f.lower().endswith('.tif')
                                                or f.lower().endswith('.png')])
    
    def __len__(self):
        return len(self.d_list)
    
    def __getitem__(self, index):
        ext = os.path.splitext(self.d_list[index])[-1]

        if ext == '.tif' or ext == '.tiff':
            raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
        else:
            raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')
        
        print(f">>> Image path: {self.d_list[index]}")
        print(f">>> Processing 3D data")
                
        data = normalize1stto99th(reformat(raw_data_vol, self.n_chan, do_3D=True))
        return data, self.d_list[index], self.resize_measure
