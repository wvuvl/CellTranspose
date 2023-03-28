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
from transforms import reformat, normalize1stto99th, train_generate_rand_crop, labels_to_flows, resize_image, padding_2D, calc_median_dim,random_rotate_and_resize, train_generate_rand_crop, diam_range
from cellpose_src import transforms, utils

    
class TrainCellTransposeData(Dataset):
    def __init__(self, data_dirs, n_chan, crop_size=112, batch_size=1, proc_every_epoch=True, 
                 result_dir=None, median_diam=30, target_median_diam=None, target=False, rescale=True, min_train_masks=1):
        
        
        self.diam_train_mean = median_diam
        self.crop_size = crop_size
        self.do_every_epoch = proc_every_epoch
        self.n_chan = n_chan
        self.d_list = []
        self.l_list = []
        self.pf_list = []
        self.save_pf_list = []
        self.min_train_masks = min_train_masks
        self.rescale = rescale
        self.resize_measure = 1.0
        
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
                   
        self.data = []
        self.labels = []
        raw_labels = []
        for index in trange(len(self.d_list), desc='Loading training data...'):
            ext = os.path.splitext(self.d_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
            else:
                raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')
            
            curr_data = normalize1stto99th(reformat(raw_data_vol, self.n_chan))
            self.data.append(curr_data) 
            
            ext = os.path.splitext(self.l_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_lbl_vol = tifffile.imread(self.l_list[index]).astype('float')
            else:
                raw_lbl_vol = cv2.imread(self.l_list[index], -1).astype('float')
            raw_labels.append(raw_lbl_vol)
            self.labels.append(reformat(raw_lbl_vol))
            
            
                    
                
        # cellpose source
        nmasks = np.array([raw_label.max() for raw_label in raw_labels])
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            print(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            self.data = [self.data[i] for i in ikeep]
            self.labels = [self.labels[i] for i in ikeep]
        
        
        
        if self.rescale:
            # self.diam_train[self.diam_train<12.] = 12.
            self.scale_range = 0.5
        else:
            self.scale_range = 1.
        
        # average cell diameter
        # self.diam_train = np.array([utils.diameters(raw_labels[i])[0] for i in range(len(raw_labels))])
        # self.diam_train_mean = self.diam_train[self.diam_train > 0].mean() if not target else median_diam
            
        print(f"Calculated/Given median diams of the train: {self.diam_train_mean}")         
        if target:
            if target_median_diam is None:
                diams = []
                for t_label in raw_labels:
                    diams = diams + diam_range(t_label)
                target_median_diam = np.percentile(np.array(diams), 75) 
            target_median_diam = target_median_diam if target_median_diam > 12. else 12.    
            print(f"Calculated median diams of the target: {target_median_diam}, \nWill get resized to equalize {self.diam_train_mean}")
            self.resize_measure = float(self.diam_train_mean/target_median_diam)
        
            self.target_data_samples = self.data
            self.target_label_samples = self.labels
            for _ in range(1, math.ceil(batch_size / len(self.data))):
                self.data = self.data + self.target_data_samples
                self.labels = self.labels + self.target_label_samples
                
            self.resize_array = np.full(len(self.data), 1./self.resize_measure )    
        else:           
            
            self.diam_train_median_percentile = np.array([np.percentile(np.array(diam_range(raw_labels[i])), 75) for i in trange(len(raw_labels), desc='Calculating Diam...')])
            self.diam_train_median_percentile[self.diam_train_median_percentile<12.] = 12.
            # cellpose original
            # self.resize_array = [float(curr_diam/self.diam_train_mean) if self.rescale else 1.0 for curr_diam in self.diam_train]    
            self.resize_array = [float(curr_diam/self.diam_train_mean) if self.rescale else 1.0 for curr_diam in self.diam_train_median_percentile]
         
            
        
        if not self.do_every_epoch:
            data_samples = []
            label_samples = []
            for index in trange(len(self.data), desc='Preprocessing training data once only...'):
                data, labels = self.process_training_data(index)
                data_samples.append(data)
                label_samples.append(labels)   
            self.data = data_samples
            self.labels = label_samples
                    
    def process_training_data(self, index):
        data, labels = self.data[index], self.labels[index]
        data, labels = random_rotate_and_resize(data, Y=labels[0], rescale=self.resize_array[index], scale_range=self.rescale, xy=(self.crop_size,self.crop_size))
        labels =  labels_to_flows(labels[0])  
        return data.copy(), labels.copy()
        

    def __getitem__(self, index):
        if self.do_every_epoch:
            return self.process_training_data(index)
        else:
            return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class ValCellTransposeData(Dataset):
    def __init__(self, data_dirs, n_chan, patch_size=112, median_diam=30., min_overlap=0.1, augment=False):
    
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
            ext = os.path.splitext(self.l_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_lbl_vol = tifffile.imread(self.l_list[index]).astype('float')
            else:
                raw_lbl_vol = cv2.imread(self.l_list[index], -1).astype('float')
            
            lbl_diam =np.percentile(np.array(diam_range(raw_lbl_vol)), 75) #  utils.diameters(raw_lbl_vol)[0] # 
            resize_measure = median_diam/(lbl_diam if lbl_diam > 12. else 12.)
                
            curr_lbl = reformat(raw_lbl_vol)
            # lable interpolation changed to nearest neighbour from linear
            curr_lbl = resize_image(curr_lbl, rsz=resize_measure, interpolation=cv2.INTER_NEAREST)
            # curr_lbl = labels_to_flows(curr_lbl[0]) # curr_lbl[0] because it has one channel in the front idx 0
            curr_lbl, _, _, _ = padding_2D(curr_lbl, patch_size)
            LBL, _, _, _, _ = transforms.make_tiles(curr_lbl, bsize=patch_size, augment=augment, tile_overlap=min_overlap)
            ny, nx, nchan, ly, lx = LBL.shape
            LBL = np.reshape(LBL, (ny*nx, nchan, ly, lx))
            self.labels = LBL if len(self.labels) == 0 else np.concatenate((self.labels, LBL))

            
            
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
        
        self.labels = np.array([labels_to_flows(self.labels[i][0]) for i in trange(len(self.labels), desc='Computing Val Patch Flows')])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):   
        return self.data[index], self.labels[index]
    

class EvalCellTransposeData(Dataset):
    def __init__(self, data_dirs, n_chan, resize_measure=1.0, median_diam=30.):
        
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
        self.resize_measure_range = []
        
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
                lbl_diam = np.percentile(np.array(diam_range(raw_lbl_vol)), 75) #  utils.diameters(raw_lbl_vol)[0] # 
                self.resize_measure_range.append(float(median_diam/(lbl_diam if lbl_diam > 12. else 12.)))
            else:
                self.resize_measure_range.append(resize_measure)
            
        if len(self.labels) != 0: print("Labels exist, they will be used to resize the image for evaluation...")  
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):   
        return self.data[index], self.d_list[index], self.resize_measure_range[index]

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
