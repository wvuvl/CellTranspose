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
from transforms import reformat, normalize1stto99th, train_generate_rand_crop, labels_to_flows, resize_image, padding_2D, calc_median_dim,random_rotate_and_resize, train_generate_rand_crop
from cellpose_src import transforms, utils

    
class TrainCellTransposeData(Dataset):
    def __init__(self, data_dirs, n_chan, crop_size=112, flows_available=False, batch_size=1, proc_every_epoch=True, 
                 result_dir=None, median_diam=30, target_median_diam=None, target=False, rescale=True, min_train_masks=1):
        
        

        self.crop_size = crop_size
        self.flows_available = flows_available
        self.do_every_epoch = proc_every_epoch
        self.n_chan = n_chan
        self.d_list = []
        self.l_list = []
        self.pf_list = []
        self.save_pf_list = []
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
            
            if flows_available and os.path.exists(os.path.join(dir_i,'flows')):
                self.pf_list = self.pf_list + sorted([dir_i + os.sep + 'flows' + os.sep + f for f in
                                                    os.listdir(os.path.join(dir_i, 'flows')) if f.lower()
                                                .endswith('.tiff') or f.lower().endswith('.tif')
                                                    or f.lower().endswith('.png')])   
            else:
                flow_dir = os.path.join(dir_i,'flows')
                os.makedirs(flow_dir)
                for lbl_name in self.l_list:
                    self.save_pf_list.append(os.path.join(flow_dir,os.path.splitext(os.path.basename(lbl_name))[0]+".tif"))
                print(f"Flows directory {os.path.join(dir_i,'flows')} does not exist, \n    will continue without loading flows and will compute")  
                   
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
            
            
            if len(self.pf_list)>0:
                ext = os.path.splitext(self.pf_list[index])[-1]
                if ext == '.tif' or ext == '.tiff':
                    flow_lbl = tifffile.imread(self.pf_list[index]).astype('float')
                else:
                    flow_lbl = cv2.imread(self.pf_list[index], -1).astype('float')   
                
            else:
                flow_lbl = labels_to_flows(raw_lbl_vol)
                tifffile.imwrite(self.save_pf_list[index], flow_lbl)
                
            flow_lbl = reformat(flow_lbl, n_chan=flow_lbl.shape[0])   
            self.labels.append(flow_lbl)
            
                    
                
        # cellpose source
        nmasks = np.array([raw_label.max() for raw_label in raw_labels])
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            print(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            self.data = [self.data[i] for i in ikeep]
            self.labels = [self.labels[i] for i in ikeep]
        
                   
        if target:
            if target_median_diam is None:
                target_median_diam = calc_median_dim(raw_labels)
            
            print(f"Calculated median diams of the target: {target_median_diam}, \nWill get resized to equalize {median_diam}")
            self.resize_measure = float(median_diam/target_median_diam)
        
            self.target_data_samples = self.data
            self.target_label_samples = self.labels
            for _ in range(1, math.ceil(batch_size / len(self.data))):
                self.data = self.data + self.target_data_samples
                self.labels = self.labels + self.target_label_samples

        # average cell diameter
        self.diam_train = np.array([utils.diameters(raw_labels[i])[0] for i in range(len(raw_labels))])
        self.diam_train_mean = self.diam_train[self.diam_train > 0].mean() if not target else median_diam
        
        if self.rescale:
            self.diam_train[self.diam_train<5] = 5.
            self.scale_range = 0.25
            print(f'Median diameter {self.diam_train_mean}')
        else:
            self.scale_range = 0
        
        # cellpose original
        self.resize_array = [float(curr_diam/self.diam_train_mean) if self.rescale else 1.0 for curr_diam in self.diam_train]    
        # self.resize_array = [float(self.diam_train_mean/curr_diam) if self.rescale else 1.0 for curr_diam in self.diam_train]     
        
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
        
                
        # # random horizontal flip
        # if np.random.rand() > .5:
        #     data = np.fliplr(data).copy()
        #     labels = np.fliplr(labels).copy()
        
        # rand_scale = self.resize_array[index] / np.random.uniform(1.-self.scale_range, 1.+self.scale_range)
        # data = resize_image(data, rsz=rand_scale)
        # labels_flows = resize_image(labels[1:,:,:], rsz=rand_scale)
        # labels_mask = resize_image(labels[:1, :, :], rsz=rand_scale, interpolation=cv2.INTER_NEAREST)
        # labels = np.concatenate((labels_mask, labels_flows))
        # data, labels = train_generate_rand_crop(data, labels, crop=self.crop_size)
        
        data, labels = random_rotate_and_resize(data, Y=labels, rescale=self.resize_array[index], scale_range=self.scale_range, xy=(self.crop_size,self.crop_size))
        
        return data, labels
        

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
            
            lbl_diam = utils.diameters(raw_lbl_vol)[0]
            resize_measure = median_diam/lbl_diam if lbl_diam > 5. else 5.
                
            curr_lbl = reformat(raw_lbl_vol)
            # lable interpolation changed to nearest neighbour from linear
            curr_lbl = resize_image(curr_lbl, rsz=resize_measure, interpolation=cv2.INTER_NEAREST)
            curr_lbl = labels_to_flows(curr_lbl[0]) # curr_lbl[0] because it has one channel in the front idx 0
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
            
            
        
        # should work since the shape of the data will be the same
        self.data = np.asarray(self.data)
        self.labels = np.asarray(self.labels)
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
                lbl_diam = utils.diameters(raw_lbl_vol)[0]
                self.resize_measure_range.append(float(median_diam/lbl_diam if lbl_diam > 5. else 5. ))
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
