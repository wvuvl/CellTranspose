"""
Data Loader implementation, specifically designed for in-house datasets. Code will be designed to reflect flexibility in
custom data loaders for new data.
"""
import copy
import os
import math
import tifffile
import cv2
import numpy as np
from math import ceil
from tqdm import trange, tqdm
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries

# local import
import transforms

# cellpose_src imports
from cellpose_src import transforms as cp_transform
    
class TrainCellTransposeData(Dataset):
    """
    Dataset subclass for loading in any tiff data.
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
    
    def __init__(self, data_dirs, n_chan, crop_size=112, batch_size=1, proc_every_epoch=True, 
                 median_diam=30, target_median_diam=None, target=False, rescale=True, min_train_masks=1):
        
        """
            Parameters
            ------------------------------------------------------------------------------------------------
            data_dirs: root directory/directories of the dataset, containing 'data' and 'labels' folders
            n_chan: Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images)
            crop_size: crop size to use for random cropping, default 112
            batch_size: batch size parameter
            proc_every_epoch: process random cropping and augmentation every epoch, default True
            median_diam: median diam for resizing training data, default 30
            target_median_diam: target median diam for resizing target data if provided
            target: provided data if target, True, default False
            rescale: rescaling boolean parameter for training to be rescaled randomly, default true
            min_train_masks: min masks each crop must have, default 1
        """
        
        self.diam_train_mean = median_diam
        self.crop_size = crop_size
        self.do_every_epoch = proc_every_epoch
        self.n_chan = n_chan
        self.min_train_masks = min_train_masks
        self.resize_measure = 1.0
        self.target_median_diam = target_median_diam
        self.scale_range = 0.5 if rescale else 1.
        
        self.d_list = []
        self.l_list = []
        self.pf_list = []
        self.save_pf_list = []

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
            
            curr_data = transforms.normalize1stto99th(transforms.reformat(raw_data_vol, self.n_chan))
            self.data.append(curr_data) 
            
            ext = os.path.splitext(self.l_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_lbl_vol = tifffile.imread(self.l_list[index]).astype('float')
            else:
                raw_lbl_vol = cv2.imread(self.l_list[index], -1).astype('float')
            raw_labels.append(raw_lbl_vol)
            self.labels.append(transforms.reformat(raw_lbl_vol))
           
            
                  
        # cellpose source
        nmasks = np.array([raw_label.max() for raw_label in raw_labels])
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            print(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            self.data = [self.data[i] for i in ikeep]
            self.labels = [self.labels[i] for i in ikeep]
           
        print(f"Calculated/Given median diams of the train: {self.diam_train_mean}")         
        if target:
            if self.target_median_diam is None:
                diams = []
                for t_label in raw_labels:
                    diams = diams + transforms.diam_range(t_label)
                self.target_median_diam = np.percentile(np.array(diams), 75) 
            self.target_median_diam = self.target_median_diam if self.target_median_diam > 12. else 12.    
            print(f"Calculated median diams of the target: {self.target_median_diam}, \nWill get resized to equalize {self.diam_train_mean}")
            self.resize_measure = float(self.diam_train_mean/self.target_median_diam)
        
            self.target_data_samples = self.data
            self.target_label_samples = self.labels
            for _ in range(1, math.ceil(batch_size / len(self.data))):
                self.data = self.data + self.target_data_samples
                self.labels = self.labels + self.target_label_samples
                
            self.resize_array = np.full(len(self.data), 1./self.resize_measure )    
        else:           
            
            self.diam_train_median_percentile = np.array([np.percentile(np.array(transforms.diam_range(raw_labels[i])), 75) for i in trange(len(raw_labels), desc='Calculating Diam...')])
            self.diam_train_median_percentile[self.diam_train_median_percentile<12.] = 12.   
            self.resize_array = [float(curr_diam/self.diam_train_mean) if rescale else 1.0 for curr_diam in self.diam_train_median_percentile]
        
        self.msk_within = []
        self.msk_boundary = [] 
        if not self.do_every_epoch:
            data_samples = []
            label_samples = []
            for index in trange(len(self.data), desc='Preprocessing training data once only...'):
                data, labels, msk_within, msk_boundary = self.process_training_data(index)
                data_samples.append(data)
                label_samples.append(labels)
                self.msk_within.append(msk_within)
                self.msk_boundary.append(msk_boundary)
            self.data = data_samples
            self.labels = label_samples
                    
    def process_training_data(self, index):
        data, labels = self.data[index], self.labels[index]
        data, labels = transforms.random_rotate_and_resize(data, Y=labels[0], rescale=self.resize_array[index], scale_range=self.scale_range, xy=(self.crop_size,self.crop_size))
        
        # calculating mask boundary
        msk_within = labels[0].copy()
        msk_boundary = labels[0].copy()
        boundary = find_boundaries(labels[0], mode="thick").astype(np.uint16)
        indices = np.where(boundary==1)
        diff_indices = np.where(boundary!=1)
        msk_within[indices] = 0
        msk_boundary[diff_indices] = 0
        
        labels =  transforms.labels_to_flows(labels[0])
        
            
          
        return data.copy(), labels.copy(), msk_within.copy(), msk_boundary.copy()
        

    def __getitem__(self, index):
        if self.do_every_epoch:
            return self.process_training_data(index)
        else:
            return self.data[index], self.labels[index], self.msk_within[index], self.msk_boundary[index]

    def __len__(self):
        return len(self.data)


class ValCellTransposeData(Dataset):
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
    
    def __init__(self, data_dirs, n_chan, patch_size=112, median_diam=30., min_overlap=0.1):
        
        """
            Parameters
            ------------------------------------------------------------------------------------------------
            data_dirs: root directory/directories of the dataset, containing 'data' and 'labels' folders
            n_chan: Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images)
            patch_size: patch size to use, default 112
            median_diam: median diam for resizing training data, default 30
            min_overlap: minimum overlap for the patches to be processed into the network
        """
        
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
            
            lbl_diam =np.percentile(np.array(transforms.diam_range(raw_lbl_vol)), 75) #  utils.diameters(raw_lbl_vol)[0] # 
            resize_measure = median_diam/(lbl_diam if lbl_diam > 12. else 12.)
                
            curr_lbl = transforms.reformat(raw_lbl_vol)
            # lable interpolation changed to nearest neighbour from linear
            curr_lbl = transforms.resize_image(curr_lbl, rsz=resize_measure, interpolation=cv2.INTER_NEAREST)
            # curr_lbl = labels_to_flows(curr_lbl[0]) # curr_lbl[0] because it has one channel in the front idx 0
            curr_lbl, _, _, _ = transforms.padding_2D(curr_lbl, patch_size)
            LBL, _, _, _, _ = cp_transform.make_tiles(curr_lbl, bsize=patch_size, tile_overlap=min_overlap)
            ny, nx, nchan, ly, lx = LBL.shape
            LBL = np.reshape(LBL, (ny*nx, nchan, ly, lx))
            self.labels = LBL if len(self.labels) == 0 else np.concatenate((self.labels, LBL))

            
            
            ext = os.path.splitext(self.d_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
            else:
                raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')    
            curr_data = transforms.normalize1stto99th(transforms.reformat(raw_data_vol, self.n_chan))
            curr_data = transforms.resize_image(curr_data, rsz=resize_measure)
            curr_data, _, _, _ = transforms.padding_2D(curr_data, patch_size)
            IMG, _, _, _, _ = cp_transform.make_tiles(curr_data, bsize=patch_size, tile_overlap=min_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
            self.data = IMG if len(self.data) == 0 else np.concatenate((self.data, IMG))
        
        self.labels = np.array([transforms.labels_to_flows(self.labels[i][0]) for i in trange(len(self.labels), desc='Computing Val Patch Flows')])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):   
        return self.data[index], self.labels[index]
    

class EvalCellTransposeData(Dataset):
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
    def __init__(self, data_dirs, n_chan, resize_measure=1.0, median_diam=30.):
        
        """
            Parameters
            ------------------------------------------------------------------------------------------------
            data_dirs: root directory/directories of the dataset, containing 'data' and 'labels' folders
            n_chan: Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images)
            resize_measure: resize measure computed based on train median diameter and test median diameter
            median_diam: median diam for resizing training data, default 30
        """
        
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
        
        for index in trange(len(self.d_list), desc="Processing Eval Data: "):
            ext = os.path.splitext(self.d_list[index])[-1]
            if ext == '.tif' or ext == '.tiff':
                raw_data_vol = tifffile.imread(self.d_list[index]).astype('float')
            else:
                raw_data_vol = cv2.imread(self.d_list[index], -1).astype('float')
            
            
            
            self.data.append(transforms.normalize1stto99th(transforms.reformat(raw_data_vol, self.n_chan))) 
            
            if os.path.exists(os.path.join(dir_i,'labels')):
                
                ext = os.path.splitext(self.l_list[index])[-1]
                if ext == '.tif' or ext == '.tiff':
                    raw_lbl_vol = tifffile.imread(self.l_list[index]).astype('float')
                else:
                    raw_lbl_vol = cv2.imread(self.l_list[index], -1).astype('float')
                self.labels.append(raw_lbl_vol) 
                lbl_diam = np.percentile(np.array(transforms.diam_range(raw_lbl_vol)), 75) #  utils.diameters(raw_lbl_vol)[0] # 
                self.resize_measure_range.append(float(median_diam/(lbl_diam if lbl_diam > 12. else 12.)))
            else:
                self.resize_measure_range.append(resize_measure)
            
        if len(self.labels) != 0: 
            print("Labels exist, they will be used to resize the image for evaluation...")  
            
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):   
        return self.data[index], self.d_list[index], self.resize_measure_range[index]

# Updated more efficient version    
class EvalCellTransposeData3D(Dataset):
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
    def __init__(self, data_dirs, n_chan, resize_measure=1.0):
        
        """
            Parameters
            ------------------------------------------------------------------------------------------------
            data_dirs: root directory/directories of the dataset, containing 'data' and 'labels' folders
            n_chan: Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images)
            resize_measure: resize measure computed based on train median diameter and test median diameter
        """
        
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
                
        data = transforms.normalize1stto99th(transforms.reformat(raw_data_vol, self.n_chan, do_3D=True))
        return data, self.d_list[index], self.resize_measure

class find_shots:
    
    def __init__(self, data_dirs, save_dir, shots=3, patch_size=112, nominal_cell_metric=30, scaling_factor=1.25, min_cells=1):
        self.shots = shots
        self.patch_size = patch_size
        self.nominal_cell_metric = nominal_cell_metric
        self.scaling_factor = scaling_factor
        self.save_dir = save_dir   
        self.min_cells = min_cells
        
        self.d_list = []
        self.l_list = []
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
                
    
    def select_sample_window(self, d, lbl, smpl_cntr):
        
        exemplar_cell = lbl[smpl_cntr]
        cell_metric = transforms.cell_range(lbl, exemplar_cell)
        patch_metric = ceil((cell_metric / self.nominal_cell_metric) * self.patch_size * self.scaling_factor / 2)
        
        crp_lbl = copy.deepcopy(lbl[max(smpl_cntr[0]-patch_metric, 0): min(smpl_cntr[0]+patch_metric, lbl.shape[0]),
                                max(smpl_cntr[1]-patch_metric, 0): min(smpl_cntr[1]+patch_metric, lbl.shape[1])])
        
        # CH x h x w
        if len(d.shape)>2:
            crp_dta = copy.deepcopy(d[:,max(smpl_cntr[0]-patch_metric, 0): min(smpl_cntr[0]+patch_metric, d.shape[1]),
                                max(smpl_cntr[1]-patch_metric, 0): min(smpl_cntr[1]+patch_metric, d.shape[2])])
        else:
            crp_dta = copy.deepcopy(d[max(smpl_cntr[0]-patch_metric, 0): min(smpl_cntr[0]+patch_metric, d.shape[0]),
                                max(smpl_cntr[1]-patch_metric, 0): min(smpl_cntr[1]+patch_metric, d.shape[1])])
        
        return crp_dta, crp_lbl

    
    def random_shots(self):
        """
        Generates square shots from the randomly sampled images based on Cellpose 2.0 guidelines
        also generates flows
        
        """
        save_path = os.path.join(self.save_dir,str(self.shots)+'-shot')
        os.makedirs(os.path.join(save_path, 'data'))
        os.makedirs(os.path.join(save_path, 'labels'))
            
        d_l_list = list(zip(self.d_list, self.l_list))
        np.random.shuffle(d_l_list)
        
        data_shots = []
        labels_shots = []
        total_masks=0
        
        curr_shot=0
        while len(data_shots)<self.shots:
            
            d_name = d_l_list[curr_shot][0]
            l_name = d_l_list[curr_shot][1]
            
            d_ext = os.path.splitext(d_name)[1]
            l_ext = os.path.splitext(l_name)[1]
            data = tifffile.imread(d_name) if d_ext == '.tif' or '.tiff' else cv2.imread(d_name, -1)
            label = tifffile.imread(l_name) if l_ext == '.tif' or '.tiff' else cv2.imread(l_name, -1)
            
            mask_IDs = np.unique(label)[1:]
            running_masks=0
            finalized_crop_data=np.array([])
            finalized_crop_label=np.array([])
            
            for ID in tqdm(mask_IDs, desc='Investigating Masks: '):
                sample_center = np.mean(np.argwhere(label==ID), axis=0).astype('int')
                sample_center = (sample_center[0],sample_center[1])
                crop_data, crop_label = self.select_sample_window(data, label, sample_center)
                current_masks =np.unique(crop_label)[1:]
                if len(current_masks) > running_masks: 
                    finalized_crop_data=crop_data
                    finalized_crop_label=crop_label
                    running_masks=len(current_masks)
                    

            if running_masks >= self.min_cells:
                print(f'Shape {finalized_crop_label.shape[-1]} x {finalized_crop_label.shape[-2]}')
                data_shots.append(finalized_crop_data)
                labels_shots.append(finalized_crop_label)
                total_masks += running_masks
                
                
                if d_ext == '.tif' or '.tiff': tifffile.imwrite(os.path.join(save_path, 'data', 'Crop_'+os.path.basename(d_name)),finalized_crop_data)
                else: cv2.imwrite(os.path.join(save_path, 'data', 'Crop_'+os.path.basename(d_name)), finalized_crop_data)
                if l_ext == '.tif' or '.tiff': tifffile.imwrite(os.path.join(save_path, 'labels', 'Crop_'+os.path.basename(l_name)), finalized_crop_label)
                else: cv2.imwrite(os.path.join(save_path, 'labels', 'Crop_'+os.path.basename(l_name)), finalized_crop_label)
                    
            else:
                print(f'Masks collected for this shot are less than the user defined value - {self.min_cells}, therefore, moving to the next random image!')    
            curr_shot += 1
        print(f'Total masks collected: {total_masks}')
        
        return save_path