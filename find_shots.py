import numpy as np 
import os
import tifffile as tiff
import cv2
from math import ceil
from sys import maxsize
import copy
import numpy as np
from tqdm import tqdm
from transforms import labels_to_flows, cell_range, remove_small_mask  # Assumes domain-adaptive_cellular_instance-seg is added to interpreter path

def select_sample_window(d, lbl, smpl_cntr, nominal_cell_metric, patch_size, scaling_factor, compute_flows=False):
    exemplar_cell = lbl[smpl_cntr]
    """if from_3d:
        raise Exception('Add 3D cell range calculation here')
    else:
        cell_metric = cell_range(lbl, exemplar_cell)"""
        
    cell_metric = cell_range(lbl, exemplar_cell)
    patch_metric = ceil((cell_metric / nominal_cell_metric) * patch_size * scaling_factor / 2)
    
    crp_lbl = copy.deepcopy(lbl[max(smpl_cntr[0]-patch_metric, 0): min(smpl_cntr[0]+patch_metric, lbl.shape[0]),
                            max(smpl_cntr[1]-patch_metric, 0): min(smpl_cntr[1]+patch_metric, lbl.shape[1])])
    
    # CH x h x w
    if len(d.shape)>2:
        crp_dta = copy.deepcopy(d[:,max(smpl_cntr[0]-patch_metric, 0): min(smpl_cntr[0]+patch_metric, d.shape[1]),
                            max(smpl_cntr[1]-patch_metric, 0): min(smpl_cntr[1]+patch_metric, d.shape[2])])
    else:
        crp_dta = copy.deepcopy(d[max(smpl_cntr[0]-patch_metric, 0): min(smpl_cntr[0]+patch_metric, d.shape[0]),
                            max(smpl_cntr[1]-patch_metric, 0): min(smpl_cntr[1]+patch_metric, d.shape[1])])
    if compute_flows:
        crp_flow_lbl = labels_to_flows(crp_lbl)
    
    return crp_dta, crp_lbl

"""
Generates square shots from the randomly sampled images based on Cellpose 2.0 guidelines
also generates flows
TODO: 3D functionality
TODO: compute flow not functional
"""
def random_shots(d_list, l_list, shots=3, patch_size=112, nominal_cell_metric=30, \
                 scaling_factor=1.25, save_dir=None, min_cells=5, from_3D=False, compute_flows=False):
    
    if save_dir is not None:
        os.makedirs(os.path.join(save_dir,str(shots)+'-shot', 'data'))
        os.makedirs(os.path.join(save_dir,str(shots)+'-shot', 'labels'))
        # if compute_flows:
        #     os.makedirs(os.path.join(save_dir,shots+'-shot_flows', 'data'))
        #     os.makedirs(os.path.join(save_dir,shots+'-shot_flows', 'labels'))
            
    # shuffling the data in the same order
    d_l_list = list(zip(d_list, l_list))
    np.random.shuffle(d_l_list)
    
    data_shots = []
    labels_shots = []
    total_masks=0
    
    curr_shot=0
    while len(data_shots)<shots:
        
        d_name = d_l_list[curr_shot][0]
        l_name = d_l_list[curr_shot][1]
        
        d_ext = os.path.splitext(d_name)[1]
        l_ext = os.path.splitext(l_name)[1]
        data = tiff.imread(d_name) if d_ext == '.tif' or '.tiff' else cv2.imread(d_name, -1)
        label = tiff.imread(l_name) if l_ext == '.tif' or '.tiff' else cv2.imread(l_name, -1)
        
        mask_IDs = np.unique(label)[1:]
        running_masks=0
        finalized_crop_data=np.array([])
        finalized_crop_label=np.array([])
        
        for ID in tqdm(mask_IDs, desc='Investigating Masks: '):
            sample_center = np.mean(np.argwhere(label==ID), axis=0).astype('int')
            sample_center = (sample_center[0],sample_center[1])
            crop_data, crop_label = select_sample_window(data, label, sample_center, nominal_cell_metric, patch_size, scaling_factor)
            current_masks =np.unique(crop_label)[1:]
            if len(current_masks) > running_masks: 
                running_masks=len(current_masks)
                finalized_crop_data=crop_data
                finalized_crop_label=crop_label
                
        # finalized_crop_label = remove_cut_cells(finalized_crop_label)
        
        # Padding if size is not square
        if finalized_crop_label.shape[-1] != finalized_crop_label.shape[-2]:
            print('Malformed shape ({} x {}); skipping...\nAdding padding'.format(finalized_crop_label.shape[-1], finalized_crop_label.shape[-2]))
                       
            # finalized_crop_label = remove_cut_cells(finalized_crop_label) # finalized_crop_label = remove_cut_cells(finalized_crop_label, flows=compute_flows) 
            
            dim = max(finalized_crop_label.shape)
            
            new_crop_lable = np.zeros((dim,dim), dtype=finalized_crop_label.dtype)
            set_corner = (max(0, (new_crop_lable.shape[0]-finalized_crop_label.shape[0])//2),
                              max(0, (new_crop_lable.shape[1]-finalized_crop_label.shape[1]) // 2))
            
            new_crop_lable[set_corner[0]:set_corner[0]+finalized_crop_label.shape[0], 
                           set_corner[1]:set_corner[1]+finalized_crop_label.shape[1]] = finalized_crop_label
            
            if len(finalized_crop_data.shape) > 2:
                new_crop_data = np.zeros((finalized_crop_data.shape[0],dim,dim), dtype=finalized_crop_data.dtype)
                new_crop_data[:,set_corner[0]:set_corner[0]+finalized_crop_data.shape[1], 
                           set_corner[1]:set_corner[1]+finalized_crop_data.shape[2]] = finalized_crop_data
            else:
                new_crop_data = np.zeros((dim,dim), dtype=finalized_crop_data.dtype)
                new_crop_data[set_corner[0]:set_corner[0]+finalized_crop_data.shape[0], 
                           set_corner[1]:set_corner[1]+finalized_crop_data.shape[1]] = finalized_crop_data
            
            finalized_crop_label = remove_small_mask(new_crop_lable)
            finalized_crop_data = new_crop_data

        unique_finalized_masks =  len(np.unique(finalized_crop_label)[1:])
        if unique_finalized_masks >= min_cells:
            curr_shot += 1
            data_shots.append(finalized_crop_data)
            labels_shots.append(finalized_crop_label)
            total_masks += unique_finalized_masks
            
            if save_dir is not None:
                if d_ext == '.tif' or '.tiff': tiff.imwrite(os.path.join(save_dir,str(shots)+'-shot', 'data', 'Crop_'+os.path.basename(d_name)),finalized_crop_data)
                else: cv2.imwrite(os.path.join(save_dir,str(shots)+'-shot', 'data', 'Crop_'+os.path.basename(d_name)), finalized_crop_data)
                if l_ext == '.tif' or '.tiff': tiff.imwrite(os.path.join(save_dir,str(shots)+'-shot', 'labels', 'Crop_'+os.path.basename(l_name)), finalized_crop_label)
                else: cv2.imwrite(os.path.join(save_dir,str(shots)+'-shot', 'labels', 'Crop_'+os.path.basename(l_name)), finalized_crop_label)
                
                # if compute_flows:
                #     if d_ext == '.tif' or '.tiff': tiff.imwrite(os.path.join(save_dir,shots+'-shot_flows', 'data', 'Crop_'+os.path.basename(d_name)), finalized_crop_data)
                #     else: cv2.imwrite(os.path.join(save_dir,shots+'-shot_flows', 'data', 'Crop_'+os.path.basename(d_name)), finalized_crop_data)
                #     if l_ext == '.tif' or '.tiff': tiff.imwrite(os.path.join(save_dir,shots+'-shot_flows', 'labels', 'Crop_'+os.path.basename(l_name)), finalized_crop_label)
                #     else: cv2.imwrite(os.path.join(save_dir,shots+'-shot_flows', 'labels', 'Crop_'+os.path.basename(l_name)), finalized_crop_label)
        else:
            print(f'Masks collected for this shot are less than the user defined value - {min_cells}, therefore, moving to the next random image!')    
    print(f'Total masks collected: {total_masks}')
    
    return data_shots, labels_shots

if __name__ == "__main__":
    data_dirs = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/BBBC006/BBBC006/z_00/test'
    
    d_list = sorted([os.path.join(data_dirs,'data',file) for file in os.listdir(os.path.join(data_dirs, 'data'))])
    l_list = sorted([os.path.join(data_dirs,'labels',file) for file in os.listdir(os.path.join(data_dirs, 'labels'))])
    
    random_shots(d_list,l_list)