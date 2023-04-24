import tifffile as tiff
import numpy as np
import argparse
import os
from scipy.ndimage.measurements import label


def remove_small_mask(label, min_mask_size=15):
    mask_unique, mask_counts = np.unique(label, return_counts=True)
    mask_unique = mask_unique[1:]
    mask_counts = mask_counts[1:]
    indices = []
    for idx,i in enumerate(mask_counts):
        if i < min_mask_size:
            indices.append(idx)
    for i in indices:
        label[np.where(label==mask_unique[i])] = 0
        
    return label


def clear_fragments(input_path, output_path):
    curr_total_masks=0
    refined_total_masks=0
    assert not os.path.exists(output_path),\
        f'Results folder {format(output_path)} currently exists; please specify new location to save segmentations.'
    
    if os.path.exists(output_path) == False: os.makedirs(output_path)
    
    total_duplicate_masks = 0
    for file in os.listdir(input_path):
        img_file = os.path.join(input_path, file)
        
        im = tiff.imread(img_file)
        new_im = np.zeros_like(im)
        unique_masks = np.unique(im)[1:]

        for curr_mask in unique_masks:
            new_im[np.where(im==curr_mask)]=0
            new_array = np.zeros_like(im)
            new_array[np.where(im==curr_mask)]=1
            
            new_array = np.ceil(new_array).astype(np.int16)
            seg, n_comp = label(new_array)
            seg_unique, seg_counts = np.unique(seg, return_counts=True)
            if len(seg_unique[1:]) > 1: 
                total_duplicate_masks+=1
                seg_max_ind = np.argmax(seg_counts[1:])
                seg[np.where(seg!=seg_unique[1:][seg_max_ind])] = 0
                new_im[np.where(seg!=0)]=curr_mask
            else:
                new_im[np.where(im==curr_mask)]=curr_mask
            
        tiff.imwrite(os.path.join(output_path, file), new_im)
        
        curr_total_masks+=len(unique_masks)
        refined_total_masks+=len(np.unique(new_im)[1:])

        print(f"{img_file}: ")
        print(f"Mask before refinement: {len(unique_masks)}")
        print(f"Mask after refinement: {len(np.unique(new_im)[1:])}")
    
    print(f"Total duplicate masks: {total_duplicate_masks}")  
    print(f"Total mask before refinement: {curr_total_masks}")
    print(f"Total mask after refinement: {refined_total_masks}")
        

if __name__ == "__main__":
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='The directory containing the segementations.')
    parser.add_argument('--output-dir', help='The directory where the new fragments free segmentations will go.')
    args = parser.parse_args()
        
    input_path  = args.input_dir # '/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/split_data/A172/test/labels'
    output_path = args.output_dir # '/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/split_data/A172/test_cleaned/labels'
    clear_fragments(input_path, output_path)