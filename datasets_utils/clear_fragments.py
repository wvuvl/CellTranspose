import tifffile as tiff
import numpy as np
import copy
import argparse
import os
from scipy.ndimage.measurements import label
from tqdm import tqdm


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

    for file in os.listdir(input_path):
        img_file = os.path.join(input_path, file)
        
        im = tiff.imread(img_file)
        new_im = copy.copy(im)
        unique_masks = np.unique(im)[1:]
        add_mask=len(unique_masks)

        for curr_mask in unique_masks:
            new_array = np.zeros((im.shape))
            new_array[np.where(im==curr_mask)] = 1
            
            new_array = np.ceil(new_array).astype(np.int16)
            seg, n_comp = label(new_array)
            
            for comp in np.unique(seg)[1:]:
                if comp==1: new_im[np.where(seg==1)] = curr_mask
                else: 
                    add_mask+=1
                    new_im[np.where(seg==comp)] = add_mask
        
        new_im = remove_small_mask(new_im)
        tiff.imwrite(os.path.join(output_path, file), new_im)
        
        curr_total_masks+=len(unique_masks)
        refined_total_masks+=len(np.unique(new_im)[1:])

        print(f"{img_file}: ")
        print(f"Mask before refinement: {len(unique_masks)}")
        print(f"Mask after refinement: {len(np.unique(new_im)[1:])}")
        
    print(f"Total mask before refinement: {curr_total_masks}")
    print(f"Total mask after refinement: {refined_total_masks}")
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='The directory containing the segementations.')
    parser.add_argument('--output-dir', help='The directory where the new fragments free segmentations will go.')
    args = parser.parse_args()
    
    input_path  = args.input_dir # '/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/LiveCell/images/test/labels'
    output_path = args.output_dir # '/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/LiveCell/images/test_cleaned/labels'
    clear_fragments(input_path, output_path)
    