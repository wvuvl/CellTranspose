import tifffile as tiff
import numpy as np
from scipy.ndimage.measurements import label
from transforms import remove_small_mask

im = tiff.imread('/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/LiveCell/images/inquiry/SHSY5Y_Phase_A10_1_00d08h00m_2_masks.tif')
new_im = im
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
tiff.imwrite('/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/LiveCell/images/inquiry/SHSY5Y_Phase_A10_1_00d08h00m_2_masks_updated.tif', new_im)
