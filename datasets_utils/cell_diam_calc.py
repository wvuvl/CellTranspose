import numpy as np
import tifffile as tiff
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def diameters(masks):
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5) / 2
    return md, counts**0.5


def calc_median_dim(dir, plot=False):
    median_list = []
    count_list = []

    for img_name in tqdm(os.listdir(dir)):
        ext = os.path.splitext(img_name)[-1]
        if ext == '.tif' or ext == '.tiff':
            mask = tiff.imread(os.path.join(dir, img_name)).astype('float')
        else:
            mask = cv2.imread(os.path.join(dir, img_name), -1).astype('float')
                    
        md, counts = diameters(mask)
        median_list = np.append(median_list, md)
        count_list = np.append(count_list, counts)
        
    print( f'median of median: {math.ceil(np.median(median_list))} \n median of total: {math.ceil(np.median(count_list))}\n')
    
    if plot:
        plt.hist(median_list)
        plt.xlabel(f'Diameter')
        plt.ylabel('Number of Cells')
        plt.show()
    

    

# BBBC006
# nums = ['00', '08', '16' , '24', '32']
# for num in nums:
#     dir = f"/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/BBBC006/BBBC006/z_{num}/test/labels"
#     calc_median_dim(dir)
    
# # TissueNet 1.0 
# platforms = ["imc", "codex", "cycif", "mibi", "vectra", "mxif"]
# tissues = ["breast", "gi", "immune", "lung", "pancreas", "skin"]

# # Platform Specific
# for platform in platforms:
#     for tissue in tissues:
#         dir = f'/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/test/specialist_updated_mask/platform_specific/{platform}/{tissue}/labels'
        
#         if os.path.exists(dir):
#             print(f"\n{platform}-{tissue}: ")
#             calc_median_dim(dir)
# # tissue specific
# for tissue in tissues:
#     for platform in platforms:
#         dir = f'/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/test/specialist_updated_mask/tissue_specific/{tissue}/{platform}/labels'
        
#         if os.path.exists(dir): 
#             print(f"\n{tissue}-{platform}: ")
#             calc_median_dim(dir)

dir = '/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test/labels'
calc_median_dim(dir)
# for cell_type in ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]:
#     dir = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/split_data_refined/{cell_type}/train/labels'
#     print(f'{cell_type}: ')
#     calc_median_dim(dir)