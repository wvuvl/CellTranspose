import os
from cv2 import data

import numpy as np
import skimage.io as io

from deepcell.utils.plot_utils import create_rgb_image
from deepcell.utils.plot_utils import make_outline_overlay

data_dir = r"/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/TissueNet_dataset_from_Greenwald_Miller/tissuenet_v1.0/tissuenet_1.0/tissuenet_v1.0_test/X.npy"
label_dir = r"/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/TissueNet_dataset_from_Greenwald_Miller/tissuenet_v1.0/tissuenet_1.0/tissuenet_v1.0_test/y.npy"
tissue = r"/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/TissueNet_dataset_from_Greenwald_Miller/tissuenet_v1.0/tissuenet_1.0/tissuenet_v1.0_test/tissue_list.npy"
platform= r"/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/TissueNet_dataset_from_Greenwald_Miller/tissuenet_v1.0/tissuenet_1.0/tissuenet_v1.0_test/platform_list.npy"


test_X = np.load(data_dir)
test_y = np.load(label_dir)
tissue_list = np.load(tissue)
platform_list = np.load(platform)

valid_tissues = np.unique(tissue_list)
print(valid_tissues)
valid_platforms = np.unique(platform_list)
print(valid_platforms)

selected_tissue = 'breast'
selected_platform = 'all'


if selected_tissue not in valid_tissues and selected_tissue != 'all':
    raise ValueError('Selected tissue must be either be part of the valid_tissues list, or all')

if selected_platform not in valid_platforms and selected_platform != 'all':
    raise ValueError('Selected platform must be either be part of the valid_platforms list, or all')

if selected_tissue == 'all':
    tissue_idx = np.repeat(True, len(tissue_list))
else:
    tissue_idx = tissue_list == selected_tissue

if selected_platform == 'all':
    platform_idx = np.repeat(True, len(platform_list))
else:
    platform_idx = platform_list == selected_platform

combined_idx = tissue_idx * platform_idx

if sum(combined_idx) == 0:
    raise ValueError("The specified combination of image platform and tissue type does not exist")

selected_X, selected_y = test_X[combined_idx, ...], test_y[combined_idx, ...]

rgb_images = create_rgb_image(selected_X, channel_colors=['green', 'blue'])
overlay_data_cell = make_outline_overlay(rgb_data=rgb_images, predictions=selected_y[..., 0:1])
overlay_data_nuc = make_outline_overlay(rgb_data=rgb_images, predictions=selected_y[..., 1:2])

plot_idx = np.random.randint(0, selected_X.shape[0])
io.imshow(overlay_data_cell[plot_idx])

io.imshow(overlay_data_nuc[plot_idx])