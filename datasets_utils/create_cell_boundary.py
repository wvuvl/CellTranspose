import numpy as np
import tifffile as tiff
from skimage import io, filters, color
from skimage.segmentation import find_boundaries, mark_boundaries
import matplotlib.pyplot as plt
import cv2

label = tiff.imread('/mnt/12F9CADD61CB0337/results/cell_analysis/new_cellTranspose/cellpose_2.0_style_model/z_16/10-shot/01-generalized_to_BBBC006_z_16-10-shot/tiff_results/mcf-z-stacks-03212011_a01_s1.tif')
data = tiff.imread('/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/BBBC006/BBBC006/z_16/test/data/mcf-z-stacks-03212011_a01_s1.tif')



msk_boundary = label.copy()
boundary = find_boundaries(label, mode="thick").astype(np.uint16)
indices = np.where(boundary==1)
diff_indices = np.where(boundary!=1)
msk_boundary[diff_indices] = 0


img = np.vstack((data, np.zeros_like(data)[:1]))
# data[0][np.where(msk_boundary!=0)] = (100,100,500)


plt.imsave('temp/temp.png', im.transpose(1,2,0))
print()
