import math
import torch
from torch import unsqueeze
import torch.nn.functional as F
import random
import skimage.io as io
import tifffile as tiff
import matplotlib.pyplot as plt


img_path = r"/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/previous_datasets/practice/val/data/BBBC024_v1_c00_highSNR_images_TIFF-image-final_0000.tif"
lbl_path = r"/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/previous_datasets/practice/val/labels/BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0000.tif"

data = torch.from_numpy(tiff.imread(img_path))
label = torch.from_numpy(tiff.imread(lbl_path))

print(data.shape)
print(label.shape)

if data.shape[1] < 1000: 
    pad_x = math.ceil((1000-data.shape[1])/2)
    data = F.pad(data,(pad_x,pad_x))
    label = F.pad(label,(pad_x,pad_x))
    
if data.shape[0] < 1000:
    pad_y = math.ceil((1000-data.shape[0])/2)
    data = F.pad(data,(0,0,pad_y,pad_y))
    label = F.pad(label,(0,0,pad_y,pad_y))

print(data.shape)
print(label.shape)

plt.figure()
plt.imshow(data.numpy())
plt.show()

plt.figure()
plt.imshow(label.numpy())
plt.show()