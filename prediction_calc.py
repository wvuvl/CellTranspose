
import os
import pickle
import numpy as np
from transforms import Resize, reformat,followflows3D,followflows2D

import tifffile

with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results_v2/results_xy', 'BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0005_raw_masks_flows.pkl'), 'rb') as rmf_pkl:
    pred_xy = np.array(pickle.load(rmf_pkl))
with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results_v2/results_zy', 'BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0005_raw_masks_flows.pkl'), 'rb') as rmf_pkl:
    pred_zy = np.array(pickle.load(rmf_pkl))
with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results_v2/results_zx', 'BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0005_raw_masks_flows.pkl'), 'rb') as rmf_pkl:
    pred_zx = np.array(pickle.load(rmf_pkl))


print(pred_xy.shape)
print(pred_zy.shape)
print(pred_zx.shape)

pred = pred_xy
pred_xy = pred_xy.transpose(1,0,2,3)
pred_zy_xy = pred_zy.transpose(1,2,3,0) #swapaxes(0,2)
pred_zx_xy = pred_zx.transpose(1,2,0,3) #swapaxes(0,1)

print(pred_xy.shape)
print(pred_zy_xy.shape)
print(pred_zx_xy.shape)

#(stacks, classes, imgs.shape[0], imgs.shape[1], imgs.shape[2])
yf = np.zeros((3, 3, 129, 565, 807), np.float32)

yf[0] = pred_xy
yf[1] = pred_zy_xy
yf[2] = pred_zx_xy

#cellprob = pred_xy[0] + pred_yz_xy[0] + pred_xz_xy[0]
        
cellprob = (yf[0][0] + yf[1][0] + yf[2][0])/3

#['YX', 'ZY', 'ZX']

#dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]), axis=0) # (dZ, dY, dX)

#sets perfect dims, but still getting wrong number of masks
dP = np.stack((yf[1][1] + yf[2][1], yf[0][1] + yf[1][2], yf[0][2] + yf[2][2]), axis=0) # (dZ, dY, dX)

#dP = np.stack((yf[1][1] + yf[2][1], yf[0][2] + yf[2][1], yf[0][2] + yf[1][2]),
#                          axis=0) # (dZ, dY, dX)
print("dP shape:", dP.shape)
print("cellprob: ",cellprob.shape)

cp = cellprob > 0.0 
print(np.unique(cp,return_counts=True))

mask = np.array(followflows3D(dP,cellprob))
print(np.unique(mask,return_counts=True))
#tifffile.imwrite(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results','3D_mask' + '.tif'), mask)


"""
with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/cellpose_results','dP' + '.tif'), 'rb') as rmf_pkl:
    cellpose_dP = np.array(pickle.load(rmf_pkl))
with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/cellpose_results','cellprob' + '.tif'), 'rb') as rmf_pkl:
    cellpose_cellprob = np.array(pickle.load(rmf_pkl))


print("Cellpose dP shape:", cellpose_dP.shape)
print("Cellpose cellprob: ",cellpose_cellprob.shape)


cp_mask = cellprob > 0.0
cp_mask_cellpose = cellpose_cellprob > 0.0

print("cp_mask: ",cp_mask.shape)
print("cp_mask_cellpose: ", cp_mask_cellpose.shape)

print(cp_mask)
print(cp_mask_cellpose)


print(np.unique(cp_mask,return_counts=True))
print(np.unique(cp_mask_cellpose,return_counts=True))


#dP = tifffile.imread(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results','dP' + '.tif'))
#cellprob = tifffile.imread(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results','cellprob' + '.tif'))





#tifffile.imwrite(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results','3D_mask' + '.tif'), mask)


mask_org = tifffile.imread('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/labels/BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0005.tif')
print(np.unique(mask_org))"""

"""
masks_xy = []
masks_yz = []
masks_xz = []

for i in tqdm(pred,desc="processing xy: "): masks_xy.append(np.array(followflows2D(tensor(i))))
for i in tqdm(pred_yz,desc="processing yz: "): masks_yz.append(np.array(followflows2D(tensor(i))))
for i in tqdm(pred_xz,desc="processing xz: "): masks_xz.append(np.array(followflows2D(tensor(i))))

print("xy: ", np.unique(np.array(masks_xy)))
print("yz: ", np.unique(np.array(masks_yz)))
print("xz: ", np.unique(np.array(masks_xz)))

import numpy as np
import tifffile
import os
       
mask_celltranspose = tifffile.imread(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results','3D_mask' + '.tif'))
mask_cellpose = tifffile.imread(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/cellpose_results','3D_mask' + '.tif'))


print(np.unique(mask_celltranspose))
print(np.unique(mask_cellpose))

import math

print((4/3)(math.pi**(1/2))*(15**(3/2)))"""