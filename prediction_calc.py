import argparse
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torch import nn, device, load, save, squeeze, as_tensor, tensor
from torch.cuda import is_available, device_count, empty_cache
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import time
import gc

from transforms import Resize, reformat,followflows3D,followflows2D
from loaddata import TrainCellTransposeData, ValTestCellTransposeData, ValTestCellTransposeData3D, path_iterator
from CellTranspose2D import CellTranspose, SizeModel, ClassLoss, FlowLoss, SASClassLoss, ContrastiveFlowLoss
from train_eval import train_network, adapt_network, eval_network, eval_network_3D, create_3D_masks, run_3D_masks
from cellpose_src.metrics import average_precision
from misc_utils import produce_logfile
from tqdm import tqdm

with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results_v1/results_xy', 'BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0005_raw_masks_flows.pkl'), 'rb') as rmf_pkl:
    pred_xy = np.array(pickle.load(rmf_pkl))
with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results_v1/results_yz', 'BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0005_raw_masks_flows.pkl'), 'rb') as rmf_pkl:
    pred_yz = np.array(pickle.load(rmf_pkl))
with open(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results_v1/results_xz', 'BBBC024_v1_c00_highSNR_images_TIFF-image-labels_0005_raw_masks_flows.pkl'), 'rb') as rmf_pkl:
    pred_xz = np.array(pickle.load(rmf_pkl))


print(pred_xy.shape)
print(pred_yz.shape)
print(pred_xz.shape)

pred_xy = pred_xy.transpose(1,0,2,3)
pred_yz_xy = pred_yz.transpose(1,3,2,0) #swapaxes(0,2)
pred_xz_xy = pred_xz.transpose(1,2,0,3) #swapaxes(0,1)

print(pred_xy.shape)
print(pred_yz_xy.shape)
print(pred_xz_xy.shape)


yf = np.zeros((3, 3, 129, 565, 807), np.float32)

yf[0] = pred_xy
yf[1] = pred_yz_xy
yf[2] = pred_xz_xy


        
cellprob =yf[0][-1] + yf[1][-1] + yf[2][-1]
dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]),
                          axis=0) # (dZ, dY, dX)


print("cellprob: ",cellprob.shape)
print("dP shape:", dP.shape)

print(dP.dtype)
print(dP.size)

print(cellprob.dtype)

print(cellprob.size)

print(dP)

tifffile.imwrite(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results','dP' + '.tif'),
                            dP)

"""masks = np.array(followflows3D(dP,cellprob))

print("masks: ", masks.shape)

tifffile.imwrite(os.path.join('/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell analysis/datasets/datasets/BBBC024_3D_test/results','3D_mask' + '.tif'),
                            masks)"""
                            
        