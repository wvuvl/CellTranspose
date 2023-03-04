"""Used to calculate AP from masks obtained through original Cellpose model"""

import os
import numpy as np
import tifffile
import cv2
from cellpose_src.metrics import average_precision,_intersection_over_union, _label_overlap
import pickle
import matplotlib.pyplot as plt
import argparse
from skimage import measure

mask_path = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/full_data_lbl_0/test/mesmer_results'
label_path = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/full_data_lbl_0/test/labels'


tif_or_png_pred = os.path.splitext(os.listdir(mask_path)[0])[-1]
# print(tif_or_png_pred)
tif_or_png_label = os.path.splitext(os.listdir(label_path)[0])[-1]
# print(tif_or_png_label)
masks = []
labels = []
filenames = []

for file in sorted(os.listdir(mask_path)):
    if tif_or_png_pred == '.png':
        masks.append(cv2.imread(os.path.join(mask_path, file), -1).astype('int32'))
    elif tif_or_png_pred == '.tif':
        masks.append(tifffile.imread(os.path.join(mask_path, file)).astype('int32'))
    else:
        raise Exception('\"tif_or_png\" must be set as \".tif\" or \".png\"')

for file in sorted(os.listdir(label_path)):
    if tif_or_png_label == '.png':
        labels.append(cv2.imread(os.path.join(label_path, file), -1).astype('int32'))
        filenames.append(file)
    elif tif_or_png_label == '.tif':
        labels.append(tifffile.imread(os.path.join(label_path, file)).astype('int32'))
        filenames.append(file)
    else:
        raise Exception('\"tif_or_png\" must be set as \".tif\" or \".png\"')

print(sorted(os.listdir(mask_path)))
print(sorted(os.listdir(label_path)))


# Calculate AP
tau = np.arange(0.0, 1.01, 0.01)
ap_info = average_precision(labels, masks, threshold=tau)
ap_overall = np.average(ap_info[0], axis=0)
tp_overall = np.sum(ap_info[1], axis=0).astype('int32')
fp_overall = np.sum(ap_info[2], axis=0).astype('int32')
fn_overall = np.sum(ap_info[3], axis=0).astype('int32')
false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
    
ap = [[]] * len(masks) # average precision matrix per image
tp = [[]] * len(masks)
fp = [[]] * len(masks)
fn = [[]] * len(masks)
IoU = [[]] * len(masks) 

api,tpi,fpi,fni = average_precision(labels,masks,threshold=tau)

for k in range(len(masks)):
    # get the IoU matrix; axis 0 corresponds to GT, axis 1 to pred 

    regions = measure.regionprops(masks[k])
    pred_areas = np.array([reg.area for reg in regions])
    IoU[k] = _intersection_over_union(labels[k], masks[k])

 

np.mean(api[0],axis=0).T
print('AP Results at IoU threshold 0.5: AP = {}\nTrue Postive: {}; False Positive: {}; '
        'False Negative: {}'.format(ap_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
print('Total false error rate: {:.6f}'.format(false_error))