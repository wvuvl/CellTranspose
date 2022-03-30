"""Used to calculate AP from masks obtained through original Cellpose model"""

import os
import numpy as np
import tifffile
import cv2
from cellpose_src.metrics import average_precision
import pickle
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mask-path')
parser.add_argument('--label-path')
args = parser.parse_args()

tif_or_png_pred = os.path.splitext(os.listdir(args.mask_path)[0])[-1]
print(tif_or_png_pred)
tif_or_png_label = os.path.splitext(os.listdir(args.label_path)[0])[-1]
print(tif_or_png_label)
masks = []
labels = []
filenames = []

# for file in sorted(i for i in os.listdir(args.mask_path) if i.endswith('.npy')):
#     masks.append(np.load(os.path.join(args.mask_path, file), allow_pickle=True).item()['masks'])

for file in sorted(os.listdir(args.mask_path)):
    if tif_or_png_pred == '.png':
        masks.append(cv2.imread(os.path.join(args.mask_path, file), -1))
    elif tif_or_png_pred == '.tif':
        masks.append(tifffile.imread(os.path.join(args.mask_path, file)))
    else:
        raise Exception('\"tif_or_png\" must be set as \".tif\" or \".png\"')

for file in sorted(os.listdir(args.label_path)):
    if tif_or_png_label == '.png':
        labels.append(cv2.imread(os.path.join(args.label_path, file), -1))
        filenames.append(file)
    elif tif_or_png_label == '.tif':
        labels.append(tifffile.imread(os.path.join(args.label_path, file)))
        filenames.append(file)
    else:
        raise Exception('\"tif_or_png\" must be set as \".tif\" or \".png\"')

# Count cells in each mask and calculate counting error
with open(os.path.join(args.mask_path, 'counted_cells.txt'), 'w') as cc:
    predicted_count = 0
    true_count = 0
    for i in range(len(masks)):
        num_masks = len(np.unique(masks[i]))-1
        num_predicted = len(np.unique(labels[i]))-1
        cc.write('{}:\nPredicted: {}; True: {}\n'.format(filenames[i], num_masks, num_predicted))
        predicted_count += num_masks
        true_count += num_predicted
    cc.write('\nTotal cell count:\nPredicted: {}; True: {}\n'.format(predicted_count, true_count))
    counting_error = (abs(true_count - predicted_count)) / true_count
    cc.write('Total counting error rate: {}'.format(counting_error))

# Calculate AP
tau = np.arange(0.0, 1.01, 0.01)
ap_info = average_precision(labels, masks, threshold=tau)
ap_overall = np.average(ap_info[0], axis=0)
tp_overall = np.sum(ap_info[1], axis=0).astype('int32')
fp_overall = np.sum(ap_info[2], axis=0).astype('int32')
fn_overall = np.sum(ap_info[3], axis=0).astype('int32')
false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
with open(os.path.join(args.mask_path, 'AP_Results.pkl'), 'wb') as apr:
    pickle.dump((tau, ap_overall, tp_overall, fp_overall, fn_overall), apr)
plt.figure()
plt.plot(tau, ap_overall)
plt.title('Average Precision for Original Cellpose')
plt.xlabel(r'IoU Matching Threshold $\tau$')
plt.ylabel('Average Precision')
plt.yticks(np.arange(0, 1.01, step=0.2))
plt.savefig(os.path.join(args.mask_path, 'AP Results'))
