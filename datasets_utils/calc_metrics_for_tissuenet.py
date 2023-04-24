"""Used to calculate AP from masks obtained through original Cellpose model"""

import os
import numpy as np
import tifffile
import cv2
from ..cellpose_src.metrics import average_precision,_intersection_over_union, _label_overlap
import pickle
import matplotlib.pyplot as plt
import argparse
from skimage import measure

mask_path = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/full_data_lbl_0/test/mesmer_results'
label_path = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/full_data_lbl_0/test/labels'
dest = ''

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


# Count cells in each mask and calculate counting error
with open(os.path.join(dest, 'counted_cells.txt'), 'w') as cc:
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
    
        
    plt.figure()
    plt.plot(tau, ap_overall)
    plt.title('Average Precision for CellTranspose')
    plt.xlabel(r'IoU Matching Threshold $\tau$')
    plt.ylabel('Average Precision')
    plt.yticks(np.arange(0, 1.01, step=0.2))
    plt.savefig(os.path.join(dest, 'AP Results'))
    cc.write('\nAP Results at IoU threshold 0.5: AP = {}\nTrue Postive: {}; False Positive: {};'
            'False Negative: {}\n'.format(ap_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
    print('AP Results at IoU threshold 0.5: AP = {}\nTrue Postive: {}; False Positive: {}; '
        'False Negative: {}'.format(ap_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
    false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
    cc.write('Total false error rate: {:.6f}'.format(false_error))
    print('Total false error rate: {:.6f}'.format(false_error))
    with open(os.path.join(dest, 'AP_Results.pkl'), 'wb') as apr:
        pickle.dump((tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error), apr)