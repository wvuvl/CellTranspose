import matplotlib.pyplot as plt
from torch import as_tensor, squeeze
import numpy as np
import cv2
import os
import pickle
import tifffile

from transforms import Resize, reformat, labels_to_flows, followflows
from loaddata import CellPoseData
from cellpose_src.metrics import average_precision

dataset_name = "Cellpose_Specialized"
results_dir = "/home/matthew/Documents/Datasets/Neuro_Proj1_Data/DA_Results/Perfect_Prediction_Results_1"
dataset_dir = "/home/matthew/Documents/Datasets/Neuro_Proj1_Data/Cellpose_Dataset/Specialized/test"
median_diams = (30, 30)

assert not os.path.exists(results_dir),\
    'Results folder {} currently exists; please specify new location to save results.'.format(results_dir)
os.mkdir(results_dir)
os.mkdir(os.path.join(results_dir, 'tiff_results'))

dataset = CellPoseData('Perfect Prediction Reconstruction', dataset_dir, n_chan=2, do_3D=False, from_3D=False,
                       resize=Resize(median_diams, use_labels=True, patch_per_batch=1))
# Labels to Flows
labels = dataset.labels
original_dims = dataset.original_dims
flows = []
for i in range(len(labels)):
    flow = as_tensor([labels_to_flows(labels[i][j].numpy()) for j in range(len(labels[i]))])
    flows.append(flow)

# Flows to Masks
masks = []
for i in range(len(flows)):
    sample_mask = followflows(flows[i])
    sample_mask = np.transpose(sample_mask.numpy(), (1, 2, 0))
    sample_mask = cv2.resize(sample_mask, (original_dims[i][1], original_dims[i][0]),
                             interpolation=cv2.INTER_NEAREST)
    masks.append(sample_mask)

print('test')

# Save Masks
for i in range(len(masks)):
    masks[i] = masks[i].astype('int32')
    tifffile.imwrite(os.path.join(results_dir, 'tiff_results', str(i) + '.tif'), masks[i])

# Calculate Error and AP
labels = []
for l in dataset.l_list:
    label = as_tensor(cv2.imread(l, -1).astype('int16'))
    # label = as_tensor(tifffile.imread(l).astype('int16'))
    label = squeeze(reformat(label), dim=0).numpy()
    labels.append(label)

with open(os.path.join(results_dir, 'counted_cells.txt'), 'w') as cc:
    predicted_count = 0
    true_count = 0
    for i in range(len(dataset)):
        num_masks = len(np.unique(masks[i]))-1
        num_predicted = len(np.unique(dataset.labels[i]))-1
        cc.write('{}:\nPredicted: {}; True: {}\n'.format(dataset.d_list[i], num_masks, num_predicted))
        predicted_count += num_masks
        true_count += num_predicted
    cc.write('\nTotal cell count:\nPredicted: {}; True: {}\n'.format(predicted_count, true_count))
    counting_error = (abs(true_count - predicted_count)) / true_count
    cc.write('Total counting error rate: {:.6f}'.format(counting_error))
    print('Total cell count:\nPredicted: {}; True: {}'.format(predicted_count, true_count))
    print('Total counting error rate: {}'.format(counting_error))
    tau = np.arange(0.0, 1.01, 0.01)
    ap_info = average_precision(labels, masks, threshold=tau)
    ap_per_im = ap_info[0]
    ap_overall = np.average(ap_per_im, axis=0)
    tp_overall = np.sum(ap_info[1], axis=0).astype('int32')
    fp_overall = np.sum(ap_info[2], axis=0).astype('int32')
    fn_overall = np.sum(ap_info[3], axis=0).astype('int32')
    plt.figure()
    plt.plot(tau, ap_overall)
    plt.title('Average Precision for Cellpose on {} Dataset'.format(dataset_name))
    plt.xlabel(r'IoU Matching Threshold $\tau$')
    plt.ylabel('Average Precision')
    plt.yticks(np.arange(0, 1.01, step=0.2))
    plt.savefig(os.path.join(results_dir, 'AP Results'))
    plt.show()
    cc.write('\nAP Results at IoU threshold 0.5: AP = {}\nTrue Postive: {}; False Positive: {}; False Negative:'
             ' {}\n'.format(ap_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
    print('AP Results at IoU threshold 0.5: AP = {}\nTrue Postive: {}; False Positive: {}; False Negative: {}'
          .format(ap_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
    false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
    cc.write('Total false error rate: {:.6f}'.format(false_error))
    print('Total false error rate: {:.6f}'.format(false_error))
    with open(os.path.join(results_dir, '{}_AP_Results.pkl'.format(dataset_name)), 'wb') as apr:
        pickle.dump((tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error), apr)

