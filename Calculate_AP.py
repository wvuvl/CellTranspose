import os
from cellpose_src.metrics import average_precision
import argparse
from tifffile import imread
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser()
# parser.add_argument('--true-dir', help='The directory containing ground truth masks.')
# parser.add_argument('--pred-dir', help='The directory containing the predicted masks.')
# args = parser.parse_args()
# true_dir = '/media/matthew/Data Drive/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim/labels'
# pred_dir = '/media/matthew/Data Drive/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim/results/results_5/tiff_results'
# true_dir = '/media/matthew/Data Drive/Datasets/Neuro_Proj1_Data/BBBC024_v1_2D_raw_tiff_combined_split/test/labels'
# pred_dir = '/media/matthew/Data Drive/Datasets/Neuro_Proj1_Data/BBBC024_v1_2D_raw_tiff_combined_split/results/training_results_4/tiff_results'
true_dir = '/media/matthew/Data Drive/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim/split_data/test/labels'
pred_dir = '/media/matthew/Data Drive/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim/results/results_04/tiff_results'

# Load all true labels into (sorted) list of numpy arrays from directory, including only '.tif' files
# if os.path.isdir(true_dir):
assert os.path.isdir(true_dir)
assert os.path.isdir(pred_dir)
true_tiffs = sorted([true_dir + os.sep + file for file in os.listdir(true_dir) if file.endswith('.tif')])
masks_true = list(map(imread, tqdm(true_tiffs, desc='True labels')))
if np.ndim(masks_true) == 4:
    masks_true = [mask[:, :, 2] for mask in masks_true]  # Added to solve SIGSEGV error encountered when masks_true and masks_pred have different dimension numbers
# Load all predicted labels into (sorted) list of numpy arrays from directory, including only '.tif' files
# if os.path.isdir(pred_dir):
pred_tiffs = sorted([pred_dir + os.sep + file for file in os.listdir(pred_dir) if file.endswith('.tif')])
masks_pred = list(map(imread, tqdm(pred_tiffs, desc='Predicted labels')))

tau = np.arange(0.01, 1.01, 0.01)
ap_info = average_precision(masks_true, masks_pred, threshold=tau)
ap_per_im = ap_info[0]
ap_overall = np.average(ap_per_im, axis=0)

plt.figure()
plt.plot(tau, ap_overall)
plt.title('Average Precision for Cellpose on Given Dataset')
plt.xlabel(r'IoU Matching Threshold $\tau$')
plt.ylabel('Average Precision')
plt.show()

print('test')
