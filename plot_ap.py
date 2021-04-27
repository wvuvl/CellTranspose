"""
Takes input labels and predicted masks, and generates plot of average precision (AP = TP / (TP + FP + FN))

Created by Matthew Keaton on 3/16/21
"""

import os
import tifffile
import pickle
from cellpose_src.metrics import average_precision
import matplotlib.pyplot as plt

label_dir = '/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim/labels'
prediction_dir = '/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim/results/results3'
import numpy as np

# Load in Ground Truth labels
labels = []
for label_file in sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]):
    labels.append(tifffile.imread(os.path.join(label_dir, label_file)))

predictions = []
for prediction_file in sorted([f for f in os.listdir(prediction_dir) if f.lower().endswith('.pkl')]):
    with open(os.path.join(prediction_dir, prediction_file), 'rb') as pf:
        predictions.append(pickle.load(pf))

threshold = np.linspace(0, 1, 101)
ap_per_im, _, _, _ = average_precision(labels, predictions, threshold=threshold)
overall_ap = ap_per_im.mean(axis=0)

plt.figure()
plt.plot(threshold, overall_ap)
plt.title('Average Precision')  # Could add further arguments here
plt.xlabel(r'IoU Matching Threshold $\tau$')
plt.ylabel('Average Precision')
plt.savefig(os.path.join(prediction_dir, 'Average_Precision.pdf'))

print('test')
