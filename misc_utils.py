"""
Utility code for miscellaneous tasks
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import time

sns.set()


# TODO: Yet to be updated to work for this project
def create_confusion_matrix(y_true, y_pred, classes, normalize=None):
    # Get data into confusion matrix (array)
    num_classes = len(classes)
    size = len(y_true)
    cols = ['True Values', 'Predicted Values', 'values']
    conf_mat = np.zeros((num_classes, num_classes))
    for i in range(size):
        conf_mat[y_true[i], y_pred[i]] += 1

    # Normalization procedure
    if normalize == 'all':
        conf_mat = np.divide(conf_mat, size)
    elif normalize == 'True':
        true_sum = np.sum(conf_mat, axis=1)
        cm_t = np.transpose(conf_mat)
        conf_mat = np.transpose(np.divide(cm_t, true_sum))
    elif normalize == 'Pred':
        pred_sum = np.sum(conf_mat, axis=0)
        conf_mat = np.divide(conf_mat, pred_sum)
    conf_frame = pd.DataFrame([], columns=cols)
    for i in range(num_classes):
        for j in range(num_classes):
            new_row = pd.DataFrame([[classes[i], classes[j], conf_mat[i][j]]], columns=cols)
            conf_frame = conf_frame.append(new_row)
    conf_pivot = conf_frame.pivot(index=cols[0], columns=cols[1], values=cols[2])
    conf_pivot = conf_pivot.reindex(classes)
    conf_pivot = conf_pivot.reindex(columns=classes)

    if normalize:
        sns.heatmap(conf_pivot, annot=True, cmap="BuGn", fmt='.3f', cbar=False)
    else:
        sns.heatmap(conf_pivot, annot=True, cmap="BuGn", fmt='g', cbar=False)

def produce_logfile(args, epochs, ttt, tte, num_workers):
    with open(os.path.join(args.results_dir, 'logfile.txt'), 'w') as log:
        if args.train_only:
            log.write('train-only\n')
        if not args.eval_only:
            log.write('Time to train: {}\n'.format(ttt))
        else:
            log.write('eval-only\n')
        if not args.train_only:
            log.write('Time to evaluate: {}\n'.format(tte))
        log.write('\n')
        log.write('Number of channels: {}\n'.format(args.n_chan))
        log.write('Cells resized to possess median diameter of {}.\n'.format(args.median_diams))
        log.write('Patch size: {}\n'.format(args.patch_size))
        log.write('Minimum patch overlap (train): {}\n'.format(args.min_overlap))
        log.write('Minimum patch overlap (test): {} \n'.format(args.test_overlap))
        log.write('\n')
        log.write('Zeros removed: all\n')
        log.write('\n')
        if not args.eval_only:
            log.write('Training dataset(s): {}\n'.format(args.train_dataset))
            log.write('Learning rate: {}; Momentum: {}\n'.format(args.learning_rate, args.momentum))
            log.write('Epochs: {}; Batch size: {}\n'.format(epochs, args.batch_size))
            log.write('GPUs: {}\n'.format(num_workers))
            log.write('Pretrained model: {}\n'.format(args.pretrained_model))
            log.write('\n')
            log.write('Validation dataset(s): {}\n'.format(args.val_dataset))
            log.write('Labels used for validation: {}\n'.format(args.val_use_labels))
        if not args.train_only:
            log.write('Test dataset(s): {}\n'.format(args.test_dataset))
            log.write('Labels used for testing: {}\n'.format(args.test_use_labels))
        log.write('Refined size predictions: {}\n'.format(args.refine_prediction))
        log.write('\n')
        log.write('Cellpose model for size prediction: {}\n'.format(args.cellpose_model))
        log.write('Size model: {}'.format(args.size_model))
