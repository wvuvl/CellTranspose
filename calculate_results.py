"""
Utility code for various performance metrics
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os
import tifffile
import cv2
import pickle
import matplotlib.pyplot as plt
from cellpose_src.metrics import average_precision

sns.set()


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


def produce_logfile(args, epochs, ttt, tte):
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
        log.write('Minimum patch overlap: {} \n'.format(args.min_overlap))
        log.write('\n')
        log.write('Zeros removed: all\n')
        log.write('\n')
        if not args.eval_only:
            log.write('Adaptation: {}\n'.format(args.do_adaptation))
            if args.do_adaptation:
                log.write('Source dataset(s): {}\n'.format(args.train_dataset))
                log.write('Target dataset(s): {}\n'.format(args.target_dataset))
            else:
                log.write('Training dataset(s): {}\n'.format(args.train_dataset))
            log.write('Learning rate: {}; Momentum: {}\n'.format(args.learning_rate, args.momentum))
            log.write('Epochs: {}; Batch size: {}\n'.format(epochs, args.batch_size))
            log.write('Pretrained model: {}\n'.format(args.pretrained_model))
            log.write('\n')
            log.write('Validation dataset(s): {}\n'.format(args.val_dataset))
        if not args.train_only:
            log.write('Test dataset(s): {}\n'.format(args.test_dataset))

        log.write('{}'.format(str(args)))


def plot_loss(train_losses, results_dir, val_dl=None, val_losses=None):
    plt.figure()
    x_range = np.arange(1, len(train_losses)+1)
    plt.plot(x_range, train_losses)
    if val_dl is not None:
        plt.plot(x_range, val_losses)
        plt.legend(['Training Losses', 'Validation Losses'])
        plt.title('Training and Validation Losses')
    else:
        plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Combined Losses')
    plt.savefig(os.path.join(results_dir, 'Training-Validation Losses'))


def save_pred(masks, test_dataset, prediction_list, data_list, results_dir, dataset_name, calc_ap=False):
    """ Saves raw predictions and masks. Counts cells and computes AP.
    
    Parameters
    -----
    
        masks: predicted masks
        test_dataset: loaded test dataset
        prediction_list: raw predictions
        label_list: list containing file names of corresponding labels
        results_dir: directory where results are stored
        dataset_name: name of the dataset
    """
    
    for i in range(len(masks)):
        masks[i] = masks[i].astype('int32')
        with open(os.path.join(results_dir, 'pkl_results', data_list[i] + '_predicted_labels.pkl'), 'wb') as m_pkl:
            pickle.dump(masks[i], m_pkl)
        tifffile.imwrite(os.path.join(results_dir, 'tiff_results', data_list[i] + '.tif'), masks[i])
        with open(os.path.join(results_dir, 'pkl_results', data_list[i] + '_raw_masks_flows.pkl'), 'wb') as rmf_pkl:
            pickle.dump(prediction_list[i], rmf_pkl)
        tifffile.imwrite(os.path.join(results_dir, 'raw_predictions_tiffs', data_list[i] + '.tif'),
                         prediction_list[i])

    if calc_ap:
        if len(test_dataset.l_list) != 0:
            with open(os.path.join(results_dir, 'counted_cells.txt'), 'w') as cc:
                predicted_count = 0
                true_count = 0
                for i in range(len(test_dataset)):
                    num_masks = len(np.unique(masks[i]))-1
                    num_labels = len(np.unique(test_dataset.labels[i]))-1
                    cc.write('{}:\nPredicted: {}; True: {}\n'.format(test_dataset.d_list[i], num_masks, num_labels))
                    predicted_count += num_masks
                    true_count += num_labels
                cc.write('\nTotal cell count:\nPredicted: {}; True: {}\n'.format(predicted_count, true_count))
                counting_error = (abs(true_count - predicted_count)) / true_count
                cc.write('Total counting error rate: {:.6f}'.format(counting_error))
                print('Total cell count:\nPredicted: {}; True: {}'.format(predicted_count, true_count))
                print('Total counting error rate: {}'.format(counting_error))

                # AP Calculation
                labels = []
                    
                for l in test_dataset.l_list:

                    if os.path.splitext(l)[-1] == '.tif':
                        label = tifffile.imread(l).astype('int32')
                    else:
                        label = cv2.imread(l, -1).astype('int32')

                    labels.append(label)
                tau = np.arange(0.0, 1.01, 0.01)
                ap_info = average_precision(labels, masks, threshold=tau)
                ap_per_im = ap_info[0]
                ap_overall = np.average(ap_per_im, axis=0)
                tp_overall = np.sum(ap_info[1], axis=0).astype('int32')
                fp_overall = np.sum(ap_info[2], axis=0).astype('int32')
                fn_overall = np.sum(ap_info[3], axis=0).astype('int32')
                f1_overall = tp_overall / (tp_overall + 0.5 * (fp_overall+ fn_overall))
    
                plt.figure()
                plt.plot(tau, ap_overall)
                plt.title('Average Precision for CellTranspose')
                plt.xlabel(r'IoU Matching Threshold $\tau$')
                plt.ylabel('Average Precision')
                plt.yticks(np.arange(0, 1.01, step=0.2))
                plt.savefig(os.path.join(results_dir, 'AP Results'))
                
                plt.figure()
                plt.plot(tau, f1_overall)
                plt.title('F1 Score for CellTranspose')
                plt.xlabel(r'IoU Matching Threshold $\tau$')
                plt.ylabel('F1 Score')
                plt.yticks(np.arange(0, 1.01, step=0.2))
                plt.savefig(os.path.join(results_dir, 'F1 Score'))
                
                cc.write('\nAP Results at IoU threshold 0.5: AP = {}\nF1 score at IoU threshold 0.5: F1 = {} \nTrue Postive: {}; False Positive: {};'
                        'False Negative: {}\n'.format(ap_overall[51], f1_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
                
                
                print('AP Results at IoU threshold 0.5: AP = {}\nF1 score at IoU threshold 0.5: F1 = {} \nTrue Postive: {}; False Positive: {}; '
                    'False Negative: {}'.format(ap_overall[51], f1_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
                false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
                cc.write('Total false error rate: {:.6f}'.format(false_error))
                print('Total false error rate: {:.6f}'.format(false_error))
                with open(os.path.join(results_dir, '{}_AP_Results.pkl'.format(dataset_name)), 'wb') as apr:
                    pickle.dump((tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error), apr)
        else:
            print('>>> Folder containing labelled images does not exist, cannot calculate AP...')
        