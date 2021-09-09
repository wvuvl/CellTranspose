import argparse
from torch.utils.data import DataLoader
from torch import nn, device, load, save, squeeze, as_tensor
from torch.cuda import is_available, device_count, empty_cache
from torch.optim import SGD
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import time

from transforms import Resize, reformat
from loaddata import CellPoseData
from Cellpose_2D_PyTorch import UpdatedCellpose, SizeModel, class_loss, flow_loss, sas_class_loss
from train_eval import train_network, adapt_network, eval_network
from cellpose_src.metrics import average_precision

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--median-diams', type=int,
                    help='Median diameter size with which to resize images to. Note: If using pretrained model, ensure '
                         'that this variable remains the same as the given model.')
parser.add_argument('--patch-size', type=int,
                    help='Size of image patches with which to tile.')
parser.add_argument('--min-overlap', type=int,
                    help='Amount of overlap to use for tiling - currently the same for train, validation, and test.')
parser.add_argument('--dataset-name', help='Name of dataset to use for reporting results (omit the word "Dataset").')
parser.add_argument('--results-dir', help='Folder in which to save experiment results.')
parser.add_argument('--train-only', help='Only perform training, no evaluation (mutually exclusive with "eval-only").',
                    action='store_true')
parser.add_argument('--eval-only', help='Only perform evaluation, no training (mutually exclusive with "train-only").',
                    action='store_true',)
parser.add_argument('--pretrained-model', help='Pretrained model to load in')
parser.add_argument('--do-adaptation', help='Whether to perform domain adaptation or standard training.',
                    action='store_true')
parser.add_argument('--do-3D', help='Whether or not to use 3D-Cellpose (Must use 3D volumes).',
                    action='store_true')
parser.add_argument('--train-dataset', help='The directory(s) containing (source) data to be used for training.',
                    nargs='+')
parser.add_argument('--train-from-3D', help='Whether the input training source data is 3D: assumes 2D if set to False.',
                    action='store_true')
parser.add_argument('--val-dataset', help='The directory(s) containing data to be used for validation.', nargs='+')
parser.add_argument('--val-from-3D', help='Whether the input validation data is 3D: assumes 2D if set to False.',
                    action='store_true')
parser.add_argument('--val-use-labels', help='Whether to use labels for resizing validation data.',
                    action='store_true')
parser.add_argument('--test-dataset', help='The directory(s) containing data to be used for testing.', nargs='+')
parser.add_argument('--test-from-3D', help='Whether the input test data is 3D: assumes 2D if set to False.',
                    action='store_true')
parser.add_argument('--test-use-labels', help='Whether to use labels for resizing test data.',
                    action='store_true')
parser.add_argument('--target-dataset', help='The directory containing target data to be used for domain adaptation.'
                                             'Note: if do-adaptation is set to False, this parameter will be ignored.',
                    nargs='+')
parser.add_argument('--target-from-3D', help='Whether the input target data is 3D: assumes 2D if set to False.',
                    action='store_true')
parser.add_argument('--target-flows', help='The directory(s) containing pre-calculated flows. If left empty, '
                                           'flows will be calculated manually.', nargs='+')
parser.add_argument('--cellpose-model',
                    help='Location of the generalized cellpose model to use for diameter estimation.')
parser.add_argument('--size-model', help='Location of the generalized size model to use for diameter estimation.')
parser.add_argument('--refine-prediction', help='Whether or not to apply refined diameter prediction with diameters of '
                                                'generalized Cellpose model predictions (better accuracy,'
                                                'slower evaluation).', action='store_true')
parser.add_argument('--calculate-ap', help='Whether to perform AP calculation at the end of evaluation.',
                    action='store_true')
args = parser.parse_args()

print(args.results_dir)

assert not os.path.exists(args.results_dir),\
    'Results folder {} currently exists; please specify new location to save results.'.format(args.results_dir)
os.mkdir(args.results_dir)
os.mkdir(os.path.join(args.results_dir, 'tiff_results'))
os.mkdir(os.path.join(args.results_dir, 'raw_predictions_tiffs'))
assert not (args.train_only and args.eval_only), 'Cannot pass in "train-only" and "eval-only" arguments simultaneously.'
num_workers = device_count()
device = device('cuda' if is_available() else 'cpu')
empty_cache()

median_diams = (args.median_diams, args.median_diams)
patch_size = (args.patch_size, args.patch_size)
min_overlap = (args.min_overlap, args.min_overlap)

if not (args.val_use_labels and args.test_use_labels):
    gen_cellpose = UpdatedCellpose(channels=1, device='cuda:0')
    gen_cellpose = nn.DataParallel(gen_cellpose, device_ids=[0])
    gen_cellpose.load_state_dict(load(args.cellpose_model))

    gen_size_model = SizeModel().to('cuda:0')
    gen_size_model.load_state_dict(load(args.size_model))
else:
    gen_cellpose = None
    gen_size_model = None

model = UpdatedCellpose(channels=1, device=device)
model = nn.DataParallel(model)
model.to(device)
if args.pretrained_model is not None:
    model.load_state_dict(load(args.pretrained_model))

if not args.eval_only:
    start_train = time.time()
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    train_dataset = CellPoseData('Training', args.train_dataset, do_3D=args.do_3D, from_3D=args.train_from_3D,
                                 resize=Resize(median_diams, use_labels=True,
                                               patch_per_batch=args.batch_size))
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.val_dataset is not None:
        val_dataset = CellPoseData('Validation', args.val_dataset, do_3D=args.do_3D, from_3D=args.val_from_3D,
                                   resize=Resize(median_diams, use_labels=args.val_use_labels, refine=True,
                                                 gc_model=gen_cellpose, sz_model=gen_size_model,
                                                 min_overlap=min_overlap, device=device,
                                                 patch_per_batch=args.batch_size)
                                   )
        val_dataset.pre_generate_patches(patch_size=patch_size, min_overlap=min_overlap)
        val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_dl = None
        print('No validation data given --> skipping validation.')

    if args.do_adaptation:
        target_dataset = CellPoseData('Target', args.target_dataset, pf_dirs=args.target_flows, do_3D=args.do_3D,
                                      from_3D=args.target_from_3D, resize=Resize(median_diams, use_labels=True,
                                                                                 patch_per_batch=args.batch_size))
        target_dl = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
        train_losses, val_losses = adapt_network(model, train_dl, target_dl, val_dl, sas_class_loss, class_loss,
                                                 flow_loss, patch_size=patch_size, min_overlap=min_overlap,
                                                 optimizer=optimizer, device=device, n_epochs=args.epochs)
    else:
        train_losses, val_losses = train_network(model, train_dl, val_dl, class_loss, flow_loss,
                                                 patch_size, min_overlap,
                                                 optimizer=optimizer, device=device, n_epochs=args.epochs)
    save(model.state_dict(), os.path.join(args.results_dir, 'trained_model.pt'))
    end_train = time.time()
    print('Time to train: {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_train - start_train))))

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
    plt.savefig(os.path.join(args.results_dir, 'Training-Validation Losses'))
    plt.show()


if not args.train_only:
    start_eval = time.time()
    test_dataset = CellPoseData('Test', args.test_dataset, do_3D=args.do_3D, from_3D=args.test_from_3D, evaluate=True,
                                resize=Resize(median_diams, use_labels=args.test_use_labels, refine=True,
                                              gc_model=gen_cellpose, sz_model=gen_size_model,
                                              min_overlap=min_overlap, device=device,
                                              patch_per_batch=args.batch_size))

    eval_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    masks, prediction_list, label_list = eval_network(model, eval_dl, device, patch_per_batch=args.batch_size,
                                                      patch_size=patch_size, min_overlap=min_overlap)

    for i in range(len(masks)):
        masks[i] = masks[i].astype('int32')
        with open(os.path.join(args.results_dir, label_list[i] + '_predicted_labels.pkl'), 'wb') as m_pkl:
            pickle.dump(masks[i], m_pkl)
        tifffile.imwrite(os.path.join(args.results_dir, 'tiff_results', label_list[i] + '.tif'),
                         masks[i])
        with open(os.path.join(args.results_dir, label_list[i] + '_raw_masks_flows.pkl'), 'wb') as rmf_pkl:
            pickle.dump(prediction_list[i], rmf_pkl)
        tifffile.imwrite(os.path.join(args.results_dir, 'raw_predictions_tiffs', label_list[i] + '.tif'),
                         prediction_list[i])
    end_eval = time.time()
    print('Time to evaluate: {}'.format(time.strftime("%H:%M:%S", time.gmtime(end_eval - start_eval))))

    with open(os.path.join(args.results_dir, 'counted_cells.txt'), 'w') as cc:
        predicted_count = 0
        true_count = 0
        for i in range(len(test_dataset)):
            num_masks = len(np.unique(masks[i]))-1
            num_predicted = len(np.unique(test_dataset.labels[i]))-1
            cc.write('{}:\nPredicted: {}; True: {}\n'.format(test_dataset.d_list[i], num_masks, num_predicted))
            predicted_count += num_masks
            true_count += num_predicted
        cc.write('\nTotal cell count:\nPredicted: {}; True: {}\n'.format(predicted_count, true_count))
        counting_error = (abs(true_count - predicted_count)) / true_count
        cc.write('Total counting error rate: {}'.format(counting_error))
    print('Total cell count:\nPredicted: {}; True: {}'.format(predicted_count, true_count))
    print('Total counting error rate: {}'.format(counting_error))

    # AP Calculation
    if args.calculate_ap:
        labels = []
        for l in test_dataset.l_list:
            label = as_tensor(tifffile.imread(l).astype('int16'))
            label = squeeze(reformat(label), dim=0).numpy()
            labels.append(label)
        tau = np.arange(0.0, 1.01, 0.01)
        ap_info = average_precision(labels, masks, threshold=tau)
        ap_per_im = ap_info[0]
        ap_overall = np.average(ap_per_im, axis=0)
        tp_overall = np.sum(ap_info[1], axis=0).astype('int32')
        fp_overall = np.sum(ap_info[2], axis=0).astype('int32')
        fn_overall = np.sum(ap_info[3], axis=0).astype('int32')
        plt.figure()
        plt.plot(tau, ap_overall)
        plt.title('Average Precision for Cellpose on {} Dataset'.format(args.dataset_name))
        plt.xlabel(r'IoU Matching Threshold $\tau$')
        plt.ylabel('Average Precision')
        plt.yticks(np.arange(0, 1.01, step=0.2))
        plt.savefig(os.path.join(args.results_dir, 'AP Results'))
        plt.show()
        print('AP Results at IoU threshold 0.5:\nTrue Postive: {}; False Positive: {}; False Negative: {}'
              .format(tp_overall[51], fp_overall[51], fn_overall[51]))
        false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
        print('Total false error rate: {}'.format(false_error))
        with open(os.path.join(args.results_dir, '{}_AP_Results.pkl'.format(args.dataset_name)), 'wb') as apr:
            pickle.dump((tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error), apr)

with open(os.path.join(args.results_dir, 'logfile.txt'), 'w') as log:
    if args.train_only:
        log.write('train-only\n')
    if not args.eval_only:
        log.write('Time to train: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(end_train - start_train))))
    else:
        log.write('eval-only\n')
    if not args.train_only:
        log.write('Time to evaluate: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(end_eval - start_eval))))
    log.write('\n')
    log.write('Cells resized to possess median diameter of {}.\n'.format(args.median_diams))
    log.write('Patch size: {}\n'.format(args.patch_size))
    log.write('Minimum patch overlap: {}\n'.format(args.min_overlap))
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
        log.write('Epochs: {}; Batch size: {}\n'.format(args.epochs, args.batch_size))
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
