import argparse
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torch import nn, device, load, save, squeeze, as_tensor
from torch.cuda import is_available, device_count, empty_cache
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import time

from transforms import Resize, reformat
from loaddata import TrainCellTransposeData, ValTestCellTransposeData
from CellTranspose2D import CellTranspose, SizeModel, ClassLoss, FlowLoss, SASClassLoss, ContrastiveFlowLoss
from train_eval import train_network, adapt_network, eval_network
from cellpose_src.metrics import average_precision
from misc_utils import produce_logfile

parser = argparse.ArgumentParser()
parser.add_argument('--n-chan', type=int,
                    help='Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images).')
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--weight-decay', type=float)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--median-diams', type=int,
                    help='Median diameter size with which to resize images to. Note: If using pretrained model, ensure '
                         'that this variable remains the same as the given model.')
parser.add_argument('--patch-size', type=int,
                    help='Size of image patches with which to tile.')
parser.add_argument('--min-overlap', type=int,
                    help='Amount of overlap to use for tiling - currently the same for train and validation.')
parser.add_argument('--test-overlap', type=int,
                    help='Amount of overlap to use for tiling during testing - if unspecified, defaults to min-overlap')
parser.add_argument('--dataset-name', help='Name of dataset to use for reporting results (omit the word "Dataset").')
parser.add_argument('--results-dir', help='Folder in which to save experiment results.')
parser.add_argument('--train-only', help='Only perform training, no evaluation (mutually exclusive with "eval-only").',
                    action='store_true')
parser.add_argument('--eval-only', help='Only perform evaluation, no training (mutually exclusive with "train-only").',
                    action='store_true',)
parser.add_argument('--pretrained-model', help='Location of pretrained model to load in. Default: None')
parser.add_argument('--do-adaptation', help='Whether to perform domain adaptation or standard training.',
                    action='store_true')
parser.add_argument('--do-3D', help='Whether or not to use CellTranspose3D (Must use 3D volumes).',
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
parser.add_argument('--target-dataset',
                    help='The directory containing target data to be used for domain adaptation. Note: if do-adaptation'
                         ' is set to False, this parameter will be ignored.', nargs='+')
parser.add_argument('--target-from-3D', help='Whether the input target data is 3D: assumes 2D if set to False.',
                    action='store_true')
parser.add_argument('--target-flows', help='The directory(s) containing pre-calculated flows. If left empty, '
                                           'flows will be calculated manually.', nargs='+')
parser.add_argument('--cellpose-model',
                    help='Location of the generalized CellTranspose model to use for diameter estimation.')
parser.add_argument('--size-model', help='Location of the generalized size model to use for diameter estimation.')
parser.add_argument('--refine-prediction', help='Whether or not to apply refined diameter prediction with diameters of '
                                                'generalized Cellpose model predictions (better accuracy,'
                                                'slower evaluation).', action='store_true')
parser.add_argument('--calculate-ap', help='Whether to perform AP calculation at the end of evaluation.',
                    action='store_true')
parser.add_argument('--save-dataset', help='Name of directory to save training dataset to: '
                                           'if None, will not save dataset.')
parser.add_argument('--load-from-torch', help='If true, assumes dataset is being loaded from torch files, with no '
                                              'preprocessing required.', action='store_true')
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

args.median_diams = (args.median_diams, args.median_diams)
args.patch_size = (args.patch_size, args.patch_size)
args.min_overlap = (args.min_overlap, args.min_overlap)
ttt = None
tte = None
train_losses = None
if args.test_overlap is not None:
    args.test_overlap = (args.test_overlap, args.test_overlap)
else:
    args.test_overlap = args.min_overlap

if not (args.val_use_labels and args.test_use_labels):
    gen_cellpose = CellTranspose(channels=1, device='cuda:0')
    gen_cellpose = nn.DataParallel(gen_cellpose, device_ids=[0])
    gen_cellpose.load_state_dict(load(args.cellpose_model))

    gen_size_model = SizeModel().to('cuda:0')
    gen_size_model.load_state_dict(load(args.size_model))
else:
    gen_cellpose = None
    gen_size_model = None

model = CellTranspose(channels=args.n_chan, device=device)
model = nn.DataParallel(model)
model.to(device)
if args.pretrained_model is not None:
    model.load_state_dict(load(args.pretrained_model))

if not args.eval_only:
    class_loss = ClassLoss(nn.BCEWithLogitsLoss(reduction='mean'))
    flow_loss = FlowLoss(nn.MSELoss(reduction='mean'))
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate/100, last_epoch=-1)
    if args.load_from_torch:
        print('Loading Saved Training Dataset... ', end='')
        train_dataset = load(args.train_dataset[0])
        print('Done.')
    else:
        train_dataset = TrainCellTransposeData('Training', args.train_dataset, args.n_chan, do_3D=args.do_3D, from_3D=args.train_from_3D,
                                            crop_size=args.patch_size, has_flows=False,
                                            resize=Resize(args.median_diams, args.patch_size, args.min_overlap,
                                                   use_labels=True, patch_per_batch=args.batch_size))
        #train_dataset.process_training_data(args.patch_size, args.min_overlap, has_flows=False)
    if args.save_dataset:
        print('Saving Training Dataset... ', end='')
        save(train_dataset, args.save_dataset)
        print('Saved.')

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.val_dataset is not None:
        val_dataset = ValTestCellTransposeData('Validation', args.val_dataset, args.n_chan, do_3D=args.do_3D,
                                               from_3D=args.val_from_3D,
                                               resize=Resize(args.median_diams, args.patch_size, args.min_overlap,
                                                             use_labels=args.val_use_labels, refine=True,
                                                             gc_model=gen_cellpose, sz_model=gen_size_model,
                                                             device=device, patch_per_batch=args.batch_size))
        val_dataset.pre_generate_validation_patches(patch_size=args.patch_size, min_overlap=args.min_overlap)
        val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_dl = None
        print('No validation data given --> skipping validation.')

    if args.do_adaptation:
        sas_class_loss = SASClassLoss(nn.BCEWithLogitsLoss(reduction='mean'))
        c_flow_loss = ContrastiveFlowLoss(nn.MSELoss(reduction='mean'))
        target_dataset = TrainCellTransposeData('Target', args.target_dataset, args.n_chan, pf_dirs=args.target_flows,
                                                do_3D=args.do_3D, from_3D=args.target_from_3D,
                                                crop_size=args.patch_size, has_flows=False,
                                                resize=Resize(args.median_diams, args.patch_size, args.min_overlap,
                                                              use_labels=True, patch_per_batch=args.batch_size))
        #target_dataset.process_training_data(args.patch_size, args.min_overlap, batch_size=args.batch_size, has_flows=True)
        rs = RandomSampler(target_dataset, replacement=False)
        bs = BatchSampler(rs, args.batch_size, True)
        target_dl = DataLoader(target_dataset, batch_sampler=bs)

        start_train = time.time()
        train_losses, val_losses = adapt_network(model, train_dl, target_dl, val_dl, sas_class_loss, c_flow_loss,
                                                 class_loss, flow_loss, optimizer=optimizer, scheduler=scheduler,
                                                 device=device, n_epochs=args.epochs)
    else:
        start_train = time.time()
        train_losses, val_losses = train_network(model, train_dl, val_dl, class_loss, flow_loss,
                                                 optimizer=optimizer, scheduler=scheduler, device=device, n_epochs=args.epochs)
    save(model.state_dict(), os.path.join(args.results_dir, 'trained_model.pt'))
    end_train = time.time()
    ttt = time.strftime("%H:%M:%S", time.gmtime(end_train - start_train))
    print('Time to train: {}'.format(ttt))

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
    test_dataset = ValTestCellTransposeData('Test', args.test_dataset, args.n_chan, do_3D=args.do_3D,
                                            from_3D=args.test_from_3D, evaluate=True,
                                            resize=Resize(args.median_diams, args.patch_size, args.test_overlap,
                                                          use_labels=args.test_use_labels, refine=True,
                                                          gc_model=gen_cellpose, sz_model=gen_size_model,
                                                          device=device, patch_per_batch=args.batch_size))

    eval_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    masks, prediction_list, label_list = eval_network(model, eval_dl, device, patch_per_batch=args.batch_size,
                                                      patch_size=args.patch_size, min_overlap=args.test_overlap)

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
    tte = time.strftime("%H:%M:%S", time.gmtime(end_eval - start_eval))
    print('Time to evaluate: {}'.format(tte))

    with open(os.path.join(args.results_dir, 'counted_cells.txt'), 'w') as cc:
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
        # TODO: Have working with 3D as well (possibly re-initialize test dataset without resizing)
        if args.calculate_ap:
            labels = []
            for l in test_dataset.l_list:
                label = as_tensor(cv2.imread(l, -1).astype('int16'))
                # label = as_tensor(tifffile.imread(l).astype('int16'))
                label = squeeze(reformat(label), dim=0).numpy().astype('int16')
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
            plt.title('Average Precision for CellTranspose on {} Dataset'.format(args.dataset_name))
            plt.xlabel(r'IoU Matching Threshold $\tau$')
            plt.ylabel('Average Precision')
            plt.yticks(np.arange(0, 1.01, step=0.2))
            plt.savefig(os.path.join(args.results_dir, 'AP Results'))
            plt.show()
            cc.write('\nAP Results at IoU threshold 0.5: AP = {}\nTrue Postive: {}; False Positive: {}; False Negative:'
                     ' {}\n'.format(ap_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
            print('AP Results at IoU threshold 0.5: AP = {}\nTrue Postive: {}; False Positive: {}; False Negative: {}'
                  .format(ap_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))
            false_error = (fp_overall[51] + fn_overall[51]) / (tp_overall[51] + fn_overall[51])
            cc.write('Total false error rate: {:.6f}'.format(false_error))
            print('Total false error rate: {:.6f}'.format(false_error))
            with open(os.path.join(args.results_dir, '{}_AP_Results.pkl'.format(args.dataset_name)), 'wb') as apr:
                pickle.dump((tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error), apr)

print(args.results_dir)

produce_logfile(args, len(train_losses) if train_losses is not None else None, ttt, tte, num_workers)
