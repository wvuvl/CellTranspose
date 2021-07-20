import argparse
import torchvision
from torch.utils.data import DataLoader
from torch import nn, device, load, save
from torch.cuda import is_available, device_count, empty_cache
from torch.optim import SGD
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from transforms import Reformat, Normalize1stTo99th
from loaddata import StandardizedTiffData
from Cellpose_2D_PyTorch import UpdatedCellpose, SizeModel, class_loss, flow_loss, sas_class_loss
from train_eval import train_network, adapt_network, eval_network
from cellpose_src.metrics import average_precision

parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--patches-per-batch', type=int,
                    help='Number of patches to pass into GPU at once - effectively batch size.')
parser.add_argument('--dataset-name', help='Name of dataset to use for reporting results (omit the word "Dataset").')
parser.add_argument('--results-dir', help='Folder in which to save experiment results.')
parser.add_argument('--train-only', help='Only perform training, no evaluation (mutually exclusive with "eval-only").',
                    action='store_true', default=False)
parser.add_argument('--eval-only', help='Only perform evaluation, no training (mutually exclusive with "train-only").',
                    action='store_true', default=False,)
parser.add_argument('--pretrained-model', help='Pretrained model to load in')
parser.add_argument('--do-adaptation', help='Whether to perform domain adaptation or standard training.',
                    action='store_true', default=False)
parser.add_argument('--do-3D', help='Whether or not to use 3D-Cellpose (Must use 3D volumes).',
                    action='store_true', default=False)
parser.add_argument('--train-dataset', help='The directory(s) containing (source) data to be used for training.',
                    nargs='+')
parser.add_argument('--train-from-3D', help='Whether the input training source data is 3D: assumes 2D if set to False.',
                    action='store_true', default=False)
parser.add_argument('--val-dataset', help='The directory(s) containing data to be used for validation.', nargs='+')
parser.add_argument('--val-from-3D', help='Whether the input validation data is 3D: assumes 2D if set to False.',
                    action='store_true', default=False)
parser.add_argument('--test-dataset', help='The directory(s) containing data to be used for testing.', nargs='+')
parser.add_argument('--test-from-3D', help='Whether the input test data is 3D: assumes 2D if set to False.',
                    action='store_true', default=False)
parser.add_argument('--target-dataset', help='The directory containing target data to be used for domain adaptation.'
                                             'Note: if do-adaptation is set to False, this parameter will be ignored.',
                    nargs='+')
parser.add_argument('--target-from-3D', help='Whether the input target data is 3D: assumes 2D if set to False.',
                    action='store_true', default=False)
parser.add_argument('--cellpose-model',
                    help='Location of the generalized cellpose model to use for diameter estimation.')
parser.add_argument('--size-model', help='Location of the generalized size model to use for diameter estimation.')
parser.add_argument('--refine-prediction', help='Whether or not to apply refined diameter prediction with diameters of '
                                                'generalized Cellpose model predictions (better accuracy,'
                                                'slower evaluation).', action='store_true', default=False)
parser.add_argument('--calculate-ap', help='Whether to perform AP calculation at the end of evaluation.',
                    action='store_true', default=False)
args = parser.parse_args()

assert not os.path.exists(args.results_dir),\
    'Results folder currently exists; please specify new location to save results.'
os.mkdir(args.results_dir)
os.mkdir(os.path.join(args.results_dir, 'tiff_results'))
assert not (args.train_only and args.eval_only), 'Cannot pass in "train-only" and "eval-only" arguments simultaneously.'
num_workers = device_count()
device = device('cuda' if is_available() else 'cpu')
empty_cache()

# Default median diameter to resize cells to
median_diams = (24, 24)
gen_cellpose = UpdatedCellpose(channels=1, device=device)
gen_cellpose = nn.DataParallel(gen_cellpose)
gen_cellpose.to(device)
gen_cellpose.load_state_dict(load(args.cellpose_model))

gen_size_model = SizeModel().to(device)
gen_size_model.load_state_dict(load(args.size_model))

data_transform = torchvision.transforms.Compose([
    Reformat(do_3D=args.do_3D),
    Normalize1stTo99th()
])
label_transform = torchvision.transforms.Compose([
    Reformat(do_3D=args.do_3D)
])

model = UpdatedCellpose(channels=1, device=device)
model = nn.DataParallel(model)
model.to(device)
if args.pretrained_model is not None:
    model.load_state_dict(load(args.pretrained_model))

if not args.eval_only:
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    train_dataset = StandardizedTiffData('Training', args.train_dataset, do_3D=args.do_3D, from_3D=args.train_from_3D,
                                         d_transform=data_transform, l_transform=label_transform)
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = StandardizedTiffData('Validation', args.val_dataset, do_3D=args.do_3D, from_3D=args.val_from_3D,
                                       d_transform=data_transform, l_transform=label_transform)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True)

    if args.do_adaptation:
        target_dataset = StandardizedTiffData('Target', args.target_dataset, do_3D=args.do_3D, from_3D=args.target_from_3D,
                                              d_transform=data_transform, l_transform=label_transform)
        target_dl = DataLoader(target_dataset, batch_size=1, shuffle=True)
        train_losses, val_losses = adapt_network(model, train_dl, target_dl, val_dl, sas_class_loss, class_loss, flow_loss,
                                                 median_diams, optimizer=optimizer, device=device, n_epochs=args.epochs,
                                                 patch_per_batch=args.patches_per_batch)
    else:  # Train network without adaptation
        train_losses, val_losses = train_network(model, train_dl, val_dl, class_loss, flow_loss, median_diams,
                                                 optimizer=optimizer, device=device, n_epochs=args.epochs,
                                                 patch_per_batch=args.patches_per_batch)
    save(model.state_dict(), os.path.join(args.results_dir, 'trained_model.pt'))

    plt.figure()
    x_range = np.arange(1, len(train_losses)+1)
    plt.plot(x_range, train_losses)
    plt.plot(x_range, val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Combined Losses')
    plt.title('Training and Validation Losses')
    plt.legend(['Training Losses', 'Validation Losses'])
    plt.savefig(os.path.join(args.results_dir, 'Training-Validation Losses'))
    plt.show()

if not args.train_only:
    test_dataset = StandardizedTiffData('Test', args.test_dataset, do_3D=args.do_3D, from_3D=args.test_from_3D,
                                        d_transform=data_transform, l_transform=label_transform)

    eval_dl = DataLoader(test_dataset, batch_size=1, shuffle=True)

    masks, labels, label_list = eval_network(model, eval_dl, device, patch_per_batch=args.patches_per_batch,
                                             default_meds=median_diams, gc_model=gen_cellpose, sz_model=gen_size_model,
                                             refine=args.refine_prediction)

    for i in range(len(masks)):
        masks[i] = masks[i].astype('int16')  # Can change back to int32 if necessary
        with open(os.path.join(args.results_dir, label_list[i] + '_predicted_labels.pkl'), 'wb') as pl:
            pickle.dump(masks[i], pl)
        tifffile.imwrite(os.path.join(args.results_dir, 'tiff_results', label_list[i] + '.tif'), masks[i].astype('int32'))

    with open(os.path.join(args.results_dir, 'settings.txt'), 'w') as txt:
        txt.write('Adaptation: {}\n'.format(args.do_adaptation))
        # if do_adaptation:
        #     txt.write('Gamma: {}; Margin: {}'.format())
        txt.write('Learning rate: {}; Momentum: {}\n'.format(args.learning_rate, args.momentum))
        txt.write('Epochs: {}; Batch size: {}\n'.format(args.epochs, args.patches_per_batch))
        txt.write('GPUs: {}'.format(num_workers))

    # AP Calculation
    if args.calculate_ap:
        tau = np.arange(0.01, 1.01, 0.01)
        ap_info = average_precision(labels, masks, threshold=tau)
        ap_per_im = ap_info[0]
        ap_overall = np.average(ap_per_im, axis=0)

        plt.figure()
        plt.plot(tau, ap_overall)
        plt.title('Average Precision for Cellpose on {} Dataset'.format(args.dataset_name))
        plt.xlabel(r'IoU Matching Threshold $\tau$')
        plt.ylabel('Average Precision')
        plt.savefig(os.path.join(args.results_dir, 'AP Results'))
        plt.show()
        with open(os.path.join(args.results_dir, '{}_AP_Results.pkl'.format(args.dataset_name)), 'wb') as apr:
            pickle.dump((tau, ap_overall), apr)
