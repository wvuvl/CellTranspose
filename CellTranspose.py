import argparse
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torch import nn, device, load, save, jit
from torch.cuda import is_available, empty_cache
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import time

from transforms import Resize
from loaddata import TrainCellTransposeData, EvalCellTransposeData, EvalCellTransposeData3D
from network import CellTranspose, ClassLoss, FlowLoss, SASMaskLoss, ContrastiveFlowLoss
from train_eval import train_network, adapt_network, eval_network, eval_network_3D
from calculate_results import produce_logfile, plot_loss, save_pred

parser = argparse.ArgumentParser()

# Model hyperparameters
parser.add_argument('--n-chan', type=int,
                    help='Maximum number of channels in input images (i.e. 2 for cytoplasm + nuclei images).')
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--eval-batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--step-gamma', type=float, default=0.1)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--gamma-1', type=float, default=0.1)
parser.add_argument('--gamma-2', type=float, default=2)
parser.add_argument('--n_thresh', type=float, default=0.05)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--median-diams', type=int,
                    help='Median diameter size with which to resize images to. Note: If using pretrained model, ensure'
                         ' that this variable remains the same as the given model.', default=30)
parser.add_argument('--patch-size', type=int, help='Size of image patches with which to tile.', default=112)
parser.add_argument('--min-overlap', type=int, help='Amount of overlap to use for tiling during testing.', default=84)

# Control
parser.add_argument('--dataset-name', help='Name of dataset to use for reporting results (omit the word "Dataset").')
parser.add_argument('--results-dir', help='Folder in which to save experiment results.')
parser.add_argument('--pretrained-model', help='Location of pretrained model to load in. Default: None')
parser.add_argument('--train-only', help='Only perform training, no evaluation (mutually exclusive with "eval-only").',
                    action='store_true')
parser.add_argument('--eval-only', help='Only perform evaluation, no training (mutually exclusive with "train-only").',
                    action='store_true')
parser.add_argument('--do-adaptation', help='Whether to perform domain adaptation or standard training.',
                    action='store_true')
parser.add_argument('--no-adaptation-loss', help='Train directly using standard loss on target samples (for'
                                                 ' experimentation use, can usually ignore this)', action='store_true')
parser.add_argument('--save-dataset', help='Name of directory to save training dataset to:'
                                           ' if None, will not save dataset.')
parser.add_argument('--load-from-torch', help='If true, assumes dataset is being loaded from torch files, with no'
                                              ' preprocessing required.', action='store_true')
parser.add_argument('--process-each-epoch', help='If true, assumes processing occurs every epoch.', action='store_true')
parser.add_argument('--load-train-from-npy', help='If provided, assumes training data is being loaded from npy files.')

# Training data
parser.add_argument('--train-dataset', help='The directory(s) containing (source) data to be used for training.',
                    nargs='+')
parser.add_argument('--train-from-3D', help='Whether the input training source data is 3D: assumes 2D if set to False.',
                    action='store_true')

# Target data
parser.add_argument('--target-dataset',
                    help='The directory containing target data to be used for domain adaptation. Note: if do-adaptation'
                         ' is set to False, this parameter will be ignored.', nargs='+')
parser.add_argument('--target-from-3D', help='Whether the input target data is 3D: assumes 2D if set to False.',
                    action='store_true')
parser.add_argument('--target-flows', help='The directory(s) containing pre-calculated flows. If left empty,'
                                           ' flows will be calculated manually.', nargs='+')

# Validation data
parser.add_argument('--val-dataset', help='The directory(s) containing data to be used for validation.', nargs='+')

# Test data
parser.add_argument('--test-dataset', help='The directory(s) containing data to be used for testing.', nargs='+')
parser.add_argument('--test-from-3D', help='Whether the input test data is 3D: assumes 2D if set to False.',
                    action='store_true')

# Note: do-3D not currently implemented. Can be used for further development with volumetric approach
parser.add_argument('--do-3D', help='Whether or not to use CellTranspose3D (Must use 3D volumes).', action='store_true')

# Calculate results - Note: currently only implemented for 2D, must perform AP calculation manually for 3D
parser.add_argument('--calculate-ap', help='Calculates average precision if labeled data is provided.',
                    action='store_true')

args = parser.parse_args()

print(args.results_dir)
assert not os.path.exists(args.results_dir),\
    'Results folder {} currently exists; please specify new location to save results.'.format(args.results_dir)
os.makedirs(args.results_dir)
os.makedirs(os.path.join(args.results_dir, 'tiff_results'))
os.makedirs(os.path.join(args.results_dir, 'raw_predictions_tiffs'))
assert not (args.train_only and args.eval_only), 'Cannot pass in "train-only" and "eval-only" arguments simultaneously.'
device = device('cuda' if is_available() else 'cpu')
empty_cache()

# TODO pass in assertion for target labels necessary for evaluation
args.median_diams = (args.median_diams, args.median_diams)
args.patch_size = (args.patch_size, args.patch_size)
args.min_overlap = (args.min_overlap, args.min_overlap)
ttt = None
tte = None
train_losses = None
if args.target_dataset is not None:
    target_dataset = TrainCellTransposeData('Target', args.target_dataset, args.n_chan, pf_dirs=args.target_flows,
                                            do_3D=args.do_3D, from_3D=args.target_from_3D,
                                            crop_size=args.patch_size, has_flows=False, batch_size=args.batch_size,
                                            resize=Resize(args.median_diams))
    rs = RandomSampler(target_dataset, replacement=False)
    bs = BatchSampler(rs, args.batch_size, True)
    target_dl = DataLoader(target_dataset, batch_sampler=bs)
else:
    target_dataset = None

model = CellTranspose(channels=args.n_chan, device=device)
model = nn.DataParallel(model)
model.to(device)
if args.pretrained_model is not None:
    model.load_state_dict(load(args.pretrained_model, map_location=device))

if not args.eval_only:
    class_loss = ClassLoss(nn.BCEWithLogitsLoss(reduction='mean'))
    flow_loss = FlowLoss(nn.MSELoss(reduction='mean'))
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.load_from_torch:
        print('Loading Saved Training Dataset... ', end='')
        train_dataset = load(args.train_dataset[0])
        print('Done.')
    else:
        if not args.do_adaptation:
            args.process_each_epoch = True
        train_dataset = TrainCellTransposeData('Training', args.train_dataset, args.n_chan, do_3D=args.do_3D,
                                               from_3D=args.train_from_3D, crop_size=args.patch_size, has_flows=False,
                                               batch_size=args.batch_size, resize=Resize(args.median_diams),
                                               preprocessed_data=args.load_train_from_npy,
                                               proc_every_epoch=args.process_each_epoch, result_dir=args.results_dir)
    
    if args.save_dataset:
        print('Saving Training Dataset... ', end='')
        save(train_dataset, args.save_dataset)
        print('Saved.')

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.val_dataset is not None:
        val_dataset = EvalCellTransposeData('Validation', args.val_dataset, args.n_chan, do_3D=args.do_3D,
                                            resize=Resize(args.median_diams))
        val_dataset.pre_generate_validation_patches(patch_size=args.patch_size, min_overlap=args.min_overlap)
        val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_dl = None
        print('No validation data given --> skipping validation.')

    if args.do_adaptation:
        sas_class_loss = SASMaskLoss(nn.BCEWithLogitsLoss(reduction='mean'))
        c_flow_loss = ContrastiveFlowLoss(nn.MSELoss(reduction='mean'))

        start_train = time.time()
        scheduler = StepLR(optimizer, step_size=1, gamma=args.step_gamma)
        train_losses, val_losses = adapt_network(model, train_dl, target_dl, val_dl, sas_class_loss, c_flow_loss,
                                                 class_loss, flow_loss, train_direct=args.no_adaptation_loss,
                                                 optimizer=optimizer, scheduler=scheduler, device=device,
                                                 n_epochs=args.epochs, k=args.k, gamma_1=args.gamma_1,
                                                 gamma_2=args.gamma_2, n_thresh=args.n_thresh,
                                                 temperature=args.temperature)
    else:
        start_train = time.time()
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate/100, last_epoch=-1)
        train_losses, val_losses = train_network(model, train_dl, val_dl, class_loss, flow_loss, optimizer=optimizer,
                                                 scheduler=scheduler, device=device, n_epochs=args.epochs)
    # compiled_model = jit.script(model)
    # jit.save(compiled_model, os.path.join(args.results_dir, 'trained_model.pt'))
    save(model.state_dict(), os.path.join(args.results_dir, 'trained_model.pt'))
    end_train = time.time()
    ttt = time.strftime("%H:%M:%S", time.gmtime(end_train - start_train))
    print('Time to train: {}'.format(ttt))
    plot_loss(train_losses, args.results_dir, val_dl=val_dl, val_losses=val_losses)
    
if not args.train_only:
    start_eval = time.time()
    if target_dataset is not None:
        target_labels = target_dataset.target_label_samples
    else:
        target_labels = None
    if not args.test_from_3D:
        test_dataset = EvalCellTransposeData('Test', args.test_dataset, args.n_chan, do_3D=args.do_3D,
                                             from_3D=args.test_from_3D, evaluate=True,
                                             resize=Resize(args.median_diams, target_labels=target_labels))
        eval_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        masks, prediction_list, data_list = eval_network(model, eval_dl, device, patch_per_batch=args.eval_batch_size,
                                                          patch_size=args.patch_size, min_overlap=args.min_overlap)
        save_pred(masks, test_dataset, prediction_list, data_list, args.results_dir, args.dataset_name, args.calculate_ap)
    else:
        test_dataset_3D = EvalCellTransposeData3D('3D_test', args.test_dataset, args.n_chan, do_3D=args.do_3D,
                                                  from_3D=args.test_from_3D, evaluate=True,
                                                  resize=Resize(args.median_diams, target_labels=target_labels))
        eval_dl_3D = DataLoader(test_dataset_3D, batch_size=1, shuffle=False)
        eval_network_3D(model, eval_dl_3D, device, patch_per_batch=args.eval_batch_size,
                        patch_size=args.patch_size, min_overlap=args.min_overlap, results_dir=args.results_dir)
    end_eval = time.time()
    tte = time.strftime("%H:%M:%S", time.gmtime(end_eval - start_eval))
    print('Time to evaluate: {}'.format(tte))

print(args.results_dir)
produce_logfile(args, len(train_losses) if train_losses is not None else None, ttt, tte)
