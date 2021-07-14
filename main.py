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

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', help='Folder in which to save experiment results.')
parser.add_argument('--do-adaptation', help='Whether to perform domain adaptation or standard training.', action='store_true')
parser.add_argument('--do-3D', help='Whether or not to use 3D-Cellpose (Must use 3D volumes).', action='store_true')
parser.add_argument('--train-dataset', help='The directory containing data to be used for training.')
parser.add_argument('--train-from-3D', help='Whether the input training (source) data is 3D: assumes 2D if set to False.', action='store_true')
parser.add_argument('--test-dataset', help='The directory containing data to be used for testing.')
parser.add_argument('--test-from-3D', help='Whether the input test data is 3D: assumes 2D if set to False.', action='store_true')
parser.add_argument('--target-dataset', help='The directory containing target data to be used for domain adaptation.'
                                           'Note: if do-adaptation is set to False, this parameter will be ignored.')
parser.add_argument('--target-from-3D', help='Whether the input target data is 3D: assumes 2D if set to False.', action='store_true')
parser.add_argument('--cellpose-model', help='The generalized cellpose model to use for diameter estimation.')
parser.add_argument('--size-model', help='The generalized size model to use for diameter estimation.')
args = parser.parse_args()

assert not os.path.exists(args.results_dir), 'Results folder currently exists; please specify new location to save results.'
os.mkdir(args.results_dir)
os.mkdir(os.path.join(args.results_dir, 'tiff_results'))
learning_rate = 1e-6
momentum = 0.5
# momentum = 0.9
patches_per_batch = 128
n_epochs = 10
num_workers = device_count()
device = device('cuda' if is_available() else 'cpu')
empty_cache()

# Default median diameter to resize cells to
median_diams = (24, 24)
gen_cellpose = UpdatedCellpose(channels=1).to(device)
gen_cellpose.load_state_dict(load(args.cellpose_model))

gen_size_model = SizeModel().to(device)
gen_size_model.load_state_dict(load(args.size_model))

data_transform = torchvision.transforms.Compose([
    Reformat(),
    Normalize1stTo99th()
])
label_transform = torchvision.transforms.Compose([
    Reformat()
])

print('Training dataset:')
train_dataset = StandardizedTiffData(args.train_dataset,
                                     do_3D=args.do_3D, from_3D=args.train_from_3D, d_transform=data_transform, l_transform=label_transform)
train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers)  # num_workers=num_workers

if args.do_adaptation:
    print('Target dataset:')
    target_dataset = StandardizedTiffData(args.target_dataset,
                                          do_3D=args.do_3D, from_3D=args.target_from_3D, d_transform=data_transform, l_transform=label_transform)
    target_dl = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=num_workers)  # num_workers=num_workers

model = UpdatedCellpose(channels=1, device=device)
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
model = nn.DataParallel(model)
model.to(device)

if args.do_adaptation:
    train_losses = adapt_network(model, train_dl, target_dl, sas_class_loss, flow_loss, median_diams, optimizer=optimizer,
                                 device=device, n_epochs=n_epochs, patch_per_batch=patches_per_batch)
else:  # Train network without adaptation
    train_losses = train_network(model, train_dl, class_loss, flow_loss, median_diams, optimizer=optimizer,
                                 device=device, n_epochs=n_epochs, patch_per_batch=patches_per_batch)
save(model.state_dict(), os.path.join(args.results_dir, 'trained_model.pt'))

plt.figure()
x_range = np.arange(1, len(train_losses)+1)
plt.plot(x_range, train_losses)
plt.title('Training Loss')
plt.savefig(os.path.join(args.results_dir, 'Training Loss'))
plt.show()

print('Test dataset:')
test_dataset = StandardizedTiffData(args.test_dataset, do_3D=args.do_3D, from_3D=args.test_from_3D,
                                    d_transform=data_transform, l_transform=label_transform)

val_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)  # num_workers=num_workers

masks, label_list = eval_network(model, val_dataloader, device, patch_per_batch=patches_per_batch,
                                 default_meds=median_diams, gc_model=gen_cellpose, sz_model=gen_size_model)

for i in range(len(masks)):
    with open(os.path.join(args.results_dir, label_list[i] + '_predicted_labels.pkl'), 'wb') as pl:
        pickle.dump(masks[i].astype('int32'), pl)
    tifffile.imwrite(os.path.join(args.results_dir, 'tiff_results', label_list[i] + '.tif'), masks[i].astype('int32'))

with open(os.path.join(args.results_dir, 'settings.txt'), 'w') as txt:
    txt.write('Adaptation: {}\n'.format(args.do_adaptation))
    # if do_adaptation:
    #     txt.write('Gamma: {}; Margin: {}'.format())
    txt.write('Learning rate: {}; Momentum: {}\n'.format(learning_rate, momentum))
    txt.write('Epochs: {}; Batch size: {}'.format(n_epochs, patches_per_batch))
