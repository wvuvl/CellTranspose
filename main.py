import torchvision

from torch.utils.data import DataLoader
from torch import nn, device, load, save
from torch.cuda import is_available, device_count, empty_cache
from torch.optim import SGD
import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from transforms import Reformat, Normalize1stTo99th
    # , ResizeImage
from loaddata import StandardizedTiffData
from Cellpose_2D_PyTorch import UpdatedCellpose, SizeModel
from train_eval import train_network, adapt_network, eval_network

do_adaptation = False

# results_dir = '/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim/results/results51'
results_dir = '/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim/results/results106'
assert not os.path.exists(results_dir), 'Results folder currently exists; please specify new location to save results.'
os.mkdir(results_dir)
os.mkdir(os.path.join(results_dir, 'tiff_results'))
learning_rate = 1e-5
momentum = 0.5
# momentum = 0.9
# batch_size = 1
patches_per_batch = 128
n_epochs = 2
# num_workers = device_count()
# num_workers = 2
device = device('cuda' if is_available() else 'cpu')
empty_cache()

# Determine resize factor such that image is resized to where median diameter is 32
default_meds = (24, 24)
# median_diam_x, median_diam_y = 22, 22
# median_diam_x, median_diam_y = 32, 32
# resize_factor_x, resize_factor_y = default_x / median_diam_x, default_y / median_diam_y
gen_cellpose = UpdatedCellpose(channels=1).to(device)
gen_cellpose.load_state_dict(load('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/'
                                  '2D Toy Dataset - 3-dim/results/results102/trained_model.pt'))

gen_size_model = SizeModel().to(device)
gen_size_model.load_state_dict(load('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim/results/'
                                'size_model_results/results3/size_model.pt'))

data_transform = torchvision.transforms.Compose([
    Reformat(),
    Normalize1stTo99th(),
    # ResizeImage(resize_factor_x, resize_factor_y, cv2.INTER_LINEAR),
    # torchvision.transforms.ToTensor()
])
label_transform = torchvision.transforms.Compose([
    Reformat(),
    # ResizeImage(resize_factor_x, resize_factor_y, cv2.INTER_NEAREST)
])

print('Training dataset:')
train_dataset = StandardizedTiffData('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim',
                                     train=True, do_3D=False, d_transform=data_transform, l_transform=label_transform)
                                     # default_meds=default_meds)
print('Validation dataset:')
val_dataset = StandardizedTiffData('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim',
                                   train=False, do_3D=False, d_transform=data_transform, l_transform=label_transform)
                                   # default_meds=default_meds)
train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)  # num_workers=num_workers

# if do_adaptation:
#     target_dataset = StandardizedTiffData('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim',
#                                           do_3D=False, d_transform=data_transform, l_transform=label_transform,
#                                           # augmentations=augmentations
#                                           )
#     target_dl = DataLoader(target_dataset, batch_size=1, shuffle=True)

# val_dataset = StandardizedTiffData('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim',
#                                      do_3D=False, d_transform=data_transform, l_transform=label_transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)  # num_workers=num_workers

model = UpdatedCellpose(channels=1, class_crit=nn.BCEWithLogitsLoss(reduction='mean'),
                        flow_crit=nn.MSELoss(reduction='mean')).to(device)
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# model = nn.DataParallel(model)

if do_adaptation:
    train_losses = adapt_network(model, train_dl, train_dl, default_meds, optimizer=optimizer, device=device, n_epochs=n_epochs,
                                 patch_per_batch=patches_per_batch)
else:  # Train network without adaptation
    train_losses = train_network(model, train_dl, default_meds, optimizer=optimizer, device=device, n_epochs=n_epochs,
                                 patch_per_batch=patches_per_batch)
save(model.state_dict(), os.path.join(results_dir, 'trained_model.pt'))

plt.figure()
x_range = np.arange(1, len(train_losses)+1)
plt.plot(x_range, train_losses)
plt.savefig(os.path.join(results_dir, 'Training Loss'))
plt.show()

masks, label_list = eval_network(model, val_dataloader, device, patch_per_batch=patches_per_batch,
                                 default_meds=default_meds, gc_model=gen_cellpose, sz_model=gen_size_model)
# TODO: Resize masks here

for i in range(len(masks)):
    # masks[i] = np.reshape(np.transpose(masks[i].numpy(), (0, 2, 1)), (masks[i].shape[2], masks[i].shape[1])).astype('int32')
    # mask = masks[i].numpy()
    with open(os.path.join(results_dir, label_list[i] + '_predicted_labels.pkl'), 'wb') as pl:
        pickle.dump(masks[i].astype('int32'), pl)
    tifffile.imwrite(os.path.join(results_dir, 'tiff_results', label_list[i] + '.tif'), masks[i])

with open(os.path.join(results_dir, 'settings.txt'), 'w') as txt:
    txt.write('Adaptation: {}\n'.format(do_adaptation))
    # if do_adaptation:
        # txt.write('Gamma: {}; Margin: {}'.format())
    txt.write('Learning rate: {}; Momentum: {}\n'.format(learning_rate, momentum))
    txt.write('Epochs: {}; Batch size: {}'.format(n_epochs, patches_per_batch))
