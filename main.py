import torchvision

from torch.utils.data import DataLoader
from torch import nn, device
from torch.cuda import is_available, device_count
from torch.optim import SGD
import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from transforms import Reshape, Normalize1stTo99th, ResizeImage
from loaddata import StandardizedTiffData
from Cellpose_2D_PyTorch import UpdatedCellpose
from train_eval import train_network, eval_network

results_dir = '/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim/results/results3'
assert not os.path.exists(results_dir), 'Results folder currently exists; please specify new location to save results.'
os.mkdir(results_dir)
os.mkdir(os.path.join(results_dir, 'tiff_results'))
learning_rate = 0.001
momentum = 0.9
batch_size = 8
n_epochs = 10
# num_workers = device_count()
# num_workers = 2
device = device('cuda' if is_available() else 'cpu')

width, height = 696, 520

data_transform = torchvision.transforms.Compose([
    Reshape(),
    Normalize1stTo99th(),
    ResizeImage(width, height, cv2.INTER_LINEAR),
    # torchvision.transforms.ToTensor()
])
label_transform = torchvision.transforms.Compose([
    Reshape(),
    ResizeImage(width, height, cv2.INTER_NEAREST)
])

train_dataset = StandardizedTiffData('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim',
                                     do_3D=False, d_transform=data_transform, l_transform=label_transform,
                                     # augmentations=augmentations
                                     )
train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)  # num_workers=num_workers

# val_dataset = StandardizedTiffData('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 2-dim',
#                                      do_3D=False, d_transform=data_transform, l_transform=label_transform)
val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # num_workers=num_workers

model = UpdatedCellpose(channels=1, flow_crit=nn.MSELoss(reduction='mean'),
                        class_crit=nn.BCEWithLogitsLoss(reduction='mean')).to(device)
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# model = nn.DataParallel(model)

train_losses = train_network(model, train_dl, optimizer=optimizer, device=device, n_epochs=n_epochs)

plt.figure()
x_range = np.arange(1, len(train_losses)+1)
plt.plot(x_range, train_losses)
plt.show()

masks, label_list = eval_network(model, val_dataloader, device)

for i in range(len(masks)):
    # masks[i] = np.reshape(np.transpose(masks[i].numpy(), (0, 2, 1)), (masks[i].shape[2], masks[i].shape[1])).astype('int32')
    mask = masks[i].numpy()
    with open(os.path.join(results_dir, label_list[i] + '_predicted_labels.pkl'), 'wb') as pl:
        pickle.dump(mask.astype('int32'), pl)
    tifffile.imwrite(os.path.join(results_dir, 'tiff_results', label_list[i] + '.tif'), mask)
