from torch.utils.data import DataLoader
from torch import nn, device, no_grad, as_tensor
from torch.optim import SGD
from time import time
import os
import pickle

from misc_utils import elapsed_time
from cellpose_src import dynamics
from transforms import LabelsToFlows, FollowFlows, random_horizontal_flip, random_rotate

import matplotlib.pyplot as plt


def train_network(model: nn.Module, data_loader: DataLoader, optimizer, device, n_epochs):
    print('Beginning network training.\n')

    train_losses = []

    for e in range(1, n_epochs + 1):
        print('Epoch {}/{}:'.format(e, n_epochs))
        model.train()
        start_train = time()
        for step, (batch_data, batch_labels, _) in enumerate(data_loader):
            batch_data, batch_labels = random_horizontal_flip(batch_data, batch_labels)
            batch_data, batch_labels = random_rotate(batch_data, batch_labels)
            batch_labels = as_tensor([LabelsToFlows()(batch_labels[i].numpy()) for i in range(len(batch_labels))])
            batch_data = batch_data.float().to(device)
            batch_labels = batch_labels.float().to(device)

            optimizer.zero_grad()
            output = model(batch_data)
            loss = model.loss_fn(output, batch_labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Train time: {}'.format(elapsed_time(time() - start_train)))

    return train_losses

def eval_network(model: nn.Module, data_loader: DataLoader, device):

    model.eval()
    start_eval = time()
    with no_grad():
        masks = []
        label_list = []
        for step, (batch_data, batch_labels, label_files) in enumerate(data_loader):
            batch_data = batch_data.float().to(device)
            batch_labels = batch_labels.to(device)
            ff = FollowFlows(niter=200, interp=True, use_gpu=True, cellprob_threshold=0.0, flow_threshold=1.0)
            # Data already rescaled by data_loader transforms
            predictions = model(batch_data)

            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(predictions.cpu()[0][0])
            # plt.colorbar()
            # plt.subplot(1, 3, 2)
            # plt.imshow(predictions.cpu()[0][1])
            # plt.colorbar()
            # plt.subplot(1, 3, 3)
            # plt.imshow(predictions.cpu()[0][2])
            # plt.colorbar()
            # plt.tight_layout()
            # plt.show()
            # plt.figure()
            # plt.imshow(batch_labels.cpu()[0])
            # plt.show()

            batch_masks = ff(predictions)
            masks.append(batch_masks)

            for i in range(len(label_files)):
                label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

            # Resize data back here
    print('Total time to evaluate: {}'.format(elapsed_time(time() - start_eval)))
    return masks, label_list
