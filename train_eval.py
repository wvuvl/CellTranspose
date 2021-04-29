from torch.utils.data import DataLoader
from torch import nn, no_grad, as_tensor
from time import time

from misc_utils import elapsed_time
from transforms import LabelsToFlows, FollowFlows, random_horizontal_flip, random_rotate
from Cellpose_2D_PyTorch import SASClassLoss

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
            flow_loss = model.flow_loss(output, batch_labels)
            mask_loss = model.flow_loss(output, batch_labels)
            train_loss = flow_loss + mask_loss
            train_losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
        print('Train time: {}'.format(elapsed_time(time() - start_train)))

    return train_losses


def adapt_network(model: nn.Module, source_data_loader: DataLoader, target_data_loader: DataLoader, optimizer, device, n_epochs):
    print('Beginning domain adaptation.\n')

    train_losses = []

    len_data = min(len(source_data_loader), len(target_data_loader))
    for e in range(1, n_epochs + 1):
        print('Epoch {}/{}:'.format(e, n_epochs))
        source_iter = iter(source_data_loader)
        target_iter = iter(target_data_loader)
        model.train()
        start_train = time()
        for batch in range(len_data):
            optimizer.zero_grad()

            source_batch = source_iter.next()
            source_batch_data, source_batch_labels, _ = source_batch
            source_batch_data, source_batch_labels = random_horizontal_flip(source_batch_data, source_batch_labels)
            source_batch_data, source_batch_labels = random_rotate(source_batch_data, source_batch_labels)
            source_batch_labels = as_tensor(
                [LabelsToFlows()(source_batch_labels[i].numpy()) for i in range(len(source_batch_labels))])
            source_batch_data = source_batch_data.float().to(device)
            source_batch_labels = source_batch_labels.float().to(device)

            source_output = model(source_batch_data)

            target_batch = target_iter.next()
            target_batch_data, target_batch_labels, _ = target_batch
            target_batch_data, target_batch_labels = random_horizontal_flip(target_batch_data, target_batch_labels)
            target_batch_data, target_batch_labels = random_rotate(target_batch_data, target_batch_labels)
            target_batch_labels = as_tensor(
                [LabelsToFlows()(target_batch_labels[i].numpy()) for i in range(len(target_batch_labels))])
            target_batch_data = target_batch_data.float().to(device)
            target_batch_labels = target_batch_labels.float().to(device)

            target_output = model(target_batch_data)

            source_flow_loss = model.flow_loss(source_output, source_batch_labels)
            adaptation_class_loss = model.sas_class_loss(source_output[:, 0], source_batch_labels[:, 0],
                                                         target_output[:, 0], target_batch_labels[:, 0],
                                                         margin=1, gamma=0.1)

            train_loss = source_flow_loss + adaptation_class_loss
            train_losses.append(train_loss.item())
            train_loss.backward()
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
