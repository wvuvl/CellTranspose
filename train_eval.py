from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, as_tensor
from time import time
from tqdm import tqdm
import cv2
import numpy as np
import math
from misc_utils import elapsed_time
from transforms import LabelsToFlows, FollowFlows, resize_from_labels, predict_and_resize, random_horizontal_flip,\
    random_rotate, generate_patches, recombine_patches

import matplotlib.pyplot as plt


def train_network(model: nn.Module, data_loader: DataLoader, class_loss, flow_loss,
                  default_meds, optimizer, device, n_epochs, patch_per_batch):
    print('Beginning network training.\n')
    train_losses = []

    for e in range(1, n_epochs + 1):
        model.train()
        start_train = time()
        for (sample_data, sample_labels, _) in tqdm(data_loader, desc='Epoch {}/{}:'.format(e, n_epochs)):
            sample_data, sample_labels = resize_from_labels(sample_data, sample_labels, default_meds)
            sample_data, sample_labels = random_horizontal_flip(sample_data, sample_labels)
            sample_data, sample_labels = random_rotate(sample_data, sample_labels)
            sample_labels = as_tensor([LabelsToFlows()(sample_labels[i].numpy()) for i in range(len(sample_labels))])
            sample_data, sample_labels = generate_patches(sample_data, sample_labels)

            # Passes in only a subset of patches (effectively batch size) to GPU at a time
            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_patch_data = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                sample_patch_labels = sample_labels[patch_ind:patch_ind + patch_per_batch].float().to(device)
                optimizer.zero_grad()
                output = model(sample_patch_data)
                grad_loss = flow_loss(output, sample_patch_labels)
                mask_loss = class_loss(output, sample_patch_labels)
                train_loss = grad_loss + mask_loss
                train_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
        print('Train time: {}'.format(elapsed_time(time() - start_train)))
    return train_losses


def adapt_network(model: nn.Module, source_data_loader: DataLoader, target_data_loader: DataLoader, sas_class_loss,
                  flow_loss, default_meds, optimizer, device, n_epochs, patch_per_batch):
    print('Beginning domain adaptation.\n')

    # Load in target data initially (Note: this assumes small target data size)
    target_data = tensor([])
    target_labels = tensor([])
    for (target_sample_data, target_sample_labels, _) in target_data_loader:
        target_sample_data, target_sample_labels = resize_from_labels(target_sample_data, target_sample_labels, default_meds)
        target_sample_data, target_sample_labels = random_horizontal_flip(target_sample_data, target_sample_labels)
        target_sample_data, target_sample_labels = random_rotate(target_sample_data, target_sample_labels)
        target_sample_labels = as_tensor([LabelsToFlows()(target_sample_labels[i].numpy()) for i in range(len(target_sample_labels))])
        target_sample_data, target_sample_labels = generate_patches(target_sample_data, target_sample_labels)
        target_data = cat((target_data, target_sample_data))
        target_labels = cat((target_labels, target_sample_labels))
    batched_target_data = target_data
    batched_target_labels = target_labels
    for _ in range(1, math.ceil(patch_per_batch/len(target_data))):
        batched_target_data = cat((batched_target_data, target_data))
        batched_target_labels = cat((batched_target_labels, target_labels))

    # Assume # of target samples << # of source samples
    train_losses = []
    start_train = time()
    for e in range(1, n_epochs + 1):
        model.train()
        for (source_sample_data, source_sample_labels, _) in tqdm(source_data_loader, desc='Epoch {}/{}:'.format(e, n_epochs)):
            optimizer.zero_grad()
            source_sample_data, source_sample_labels = resize_from_labels(source_sample_data, source_sample_labels, default_meds)
            source_sample_data, source_sample_labels = random_horizontal_flip(source_sample_data, source_sample_labels)
            source_sample_data, source_sample_labels = random_rotate(source_sample_data, source_sample_labels)
            source_sample_labels = as_tensor(
                [LabelsToFlows()(source_sample_labels[i].numpy()) for i in range(len(source_sample_labels))])
            source_sample_data, source_sample_labels = generate_patches(source_sample_data, source_sample_labels)

            # Passes in only a subset of patches (effectively batch size) to GPU at a time
            for patch_ind in range(0, len(source_sample_data), patch_per_batch):
                source_sample_patch_data = source_sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                source_sample_patch_labels = source_sample_labels[patch_ind:patch_ind + patch_per_batch].float().to(device)
                source_output = model(source_sample_patch_data)

                # Shuffle the target data
                t_len = len(source_sample_patch_data)
                target_sample_patch_data = batched_target_data
                target_sample_patch_labels = batched_target_labels
                shuffled_inds = np.array(range(t_len))
                np.random.shuffle(shuffled_inds)
                target_sample_patch_data = target_sample_patch_data[:t_len]
                target_sample_patch_data[np.array(range(t_len))] = target_sample_patch_data[shuffled_inds]
                target_sample_patch_data = target_sample_patch_data.float().to(device)
                target_sample_patch_labels = target_sample_patch_labels[:t_len]
                target_sample_patch_labels[np.array(range(t_len))] = target_sample_patch_labels[shuffled_inds]
                target_sample_patch_labels = target_sample_patch_labels.float().to(device)

                target_output = model(target_sample_patch_data)
                source_grad_loss = flow_loss(source_output, source_sample_patch_labels)
                adaptation_class_loss = sas_class_loss(source_output[:, 0], source_sample_patch_labels[:, 0],
                                                       target_output[:, 0], target_sample_patch_labels[:, 0],
                                                       margin=1, gamma=0.1)

                train_loss = source_grad_loss + adaptation_class_loss
                train_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

    print('Train time: {}'.format(elapsed_time(time() - start_train)))
    return train_losses


def eval_network(model: nn.Module, data_loader: DataLoader, device, patch_per_batch, default_meds, gc_model, sz_model):

    model.eval()
    start_eval = time()
    with no_grad():
        masks = []
        label_list = []
        for step, (sample_data, sample_labels, label_files) in enumerate(data_loader):
            original_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, sample_labels = predict_and_resize(sample_data.float().to(device), sample_labels.to(device),
                                                            default_meds, gc_model, sz_model, refine=False)
            resized_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, sample_labels = generate_patches(sample_data, sample_labels, eval=True)
            predictions = tensor([]).to(device)

            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_patch_data = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                ff = FollowFlows(niter=100, interp=True, use_gpu=True, cellprob_threshold=0.0, flow_threshold=1.0)
                p = model(sample_patch_data)
                predictions = cat((predictions, p))

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

            predictions = recombine_patches(predictions, resized_dims)
            sample_mask = ff(predictions)
            sample_mask = np.transpose(sample_mask.numpy(), (1, 2, 0))
            sample_mask = cv2.resize(sample_mask, (original_dims[1], original_dims[0]), interpolation=cv2.INTER_NEAREST)
            masks.append(sample_mask)
            for i in range(len(label_files)):
                label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

    print('Total time to evaluate: {}'.format(elapsed_time(time() - start_eval)))
    return masks, label_list
