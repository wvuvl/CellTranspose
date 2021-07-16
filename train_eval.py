from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, as_tensor
from time import time
from tqdm import tqdm
import cv2
import numpy as np
from statistics import mean
import math
from misc_utils import elapsed_time
from transforms import LabelsToFlows, FollowFlows, resize_from_labels, predict_and_resize, random_horizontal_flip,\
    random_rotate, generate_patches, recombine_patches

import matplotlib.pyplot as plt


def preprocess_samples(data, labels, default_meds):
    data, labels = resize_from_labels(data, labels, default_meds)
    data, labels = random_horizontal_flip(data, labels)
    data, labels = random_rotate(data, labels)
    labels = as_tensor([LabelsToFlows()(labels[i].numpy()) for i in range(len(labels))])
    data, labels = generate_patches(data, labels)
    return data, labels


def train_network(model, train_dl, val_dl, class_loss, flow_loss,
                  default_meds, optimizer, device, n_epochs, patch_per_batch):
    train_losses = []
    val_losses = []
    print('Beginning network training.\n')
    start_train = time()
    for e in range(1, n_epochs + 1):
        train_epoch_losses = []
        model.train()
        for (sample_data, sample_labels, _) in tqdm(train_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs)):
            sample_data, sample_labels = preprocess_samples(sample_data, sample_labels, default_meds)
            # Passes in only a subset of patches (effectively batch size) to GPU at a time
            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_patch_data = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                sample_patch_labels = sample_labels[patch_ind:patch_ind + patch_per_batch].float().to(device)
                optimizer.zero_grad()
                output = model(sample_patch_data)
                grad_loss = flow_loss(output, sample_patch_labels)
                mask_loss = class_loss(output, sample_patch_labels)
                train_loss = grad_loss + mask_loss
                train_epoch_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

        train_losses.append(mean(train_epoch_losses))
        val_losses.append(validate_network(model, val_dl, flow_loss, class_loss, device, patch_per_batch, default_meds))

    print('Train time: {}'.format(elapsed_time(time() - start_train)))
    return train_losses, val_losses


def adapt_network(model: nn.Module, source_dl, target_dl, val_dl, sas_class_loss, class_loss, flow_loss,
                  default_meds, optimizer, device, n_epochs, patch_per_batch):
    train_losses = []
    val_losses = []
    print('Beginning domain adaptation.\n')

    # Load in target data initially (Note: this assumes small target data size)
    target_data = tensor([])
    target_labels = tensor([])
    for (target_sample_data, target_sample_labels, _) in target_dl:
        target_sample_data, target_sample_labels = preprocess_samples(target_sample_data, target_sample_labels, default_meds)
        target_data = cat((target_data, target_sample_data))
        target_labels = cat((target_labels, target_sample_labels))
    batched_target_data = target_data
    batched_target_labels = target_labels
    for _ in range(1, math.ceil(patch_per_batch/len(target_data))):
        batched_target_data = cat((batched_target_data, target_data))
        batched_target_labels = cat((batched_target_labels, target_labels))

    # Assume # of target samples << # of source samples
    start_train = time()
    for e in range(1, n_epochs + 1):
        model.train()
        train_epoch_losses = []
        for (source_sample_data, source_sample_labels, _) in tqdm(source_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs)):
            optimizer.zero_grad()
            source_sample_data, source_sample_labels = preprocess_samples(source_sample_data, source_sample_labels, default_meds)

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
                train_epoch_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

        train_losses.append(mean(train_epoch_losses))
        val_losses.append(validate_network(model, val_dl, flow_loss, class_loss, device, patch_per_batch, default_meds))

    print('Train time: {}'.format(elapsed_time(time() - start_train)))
    return train_losses, val_losses


def validate_network(model, data_loader, flow_loss, class_loss, device, patch_per_batch, default_meds):
    model.eval()
    val_epoch_losses = []
    with no_grad():
        for (val_sample_data, val_sample_labels, _) in tqdm(data_loader, desc='Performing validation'):
            val_sample_data, val_sample_labels = resize_from_labels(val_sample_data, val_sample_labels, default_meds)
            val_sample_data, val_sample_labels = generate_patches(val_sample_data, val_sample_labels, eval=True)
            val_sample_labels = as_tensor(
                [LabelsToFlows()(val_sample_labels[i].numpy()) for i in range(len(val_sample_labels))])
            for patch_ind in range(0, len(val_sample_data), patch_per_batch):
                val_sample_patch_data = val_sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                val_sample_patch_labels = val_sample_labels[patch_ind:patch_ind + patch_per_batch].float().to(device)
                output = model(val_sample_patch_data)
                grad_loss = flow_loss(output, val_sample_patch_labels).item()
                mask_loss = class_loss(output, val_sample_patch_labels).item()
                val_loss = grad_loss + mask_loss
                val_epoch_losses.append(val_loss)
    return mean(val_epoch_losses)


def eval_network(model: nn.Module, data_loader: DataLoader, device, patch_per_batch, default_meds, gc_model, sz_model):

    model.eval()
    start_eval = time()
    ff = FollowFlows(niter=100, interp=True, use_gpu=True, cellprob_threshold=0.0, flow_threshold=1.0)
    with no_grad():
        masks = []
        labels = []
        label_list = []
        for (sample_data, sample_labels, label_files) in data_loader:
            original_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, resized_sample_labels = predict_and_resize(sample_data.float().to(device), sample_labels.to(device),
                                                            default_meds, gc_model, sz_model, refine=False)
            resized_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, _ = generate_patches(sample_data, resized_sample_labels, eval=True)
            predictions = tensor([]).to(device)

            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_patch_data = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                p = model(sample_patch_data)
                predictions = cat((predictions, p))

            predictions = recombine_patches(predictions, resized_dims)
            sample_mask = ff(predictions)
            sample_mask = np.transpose(sample_mask.numpy(), (1, 2, 0))
            sample_mask = cv2.resize(sample_mask, (original_dims[1], original_dims[0]), interpolation=cv2.INTER_NEAREST)
            masks.append(sample_mask)
            labels.append(sample_labels.numpy().squeeze(axis=(0,1)))
            for i in range(len(label_files)):
                label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

    print('Total time to evaluate: {}'.format(elapsed_time(time() - start_eval)))
    return masks, labels, label_list
