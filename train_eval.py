from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, as_tensor, unsqueeze
from time import time
from tqdm import tqdm
import cv2
import numpy as np
from statistics import mean
import math
from misc_utils import elapsed_time
from transforms import LabelsToFlows, FollowFlows, resize_from_labels, predict_and_resize, random_horizontal_flip,\
    random_rotate, generate_patches, recombine_patches, refined_predict_and_resize

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
        train_dl.dataset.reprocess_on_epoch(default_meds)
        for (sample_data, sample_labels) in tqdm(train_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs)):
            sample_data = sample_data.to(device)
            sample_labels = sample_labels.to(device)
            optimizer.zero_grad()
            output = model(sample_data)
            grad_loss = flow_loss(output, sample_labels)
            mask_loss = class_loss(output, sample_labels)
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

    # Assume # of target samples << # of source samples
    start_train = time()
    for e in range(1, n_epochs + 1):
        model.train()
        train_epoch_losses = []
        reprocess_source_time = time()
        source_dl.dataset.reprocess_on_epoch(default_meds)
        print('Time to reprocess source data: {}'.format(time() - reprocess_source_time))

        reprocess_target_time = time()
        target_dl.dataset.reprocess_on_epoch(default_meds)
        target_data = target_dl.dataset.data_samples
        target_labels = target_dl.dataset.label_samples
        batched_target_data = target_data
        batched_target_labels = target_labels
        for _ in range(1, math.ceil(len(source_dl.dataset) / len(target_data))):
            batched_target_data = cat((batched_target_data, target_data))
            batched_target_labels = cat((batched_target_labels, target_labels))

        # Shuffle the target data
        t_len = len(source_dl.dataset)
        shuffled_inds = np.array(range(t_len))
        np.random.shuffle(shuffled_inds)
        batched_target_data = batched_target_data[:t_len]
        batched_target_data[np.array(range(t_len))] = batched_target_data[shuffled_inds]
        batched_target_data = batched_target_data.float()
        batched_target_labels = batched_target_labels[:t_len]
        batched_target_labels[np.array(range(t_len))] = batched_target_labels[shuffled_inds]
        batched_target_labels = batched_target_labels.float()
        print('Time to reprocess target data: {}'.format(time() - reprocess_target_time))

        for i, (source_sample_data, source_sample_labels) in enumerate(tqdm(source_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs))):
            optimizer.zero_grad()

            source_sample_data = source_sample_data.to(device)
            source_sample_labels = source_sample_labels.to(device)
            source_output = model(source_sample_data)

            target_sample_data = batched_target_data[i*target_dl.batch_size:(i+1)*target_dl.batch_size].to(device)
            target_sample_labels = batched_target_labels[i*target_dl.batch_size:(i+1)*target_dl.batch_size].to(device)
            target_output = model(target_sample_data)
            source_grad_loss = flow_loss(source_output, source_sample_labels)
            adaptation_class_loss = sas_class_loss(source_output[:, 0], source_sample_labels[:, 0],
                                                   target_output[:, 0], target_sample_labels[:, 0],
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
            val_sample_data, val_sample_labels = resize_from_labels(val_sample_data[0], val_sample_labels[0], default_meds)  # MUST BE FIXED IF BATCH SIZE > 1 IS USED
            val_sample_data, val_sample_labels = generate_patches(unsqueeze(val_sample_data, 0), val_sample_labels, eval=True)
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


def eval_network(model: nn.Module, data_loader: DataLoader, device, patch_per_batch, default_meds, gc_model, sz_model,
                 refine):

    model.eval()
    start_eval = time()
    ff = FollowFlows(niter=100, interp=True, use_gpu=True, cellprob_threshold=0.0, flow_threshold=1.0)
    with no_grad():
        masks = []
        labels = []
        label_list = []
        for (sample_data, sample_labels, label_files) in data_loader:
            original_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, resized_sample_labels = predict_and_resize(sample_data.float().to(device),
                                                                    sample_labels.to(device), default_meds, gc_model,
                                                                    sz_model)
            if refine:
                sample_data, resized_sample_labels = refined_predict_and_resize(sample_data, resized_sample_labels,
                                                                                default_meds, gc_model, device,
                                                                                patch_per_batch, ff)
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
            labels.append(sample_labels.numpy().squeeze(axis=(0, 1)))
            for i in range(len(label_files)):
                label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

    print('Total time to evaluate: {}'.format(elapsed_time(time() - start_eval)))
    return masks, labels, label_list
