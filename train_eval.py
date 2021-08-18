from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, as_tensor, squeeze
import time
from tqdm import tqdm
import cv2
import numpy as np
from statistics import mean
import math
from transforms import LabelsToFlows, FollowFlows, generate_patches, recombine_patches

import matplotlib.pyplot as plt

def train_network(model, train_dl, val_dl, class_loss, flow_loss,
                  optimizer, device, n_epochs):
    train_losses = []
    val_losses = []
    start_train = time.time()

    print('Preprocessing data:')
    # Test reprocessing once
    reprocess_train_time = time.time()
    train_dl.dataset.reprocess_on_epoch()
    print('Time to reprocess training data: {}'.format(time.strftime("%H:%M:%S",
                                                                     time.gmtime(time.time() - reprocess_train_time))))
    print('Beginning network training.\n')

    for e in range(1, n_epochs + 1):
        train_epoch_losses = []
        model.train()
        # reprocess_train_time = time.time()
        # train_dl.dataset.reprocess_on_epoch()
        # print('Time to reprocess training data: {}'.format(time.strftime("%H:%M:%S",
        #                                                                  time.gmtime(time.time() - reprocess_train_time))))
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
        if val_dl is not None:
            val_epoch_loss = validate_network(model, val_dl, flow_loss, class_loss, device)
            val_losses.append(val_epoch_loss)
            print('Train loss: {:.3f}; Validation loss: {:.3f}'.format(mean(train_epoch_losses), val_epoch_loss))
        else:
            print('Train loss: {:.3f}'.format(mean(train_epoch_losses)))

    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def adapt_network(model: nn.Module, source_dl, target_dl, val_dl, sas_class_loss, class_loss, flow_loss,
                  optimizer, device, n_epochs):
    train_losses = []
    val_losses = []
    batched_target_data, batched_target_labels = process_src_tgt()
    print('Beginning domain adaptation.\n')

    # Assume # of target samples << # of source samples
    start_train = time.time()
    for e in range(1, n_epochs + 1):
        model.train()
        train_epoch_losses = []
        # batched_target_data, batched_target_labels = process_data()
        for i, (source_sample_data, source_sample_labels) in enumerate(tqdm(
                source_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs))):
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
        if val_dl is not None:
            val_epoch_loss = validate_network(model, val_dl, flow_loss, class_loss, device)
            val_losses.append(val_epoch_loss)
            print('Train loss: {:.3f}; Validation loss: {:.3f}'.format(mean(train_epoch_losses), val_epoch_loss))
        else:
            print('Train loss: {:.3f}'.format(mean(train_epoch_losses)))

    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def validate_network(model, data_loader, flow_loss, class_loss, device):
    model.eval()
    val_epoch_losses = []
    with no_grad():
        for (val_sample_data, val_sample_labels) in tqdm(data_loader, desc='Performing validation'):
            val_sample_data = val_sample_data.to(device)
            val_sample_labels = as_tensor(
                [LabelsToFlows()(val_sample_labels[i].numpy()) for i in range(len(val_sample_labels))]).to(device)
            output = model(val_sample_data)
            grad_loss = flow_loss(output, val_sample_labels).item()
            mask_loss = class_loss(output, val_sample_labels).item()
            val_loss = grad_loss + mask_loss
            val_epoch_losses.append(val_loss)
    return mean(val_epoch_losses)


# Evaluation - due to image size mismatches, must currently be run one image at a time
def eval_network(model: nn.Module, data_loader: DataLoader, device, patch_per_batch):

    model.eval()
    print('Beginning evaluation.')
    ff = FollowFlows(niter=100, interp=True, use_gpu=True, cellprob_threshold=0.0, flow_threshold=1.0)
    with no_grad():
        masks = []
        label_list = []
        for (sample_data, sample_labels, label_files, original_dims) in data_loader:
            resized_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, _ = generate_patches(sample_data, squeeze(sample_labels, dim=0), eval=True)
            predictions = tensor([]).to(device)

            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_patch_data = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                p = model(sample_patch_data)
                predictions = cat((predictions, p))

            predictions = recombine_patches(predictions, resized_dims)
            sample_mask = ff(predictions)
            sample_mask = np.transpose(sample_mask.numpy(), (1, 2, 0))
            sample_mask = cv2.resize(sample_mask, (original_dims[1].item(), original_dims[0].item()),
                                     interpolation=cv2.INTER_NEAREST)
            masks.append(sample_mask)
            for i in range(len(label_files)):
                label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

    return masks, label_list


def process_src_tgt(dl_src, dl_tgt):
    print('Processing data:')
    reprocess_source_time = time.time()
    dl_src.dataset.reprocess_on_epoch()
    print('Time to process source data: {}'.format(time.strftime("%H:%M:%S",
                                                                   time.gmtime(time.time() - reprocess_source_time))))
    reprocess_target_time = time.time()
    dl_tgt.dataset.reprocess_on_epoch()
    target_data = dl_tgt.dataset.data_samples
    target_labels = dl_tgt.dataset.label_samples
    batched_target_data = target_data
    batched_target_labels = target_labels
    for _ in range(1, math.ceil(len(dl_src.dataset) / len(target_data))):
        batched_target_data = cat((batched_target_data, target_data))
        batched_target_labels = cat((batched_target_labels, target_labels))

    # Shuffle the target data
    t_len = len(dl_src.dataset)
    shuffled_inds = np.array(range(t_len))
    np.random.shuffle(shuffled_inds)
    batched_target_data = batched_target_data[:t_len]
    batched_target_data[np.array(range(t_len))] = batched_target_data[shuffled_inds]
    batched_target_data = batched_target_data.float()
    batched_target_labels = batched_target_labels[:t_len]
    batched_target_labels[np.array(range(t_len))] = batched_target_labels[shuffled_inds]
    batched_target_labels = batched_target_labels.float()
    print('Time to process target data: {}'.format(time.strftime("%H:%M:%S",
                                                                   time.gmtime(time.time() - reprocess_target_time))))
    return batched_target_data, batched_target_labels
