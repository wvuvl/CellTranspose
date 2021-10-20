from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, as_tensor, squeeze
import time
from tqdm import tqdm
import cv2
import numpy as np
from statistics import mean
from transforms import followflows, generate_patches, recombine_patches

import matplotlib.pyplot as plt


def train_network(model, train_dl, val_dl, class_loss, flow_loss, optimizer, scheduler, device, n_epochs):
    train_losses = []
    val_losses = []
    start_train = time.time()

    print('Beginning network training.\n')
    for e in range(1, n_epochs + 1):
        try:
            train_epoch_losses = []
            model.train()
            print(scheduler.get_last_lr())
            for (sample_data, sample_labels) in tqdm(train_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs)):
                sample_data = sample_data.to(device)
                sample_labels = sample_labels.to(device)
                optimizer.zero_grad()
                output = model(sample_data)
                mask_loss = class_loss(output, sample_labels)
                grad_loss = flow_loss(output, sample_labels)
                train_loss = mask_loss + grad_loss
                train_epoch_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

            scheduler.step()
            train_epoch_loss = mean(train_epoch_losses)
            train_losses.append(train_epoch_loss)
            if val_dl is not None:
                val_epoch_loss = validate_network(model, val_dl, flow_loss, class_loss, device)
                val_losses.append(val_epoch_loss)
                print('Train loss: {:.3f}; Validation loss: {:.3f}'.format(train_epoch_loss, val_epoch_loss))
            else:
                print('Train loss: {:.3f}'.format(train_epoch_loss))

            if e % (n_epochs / 5) == 0:
                plt.figure()
                epoch_i = np.arange(1, e+1)
                plt.plot(epoch_i, train_losses)
                plt.plot(epoch_i, val_losses)
                plt.legend(('Train Losses', 'Validation Losses'))
                plt.show()
        except KeyboardInterrupt:
            print('Exiting early.')
            break

    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def adapt_network(model: nn.Module, source_dl, target_dl, val_dl, sas_class_loss, c_flow_loss,
                  class_loss, flow_loss, optimizer, scheduler, device, n_epochs):
    train_losses = []
    val_losses = []
    print('Beginning domain adaptation.\n')

    # Assume # of target samples << # of source samples
    start_train = time.time()
    for e in range(1, n_epochs + 1):
        try:
            model.train()
            train_epoch_losses = []
            target_dl_iter = iter(target_dl)
            for i, (source_sample_data, source_sample_labels) in enumerate(tqdm(
                    source_dl, desc='Training - Epoch {}/{}'.format(e, n_epochs))):
                optimizer.zero_grad()

                source_sample_data = source_sample_data.to(device)
                source_sample_labels = source_sample_labels.to(device)
                source_output = model(source_sample_data)

                try:
                    target_sample = next(target_dl_iter)
                except StopIteration:
                    target_dl_iter = iter(target_dl)
                    target_sample = next(target_dl_iter)
                target_sample_data = target_sample[0].to(device)
                target_sample_labels = target_sample[1].to(device)
                target_output = model(target_sample_data)
                # source_class_loss = class_loss(source_output, source_sample_labels)
                # target_class_loss = class_loss(target_output, target_sample_labels)
                # source_grad_loss = flow_loss(source_output, source_sample_labels)
                target_grad_loss = flow_loss(target_output, target_sample_labels)
                adaptation_class_loss = sas_class_loss(source_output[:, 0], source_sample_labels[:, 0],
                                                       target_output[:, 0], target_sample_labels[:, 0],
                                                       margin=0.25, gamma_1=0.2, gamma_2=0.25)
                adaptation_flow_loss = c_flow_loss(source_output[:, 1:], source_sample_labels[:, 1:],
                                                   target_output[:, 1:], target_sample_labels[:, 1:],
                                                   temperature=0.1)

                # train_loss = target_class_loss + target_grad_loss
                train_loss = adaptation_class_loss + target_grad_loss
                # train_loss = adaptation_class_loss + source_grad_loss + target_grad_loss
                train_epoch_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

            scheduler.step()
            train_losses.append(mean(train_epoch_losses))
            if val_dl is not None:
                val_epoch_loss = validate_network(model, val_dl, flow_loss, class_loss, device)
                val_losses.append(val_epoch_loss)
                print('Train loss: {:.3f}; Validation loss: {:.3f}'.format(mean(train_epoch_losses), val_epoch_loss))
            else:
                print('Train loss: {:.3f}'.format(mean(train_epoch_losses)))

            if e % (n_epochs / 5) == 0:
                plt.figure()
                epoch_i = np.arange(1, e+1)
                plt.plot(epoch_i, train_losses)
                plt.plot(epoch_i, val_losses)
                plt.legend(('Train Losses', 'Validation Losses'))
                plt.show()
        except KeyboardInterrupt:
            print('Exiting early.')
            break

    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def validate_network(model, data_loader, flow_loss, class_loss, device):
    model.eval()
    val_epoch_losses = []
    with no_grad():
        for (val_sample_data, val_sample_labels) in tqdm(data_loader, desc='Performing validation'):
            # When not using precalculated flows, makes up majority of validation time (~85-90%)
            val_sample_data = val_sample_data.to(device)
            val_sample_labels = val_sample_labels.to(device)
            output = model(val_sample_data)
            grad_loss = flow_loss(output, val_sample_labels).item()
            mask_loss = class_loss(output, val_sample_labels).item()
            val_loss = grad_loss + mask_loss
            val_epoch_losses.append(val_loss)
    return mean(val_epoch_losses)


# Evaluation - due to image size mismatches, must currently be run one image at a time
def eval_network(model: nn.Module, data_loader: DataLoader, device, patch_per_batch, patch_size, min_overlap):

    model.eval()
    print('Beginning evaluation.')
    with no_grad():
        masks = []
        label_list = []
        pred_list = []
        for (sample_data, sample_labels, label_files, original_dims) in data_loader:
            resized_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, _ = generate_patches(sample_data, squeeze(sample_labels, dim=0), eval=True,
                                              patch=patch_size, min_overlap=min_overlap)
            predictions = tensor([]).to(device)

            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_data_patches = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                p = model(sample_data_patches)
                predictions = cat((predictions, p))

            predictions = recombine_patches(predictions, resized_dims, min_overlap)
            sample_mask = followflows(predictions)
            sample_mask = np.transpose(sample_mask.numpy(), (1, 2, 0))
            sample_mask = cv2.resize(sample_mask, (original_dims[1].item(), original_dims[0].item()),
                                     interpolation=cv2.INTER_NEAREST)
            pred_list.append(predictions.cpu().numpy()[0])
            masks.append(sample_mask)
            for i in range(len(label_files)):
                label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

    return masks, pred_list, label_list
