from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, as_tensor, squeeze, zeros
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

            """if e % (n_epochs / 5) == 0:
                plt.figure()
                epoch_i = np.arange(1, e+1)
                plt.plot(epoch_i, train_losses)
                plt.plot(epoch_i, val_losses)
                plt.legend(('Train Losses', 'Validation Losses'))
                plt.show()"""
        except KeyboardInterrupt:
            print('Exiting early.')
            break

    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def adapt_network(model: nn.Module, source_dl, target_dl, val_dl, sas_class_loss, c_flow_loss,
                  class_loss, flow_loss, optimizer, scheduler, device, n_epochs,
                  k, lmbda, hardness_thresh, temperature):
    train_losses = []
    val_losses = []
    print('Beginning domain adaptation.\n')

    # Assume # of target samples << # of source samples
    start_train = time.time()
    for e in range(1, n_epochs + 1):
        try:
            model.train()
            print(scheduler.get_last_lr())
            train_epoch_losses = []
            train_adaptation_class_losses = []
            train_adaptation_flow_losses = []
            train_target_class_losses = []
            train_target_flow_losses = []
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
                target_class_loss = class_loss(target_output, target_sample_labels)
                # source_flow_loss = flow_loss(source_output, source_sample_labels)
                target_flow_loss = flow_loss(target_output, target_sample_labels)
                adaptation_class_loss = sas_class_loss(source_output[:, 0], source_sample_labels[:, 0],
                                                       target_output[:, 0], target_sample_labels[:, 0],
                                                       margin=10, gamma_1=0.8, gamma_2=0.25)
                adaptation_flow_loss = c_flow_loss(source_output[:, 1:], source_sample_labels,
                                                   target_output[:, 1:], target_sample_labels,
                                                   k=k, lmbda=lmbda, hardness_thresh=hardness_thresh, temperature=temperature)

                if e == 1:
                    train_loss = target_class_loss + target_flow_loss
                else:
                    train_loss = adaptation_class_loss + adaptation_flow_loss
                # train_loss = source_class_loss + source_flow_loss
                train_epoch_losses.append(train_loss.item())
                train_adaptation_class_losses.append(adaptation_class_loss.item())
                train_adaptation_flow_losses.append(adaptation_flow_loss.item())
                train_target_class_losses.append(target_class_loss.item())
                train_target_flow_losses.append(target_flow_loss.item())
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

    # temporary for speedup
    patch_per_batch = 256

    model.eval()
    with no_grad():
        masks = []
        label_list = []
        pred_list = []
        for (sample_data, sample_labels, label_files, original_dims) in tqdm(data_loader,
                                                                             desc='Evaluating Test Dataset'):
            resized_dims = (sample_data.shape[2], sample_data.shape[3])
            padding = sample_data.shape[2] < patch_size[0] or sample_data.shape[3] < patch_size[1]
            # Add padding if image is smaller than patch size in at least one dimension
            if padding:
                unpadded_dims = resized_dims
                sd = zeros((sample_data.shape[0], sample_data.shape[1], max(patch_size[0], sample_data.shape[2]),
                            max(patch_size[1], sample_data.shape[3])))
                sl = zeros((sample_labels.shape[0], sample_labels.shape[1], max(patch_size[0], sample_data.shape[2]),
                            max(patch_size[1], sample_labels.shape[3])))
                set_corner = (max(0, (patch_size[0] - sample_data.shape[2]) // 2),
                              max(0, (patch_size[1] - sample_data.shape[3]) // 2))
                sd[:, :, set_corner[0]:set_corner[0] + sample_data.shape[2],
                   set_corner[1]:set_corner[1] + sample_data.shape[3]] = sample_data
                sl[:, :, set_corner[0]:set_corner[0] + sample_labels.shape[2],
                   set_corner[1]:set_corner[1] + sample_labels.shape[3]] = sample_labels
                sample_data = sd
                sample_labels = sl
                resized_dims = (sample_data.shape[2], sample_data.shape[3])
            sample_data, _ = generate_patches(sample_data, squeeze(sample_labels, dim=0), patch=patch_size,
                                              min_overlap=min_overlap, lbl_flows=False)
            predictions = tensor([]).to(device)

            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_data_patches = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                p = model(sample_data_patches)
                predictions = cat((predictions, p))

            predictions = recombine_patches(predictions, resized_dims, min_overlap)
            pred_list.append(predictions.cpu().numpy()[0])
            for i in range(len(label_files)):
                    label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

            sample_mask = followflows(predictions)
            sample_mask = np.transpose(sample_mask.numpy(), (1, 2, 0))
            if padding:
                sample_mask = sample_mask[set_corner[0]:set_corner[0]+unpadded_dims[0],
                                        set_corner[1]:set_corner[1]+unpadded_dims[1]]
            sample_mask = cv2.resize(sample_mask, (original_dims[1].item(), original_dims[0].item()),
                                    interpolation=cv2.INTER_NEAREST)
            masks.append(sample_mask)

            
    return masks, pred_list, label_list

# Evaluation - due to image size mismatches, must currently be run one image at a time
def eval_network_3D(model: nn.Module, data_loader: DataLoader, device, patch_per_batch, patch_size, min_overlap):

    # temporary for speedup
    patch_per_batch = 256

    model.eval()
    with no_grad():
        label_list = []
        pred_list = []
        for (sample_data_obj, sample_labels_obj, label_files, original_dims) in tqdm(data_loader,
                                                                             desc='Evaluating Test Dataset'):       
            pred_list_slices = []
            label_list_slices = []
            for i in range(len(sample_data_obj)):
                sample_data, sample_labels = sample_data_obj[i], sample_labels_obj[i]

                resized_dims = (sample_data.shape[2], sample_data.shape[3])
                padding = sample_data.shape[2] < patch_size[0] or sample_data.shape[3] < patch_size[1]
                # Add padding if image is smaller than patch size in at least one dimension
                if padding:
                    unpadded_dims = resized_dims
                    sd = zeros((sample_data.shape[0], sample_data.shape[1], max(patch_size[0], sample_data.shape[2]),
                                max(patch_size[1], sample_data.shape[3])))
                    sl = zeros((sample_labels.shape[0], sample_labels.shape[1], max(patch_size[0], sample_data.shape[2]),
                                max(patch_size[1], sample_labels.shape[3])))
                    set_corner = (max(0, (patch_size[0] - sample_data.shape[2]) // 2),
                                max(0, (patch_size[1] - sample_data.shape[3]) // 2))
                    sd[:, :, set_corner[0]:set_corner[0] + sample_data.shape[2],
                    set_corner[1]:set_corner[1] + sample_data.shape[3]] = sample_data
                    sl[:, :, set_corner[0]:set_corner[0] + sample_labels.shape[2],
                    set_corner[1]:set_corner[1] + sample_labels.shape[3]] = sample_labels
                    sample_data = sd
                    sample_labels = sl
                    resized_dims = (sample_data.shape[2], sample_data.shape[3])
                sample_data, _ = generate_patches(sample_data, squeeze(sample_labels, dim=0), patch=patch_size,
                                                min_overlap=min_overlap, lbl_flows=False)
                predictions = tensor([]).to(device)

                for patch_ind in range(0, len(sample_data), patch_per_batch):
                    sample_data_patches = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                    p = model(sample_data_patches)
                    predictions = cat((predictions, p))

                predictions = recombine_patches(predictions, resized_dims, min_overlap)
                pred_list_slices.append(predictions.cpu().numpy()[0])
                for i in range(len(label_files)):
                        label_list_slices.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

            pred_list.append(pred_list_slices)
            label_list.append(label_list_slices)
            
    return pred_list, label_list

def create_3D_masks(pred_xy, pred_yz, pred_xz, label_xy, label_yz, label_xz):
    flows = []
    masks = []
    #Iterate though list of 3D objects
    for i in range(len(pred_xy)):
        obj_xy = pred_xy[i]
        obj_yz = pred_yz[i]
        obj_xz = pred_xz[i]

        #Average predictions from each slice
        x_pred = [(x1+x2) / 2 for x1, x2 in zip(obj_xy[0, :, 0], obj_xz[0, :, 0])]
        y_pred = [(y1+y2) / 2 for y1, y2 in zip(obj_xy[0, 0, :], obj_yz[0, :, 0])]
        z_pred = [(z1+z2) / 2 for z1, z2 in zip(obj_yz[0, 0, :], obj_xz[0, 0, :])]

        flows.append(tensor([x_pred, y_pred, z_pred]))
    
    masks = followflows(tensor([x_pred, y_pred, z_pred])) #???
    #resize

    return masks
        
