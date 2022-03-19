from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, squeeze, zeros
import time
from tqdm import tqdm
import cv2
import numpy as np
import os
import pickle
import tifffile
from statistics import mean
from transforms import followflows, followflows3D, generate_patches, recombine_patches
from cellpose_src import transforms

def train_network(model, train_dl, val_dl, class_loss, flow_loss, optimizer, scheduler, device, n_epochs):
    train_losses = []
    val_losses = []
    start_train = time.time()

    print('Beginning network training.\n')
    for e in range(1, n_epochs + 1):
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

    print('Train time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_train))))
    return train_losses, val_losses


def adapt_network(model: nn.Module, source_dl, target_dl, val_dl, sas_mask_loss, contrastive_flow_loss,
                  class_loss, flow_loss, train_direct, optimizer, scheduler, device, n_epochs,
                  k, gamma_1, gamma_2, n_thresh, temperature):
    train_losses = []
    val_losses = []
    print('Beginning domain adaptation.\n')

    # Assume # of target samples << # of source samples
    start_train = time.time()
    for e in range(1, n_epochs + 1):
        model.train()
        print(scheduler.get_last_lr())
        train_epoch_losses = []
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
            if not train_direct:
                if e <= n_epochs/2:
                    adaptation_class_loss = sas_mask_loss(source_output[:, 0], source_sample_labels[:, 0],
                                                          target_output[:, 0], target_sample_labels[:, 0],
                                                          margin=10, gamma_1=gamma_1, lam=0.5)
                    target_class_loss = class_loss(target_output, target_sample_labels)
                    c_loss = target_class_loss + adaptation_class_loss
                    adaptation_flow_loss = contrastive_flow_loss(source_output[:, 1:], source_sample_labels,
                                                                 target_output[:, 1:], target_sample_labels,
                                                                 k=k, gamma_2=gamma_2, n_thresh=n_thresh,
                                                                 temperature=temperature)
                    target_flow_loss = flow_loss(target_output, target_sample_labels)
                    f_loss = target_flow_loss + adaptation_flow_loss
                    train_loss = c_loss + f_loss
                else:
                    target_class_loss = class_loss(target_output, target_sample_labels)
                    target_flow_loss = flow_loss(target_output, target_sample_labels)
                    train_loss = target_class_loss + target_flow_loss
            else:
                target_class_loss = class_loss(target_output, target_sample_labels)
                target_flow_loss = flow_loss(target_output, target_sample_labels)
                train_loss = target_class_loss + target_flow_loss

            train_epoch_losses.append(train_loss.item())
            train_target_class_losses.append(target_class_loss.item())
            train_target_flow_losses.append(target_flow_loss.item())
            train_loss.backward()
            optimizer.step()

        if e <= n_epochs/2:
            scheduler.step()
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
    with no_grad():
        masks = []
        label_list = []
        pred_list = []
        # original_dims_list = []
        for (sample_data, sample_labels, label_files, original_dims) in tqdm(data_loader, desc='Evaluating Test Dataset'):
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
            #original_dims_list.append(original_dims)

    # TODO: Ram: remember to remove original_dims_list before merging
    return masks, pred_list, label_list  # , original_dims_list


# Evaluation - due to image size mismatches, must currently be run one image at a time
def eval_network_3D(model: nn.Module, data_loader: DataLoader, device,
                    patch_per_batch, patch_size, min_overlap, results_dir):
    model.eval()
    with no_grad():
        for (data_vol, label_vol, label_files, X, dX,dim) in data_loader:
            pred_yx = []
            pred_zx = []
            pred_zy = []
        
            for index in range(len(dX)): 

                for (sample_data, sample_labels,origin_dim) in tqdm(zip(data_vol[index],label_vol[index],dim[index]),desc=f'Processing {dX[index]}'):
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

                    predictions = recombine_patches(predictions, resized_dims, min_overlap).cpu().numpy()[0]
                    predictions = predictions.transpose(1,2,0)
                    predictions = transforms.resize_image(predictions, origin_dim[0].item(), origin_dim[1].item())
                    
                    predictions = predictions.transpose(2,0,1)
                    if index == 0: pred_yx.append(predictions)
                    elif index == 1: pred_zx.append(predictions)
                    else: pred_zy.append(predictions)

            pred_yx, pred_zy, pred_zx = np.array(pred_yx),np.array(pred_zy),np.array(pred_zx)
            run_3D_masks(pred_yx,pred_zy,pred_zx,label_files,results_dir)    
    
    # TODO: Ram: remember to remove original_dims_list before merging
    # return masks, pred_list, label_list, original_dims_list


# adapted from cellpose original implementation
# TODO: does not work for patch size smaller than at least one image dimension, padding required
def run_3D_masks(pred_yx, pred_zy, pred_zx,label_name,results_dir):
    
    pred_yx = pred_yx.transpose(1,0,2,3)
    pred_zy_xy = pred_zy.transpose(1,2,3,0)
    pred_zx_xy = pred_zx.transpose(1,2,0,3)

    yf = np.zeros((3, 3, pred_yx.shape[1], pred_yx.shape[2], pred_yx.shape[3]), np.float32)
    
    yf[0] = pred_yx
    yf[1] = pred_zy_xy
    yf[2] = pred_zx_xy
    
    cellprob = yf[0][0] + yf[1][0] + yf[2][0]
    
    # sets perfect dims, but still getting wrong number of masks
    dP = np.stack((yf[1][1] + yf[2][1], yf[0][1] + yf[1][2], yf[0][2] + yf[2][2]), axis=0) # (dZ, dY, dX)
    
    
    mask = np.array(followflows3D(dP,cellprob))
    
    print(f">>>Total masks found in this 3D image: ", len(np.unique(mask))-1)
    
    label_list = []
    for i in range(len(label_name)):
        label_list.append(label_name[i][label_name[i].rfind('/')+1: label_name[i].rfind('.')])
                    
    with open(os.path.join(results_dir, label_list[0] + '_raw_masks_flows.pkl'), 'wb') as rmf_pkl:
            pickle.dump(yf, rmf_pkl)
    tifffile.imwrite(os.path.join(results_dir, 'tiff_results', label_list[0] + '.tif'), mask)
    
    del yf
    del dP
    del cellprob
    del mask
    #return mask, yf
