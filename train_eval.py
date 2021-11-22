from torch.utils.data import DataLoader
from torch import nn, tensor, cat, no_grad, as_tensor, squeeze, zeros
import time
from tqdm import tqdm
import cv2
import numpy as np
from statistics import mean
from transforms import followflows, generate_patches, recombine_patches


def train_network(model, train_dl, val_dl, class_loss, flow_loss, optimizer, scheduler, device, n_epochs):
    train_losses = []
    val_losses = []
    start_train = time.time()

    print('Beginning network training.\n')
    for e in range(1, n_epochs + 1):
        train_epoch_losses = []
        model.train()
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
            sample_mask = followflows(predictions)
            sample_mask = np.transpose(sample_mask.numpy(), (1, 2, 0))
            if padding:
                sample_mask = sample_mask[set_corner[0]:set_corner[0]+unpadded_dims[0],
                                          set_corner[1]:set_corner[1]+unpadded_dims[1]]
            sample_mask = cv2.resize(sample_mask, (original_dims[1].item(), original_dims[0].item()),
                                     interpolation=cv2.INTER_NEAREST)
            pred_list.append(predictions.cpu().numpy()[0])
            masks.append(sample_mask)
            for i in range(len(label_files)):
                label_list.append(label_files[i][label_files[i].rfind('/')+1: label_files[i].rfind('.')])

    return masks, pred_list, label_list
