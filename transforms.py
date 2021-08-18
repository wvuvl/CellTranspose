import math
# from torch import tensor, mean, unique, zeros, ones, empty, cat, squeeze, unsqueeze, as_tensor, no_grad, equal
import torch
import cv2
import numpy as np
import copy
from cellpose_src.dynamics import masks_to_flows, follow_flows, get_masks, remove_bad_flow_masks
from cellpose_src.utils import diameters, fill_holes_and_remove_small_masks
from cellpose_src.transforms import _taper_mask
import random
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt


# TODO: Need to update to work for all situations (currently only for when 1-channel 2D image doesn't include channel dim)
"""
Reformats raw input data with the following expected output:
If 2-D -> torch.tensor with shape [1, x_dim, y_dim]
If 3-D -> torch.tensor with shape [1, x_dim, y_dim, z_dim]
"""
class Reformat(object):
    def __init__(self, do_3D=False):
        super().__init__()
        self.do_3D = do_3D

    def __call__(self, x):
        if not self.do_3D:
            if x.dim() == 2:
                x = x.view(1, x.shape[0], x.shape[1])
            # Currently transforms multi-channel input to greyscale
            elif x.dim() == 3:
                # TODO: copying Cellpose implementation, find a cleaner method for solving this
                if x.shape[2] < 10:
                    info_chans = [len(torch.unique(x[:, :, i])) > 1 for i in range(x.shape[2])]
                    x = x[:, :, info_chans]
                    if x.shape[2] == 1:
                        x = x.view(1, x.shape[0], x.shape[1])
                    # else:
                else:
                    raise ValueError('Data is not 2D; if intending to use 3D volumes, pass in "--do_3D" argument.')
        # else:
        return x


class Normalize1stTo99th(object):
    """
    Normalize each channel of input image so that 0.0 corresponds to 1st percentile and 1.0 corresponds to 99th -
    Made to mimic Cellpose's normalization implementation
    """
    def __call__(self, x):
        sample = x.clone()
        for chan in range(len(sample)):
            sample[chan] = (sample[chan] - np.percentile(sample[chan], 1))\
                           / (np.percentile(sample[chan], 99) - np.percentile(sample[chan], 1))
        return sample


class LabelsToFlows(object):
    """
    Converts labels to flows for training and validation - Interfaces with Cellpose's masks_to_flows dynamics
    Returns:
        flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell probability, flows[k][2] is Y flow, and flows[k][3] is X flow
    """
    def __call__(self, label):
        flows = masks_to_flows(label.astype(int))[0]
        label = (label[np.newaxis, :] > 0.5).astype(np.float32)
        return np.concatenate((label, flows))


class FollowFlows(object):
    """
    Combines follow_flows, get_masks, and fill_holes_and_remove_small_masks from Cellpose implementation
    """
    def __init__(self, niter, interp, use_gpu, cellprob_threshold=0.0, flow_threshold=0.4, min_size=30):  # min_size=15
        super().__init__()
        self.niter = niter
        self.interp = interp
        self.use_gpu = use_gpu
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.min_size = min_size

    def __call__(self, flows):
        masks = torch.zeros((flows.shape[0], flows.shape[-2], flows.shape[-1]))
        for i, flow in enumerate(flows):
            cellprob = flow[0].cpu().numpy()
            dP = flow[1:].cpu().numpy()
            p = follow_flows(-1 * dP * (cellprob > self.cellprob_threshold) / 5., self.niter, self.interp, self.use_gpu)
            # p = follow_flows(dP * (cellprob > self.cellprob_threshold) / 5., self.niter, self.interp, self.use_gpu)

            maski = get_masks(p, iscell=(cellprob > self.cellprob_threshold), flows=dP, threshold=self.flow_threshold)
            maski = fill_holes_and_remove_small_masks(maski, min_size=self.min_size)
            masks[i] = torch.tensor(maski)
        return masks


class Resize(object):
    def __init__(self, default_med, use_labels=False, refine=True, gc_model=None, sz_model=None,
                 device='cpu', patch_per_batch=None, follow_flows_function=None):
        self.use_labels = use_labels
        self.default_med = default_med
        if not self.use_labels:
            self.gc_model = gc_model
            self.sz_model = sz_model
            self.refine = refine
            if refine:
                self.device = device
                self.patch_per_batch = patch_per_batch
                self.follow_flows_function = follow_flows_function

    def __call__(self, X, y):
        original_dims = X.shape[1], X.shape[2]  # CHECK HERE DEBUG
        if self.use_labels:
            X, y = resize_from_labels(X, y, self.default_med)
            return X, y, original_dims
        else:
            X, y = predict_and_resize(X, y, self.default_med, self.gc_model, self.sz_model)
            if self.refine:
                X, y = refined_predict_and_resize(X, y, self.default_med, self.gc_model, self.device,
                                                  self.patch_per_batch, self.follow_flows_function)
            return X, y, original_dims


def resize_from_labels(X, y, default_med):
    # calculate diameters using only full cells in image - remove cut off cells during median diameter calculation
    y_cf = copy.deepcopy(torch.squeeze(y, dim=0))
    y_cf = remove_cut_cells(y_cf)
    # cc = sorted(np.unique(np.concatenate((np.unique(y_cf[0]), np.unique(y_cf[:, 0]),
    #                                       np.unique(y_cf[-1]), np.unique(y_cf[:, -1])))))
    # for i in range(1, len(cc)):
    #     y_cf[y_cf == cc[i]] = 0
    med, cts = diameters(y_cf)
    rescale_x, rescale_y = default_med[0] / med, default_med[1] / med
    X = np.transpose(X.numpy(), (1, 2, 0))
    X = cv2.resize(X, (int(X.shape[1] * rescale_x), int(X.shape[0] * rescale_y)),
                   interpolation=cv2.INTER_LINEAR)[np.newaxis, :]
    y = np.transpose(y.numpy(), (1, 2, 0))
    y = cv2.resize(y, (int(y.shape[1] * rescale_x), int(y.shape[0] * rescale_y)),
                   interpolation=cv2.INTER_NEAREST)[np.newaxis, :]
    return torch.tensor(X), torch.tensor(y)


def predict_and_resize(X, y, default_med, gc_model, sz_model):
    X = torch.unsqueeze(X, dim=0).float()
    with torch.no_grad():
        style = gc_model(X, style_only=True)
        med = sz_model(style)
    X = torch.squeeze(X, dim=0)
    rescale_x, rescale_y = default_med[0] / med, default_med[1] / med
    X = np.transpose(X.cpu().numpy(), (1, 2, 0))
    X = cv2.resize(X, (int(X.shape[1] * rescale_x), int(X.shape[0] * rescale_y)),
                   interpolation=cv2.INTER_LINEAR)[np.newaxis, :]
    y = np.transpose(y.cpu().numpy(), (1, 2, 0))
    y = cv2.resize(y, (int(y.shape[1] * rescale_x), int(y.shape[0] * rescale_y)),
                   interpolation=cv2.INTER_NEAREST)[np.newaxis, :]
    return torch.tensor(X), torch.tensor(y)


# produce output masks using gc_model, then calculate mean diameter
def refined_predict_and_resize(X, y, default_med, gc_model, device, patch_per_batch, follow_flows_function):
        X = torch.unsqueeze(X, dim=0)
        im_dims = (X.shape[2], X.shape[3])
        sample_data, _ = generate_patches(X, y, eval=True)
        with torch.no_grad():
            predictions = torch.tensor([]).to(device)
            for patch_ind in range(0, len(sample_data), patch_per_batch):
                sample_patch_data = sample_data[patch_ind:patch_ind + patch_per_batch].float().to(device)
                p = gc_model(sample_patch_data)
                predictions = torch.cat((predictions, p))
        predictions = recombine_patches(predictions, im_dims)
        sample_mask = follow_flows_function(predictions)
        med, cts = diameters(sample_mask.numpy())
        rescale_x, rescale_y = default_med[0] / med, default_med[1] / med
        X = torch.squeeze(X, dim=0)
        X = np.transpose(X.cpu().numpy(), (1, 2, 0))
        X = cv2.resize(X, (int(X.shape[1] * rescale_x), int(X.shape[0] * rescale_y)),
                       interpolation=cv2.INTER_LINEAR)[np.newaxis, :]
        y = np.transpose(y.cpu().numpy(), (1, 2, 0))
        y = cv2.resize(y, (int(y.shape[1] * rescale_x), int(y.shape[0] * rescale_y)),
                       interpolation=cv2.INTER_NEAREST)[np.newaxis, :]
        return torch.tensor(X), torch.tensor(y)


def random_horizontal_flip(X, y):
    if np.random.rand() > .5:
        X = TF.hflip(X)
        y = TF.hflip(y)
    return X, y


def random_rotate(X, y):
    angle = random.random() * 360
    return TF.rotate(X, angle), TF.rotate(y, angle)


# Generate patches of input to be passed into model. Currently set to 64x64 patches with at least 32x32 overlap
# - image should also already be resized such that median cell diameter is 32
def generate_patches(data, label=None, eval=False, patch=(64, 64), min_overlap=(32, 32)):
    num_x_patches = math.ceil((data.shape[3] - min_overlap[0]) / (patch[0] - min_overlap[0]))
    x_patches = np.linspace(0, data.shape[3] - patch[0], num_x_patches, dtype=int)
    num_y_patches = math.ceil((data.shape[2] - min_overlap[1]) / (patch[1] - min_overlap[1]))
    y_patches = np.linspace(0, data.shape[2] - patch[1], num_y_patches, dtype=int)

    patch_data = torch.empty((data.shape[0] * num_x_patches * num_y_patches, 1, patch[0], patch[1]))
    if eval:
        patch_label = torch.empty((label.shape[0] * num_x_patches * num_y_patches, patch[0], patch[1]))
    else:
        patch_label = torch.empty((label.shape[0] * num_x_patches * num_y_patches, 3, patch[0], patch[1]))

    for b in range(data.shape[0]):
        for i in range(num_x_patches):
            for j in range(num_y_patches):
                d_patch = data[b, 0, y_patches[j]:y_patches[j] + patch[1], x_patches[i]:x_patches[i] + patch[0]]
                patch_data[(b * num_y_patches * num_x_patches) + (num_y_patches * i + j)] = d_patch
                if eval:
                    l_patch = label[b, y_patches[j]:y_patches[j] + patch[1], x_patches[i]:x_patches[i] + patch[0]]
                else:
                    l_patch = label[b, :, y_patches[j]:y_patches[j] + patch[1], x_patches[i]:x_patches[i] + patch[0]]
                patch_label[(b * num_y_patches * num_x_patches) + (num_y_patches * i + j)] = l_patch

    return patch_data, patch_label


# Removes all cell labels on the edges of the given samples
def remove_cut_cells(labels, flows=False):
    if flows:
        for i in range(len(labels)):
            label_mask = labels[i, 0]
            label_flows1 = labels[i, 1]
            label_flows2 = labels[i, 2]
            cc = sorted(np.unique(np.concatenate((np.unique(label_mask[0]), np.unique(label_mask[:, 0]),
                                                  np.unique(label_mask[-1]), np.unique(label_mask[:, -1])))))
            for j in range(1, len(cc)):
                b = (label_mask == cc[j]).nonzero(as_tuple=True)
                label_mask[b] = 0  # Shallow copy means this is reflected in labels
                label_flows1[b] = 0  # Shallow copy means this is reflected in labels
                label_flows2[b] = 0  # Shallow copy means this is reflected in labels

            labels[i, 0] = label_mask  # More explicit
            labels[i, 1] = label_flows1  # More explicit
            labels[i, 2] = label_flows2  # More explicit
    else:
        cc = sorted(np.unique(np.concatenate((np.unique(labels[0]), np.unique(labels[:, 0]),
                                              np.unique(labels[-1]), np.unique(labels[:, -1])))))
        for i in range(1, len(cc)):
            labels[labels == cc[i]] = 0
    return labels

# Removes any samples which contain labels without cells
def remove_empty_label_patches(data, labels):
    # keep_samples = []
    # for i in range(labels.shape[0]):
    #     if not equal(labels[i], zeros((labels.shape[1:]))):
    #         keep_samples.append(i)
    # data = data[keep_samples, :]
    # labels = labels[keep_samples, :]
    # return data, labels
    keep_samples = []
    num_labels = labels.shape[0]
    for i in range(num_labels):
        if not torch.equal(labels[i], torch.zeros((labels.shape[1:]))):
            keep_samples.append(i)
    num_zeros = num_labels - len(keep_samples)
    actual_keep_samples = []
    for i in range(num_labels):
        if not torch.equal(labels[i], torch.zeros((labels.shape[1:]))):
            actual_keep_samples.append(i)
        elif random.random() > 0.75:
            actual_keep_samples.append(i)
    data = data[actual_keep_samples, :]
    labels = labels[actual_keep_samples, :]
    return data, labels

"""
Creates recombined images after averaging together.
Note: Cellpose uses a tapered mask rather than simple averaging; this can be applied by simply replacing the mask_patch
with a tapered mask
"""
# TODO: Update _taper_mask to have more optimal parameters (some not included in function parameters)
def recombine_patches(labels, im_dims=(500, 500), min_overlap=(32, 32)):
    num_x_patches = math.ceil((im_dims[1] - min_overlap[0]) / (labels.shape[3] - min_overlap[0]))
    x_patches = np.linspace(0, im_dims[1] - labels.shape[3], num_x_patches, dtype=int)
    num_y_patches = math.ceil((im_dims[0] - min_overlap[1]) / (labels.shape[2] - min_overlap[1]))
    y_patches = np.linspace(0, im_dims[0] - labels.shape[2], num_y_patches, dtype=int)

    # mask_patch = ones((labels.shape[3], labels.shape[2])).to('cuda')
    mask_patch = torch.tensor(_taper_mask(lx=labels.shape[3], ly=labels.shape[2], sig=7.5)).to('cuda')
    num_ims = labels.shape[0] // (num_x_patches * num_y_patches)
    recombined_labels = torch.zeros((num_ims, 3, im_dims[0], im_dims[1])).to('cuda')
    recombined_mask = torch.zeros((num_ims, 3, im_dims[0], im_dims[1])).to('cuda')

    for b in range(num_ims):
        for i in range(num_x_patches):
            for j in range(num_y_patches):
                recombined_labels[b, :, y_patches[j]:y_patches[j] + labels.shape[2],
                                  x_patches[i]:x_patches[i] + labels.shape[3]] +=\
                    labels[(b * num_y_patches * num_x_patches) + (num_y_patches * i + j)] * mask_patch
                recombined_mask[b, :, y_patches[j]:y_patches[j] + labels.shape[2],
                                x_patches[i]:x_patches[i] + labels.shape[3]] += mask_patch

    recombined_labels /= recombined_mask

    return recombined_labels
