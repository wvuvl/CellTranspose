import torchvision
import math
from torch import tensor, mean, unique, zeros, ones, empty, cat
import cv2
import numpy as np
from cellpose_src.dynamics import masks_to_flows, follow_flows, get_masks, remove_bad_flow_masks
from cellpose_src.utils import fill_holes_and_remove_small_masks
from cellpose_src.transforms import _taper_mask
import tifffile
import os
import random
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt


# Need to update to work for all situations (currently only for when 1-channel 2D image doesn't include channel dim)
class Reformat(object):
    def __init__(self, do_2D=True):
        super().__init__()
        self.do_2D = do_2D

    def __call__(self, x):
        if self.do_2D:
            if x.dim() == 2:
                x = x.view(1, x.shape[0], x.shape[1])
            # Currently transforms multi-channel input to greyscale
            elif x.dim() == 3:
                # TODO: copying Cellpose implementation, find a cleaner method for solving this
                if x.shape[2] < 10:
                    info_chans = [len(unique(x[:, :, i])) > 1 for i in range(x.shape[2])]
                    x = x[:, :, info_chans]
                    if x.shape[2] == 1:
                        x = x.view(1, x.shape[0], x.shape[1])
                    # else:
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


class ResizeImage(object):  # Pass as <tensor> or numpy array? Return as tensor or <numpy array>?
    def __init__(self, x_rf, y_rf, interpolation):
        super().__init__()
        self.rescale_x = x_rf
        self.rescale_y = y_rf
        self.interpolation = interpolation

    def __call__(self, im):
        im = np.transpose(im.numpy(), (1, 2, 0))
        im = cv2.resize(im, (int(im.shape[1] * self.rescale_x), int(im.shape[0] * self.rescale_y)), interpolation=self.interpolation)
        return im

        # Do 3D part


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
        masks = zeros((flows.shape[0], flows.shape[-2], flows.shape[-1]))
        for i, flow in enumerate(flows):
            cellprob = flow[0].cpu().numpy()
            dP = flow[1:].cpu().numpy()
            p = follow_flows(-1 * dP * (cellprob > self.cellprob_threshold) / 5., self.niter, self.interp, self.use_gpu)
            # p = follow_flows(dP * (cellprob > self.cellprob_threshold) / 5., self.niter, self.interp, self.use_gpu)

            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(p[0])
            # plt.colorbar()
            # plt.subplot(1, 2, 2)
            # plt.imshow(p[1])
            # plt.colorbar()
            # plt.tight_layout()
            # plt.show()

            maski = get_masks(p, iscell=(cellprob > self.cellprob_threshold), flows=dP, threshold=self.flow_threshold)

            maski = fill_holes_and_remove_small_masks(maski, min_size=self.min_size)

            # plt.figure()
            # plt.imshow(maski)
            # plt.show()

            masks[i] = tensor(maski)
        return masks


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

    patch_data = empty((data.shape[0] * num_x_patches * num_y_patches, 1, patch[0], patch[1]))
    if eval:
        patch_label = empty((label.shape[0] * num_x_patches * num_y_patches, patch[0], patch[1]))
    else:
        patch_label = empty((label.shape[0] * num_x_patches * num_y_patches, 3, patch[0], patch[1]))

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
    mask_patch = tensor(_taper_mask(lx=labels.shape[3], ly=labels.shape[2], sig=7.5)).to('cuda')
    num_ims = labels.shape[0] // (num_x_patches * num_y_patches)
    recombined_labels = zeros((num_ims, 3, im_dims[0], im_dims[1])).to('cuda')
    recombined_mask = zeros((num_ims, 3, im_dims[0], im_dims[1])).to('cuda')

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
