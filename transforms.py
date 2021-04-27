import torchvision
from torch import tensor, nn, zeros
import cv2
import numpy as np
from cellpose_src.dynamics import masks_to_flows, follow_flows, get_masks, remove_bad_flow_masks
from cellpose_src.utils import fill_holes_and_remove_small_masks
import tifffile
import os
import random
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt


# Need to update to work for all situations (currently only for when 1-channel 2D image doesn't include channel dim)
class Reshape(object):
    def __init__(self, do_2D=True):
        super().__init__()
        self.do_2D = do_2D

    def __call__(self, x):
        if self.do_2D:
            if x.dim() == 2:
                x = x.view(1, x.shape[0], x.shape[1])
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
    def __init__(self, x_width, x_height, interpolation):
        super().__init__()
        self.x_width = x_width
        self.x_height = x_height
        self.interpolation = interpolation

    def __call__(self, x):
        x = np.transpose(x.numpy(), (1, 2, 0))
        x = cv2.resize(x, (self.x_width, self.x_height), interpolation=self.interpolation)
        return x

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
    def __init__(self, niter, interp, use_gpu, cellprob_threshold=0.0, flow_threshold=0.4, min_size=30): # min_size=15
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
