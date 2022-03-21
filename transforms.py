import math
import torch
import cv2
import numpy as np
import copy
from cellpose_src.dynamics import masks_to_flows, follow_flows, get_masks, compute_masks
from cellpose_src.utils import fill_holes_and_remove_small_masks
from cellpose_src.transforms import _taper_mask
import random
import torchvision.transforms.functional as TF


def reformat(x, n_chan=1, do_3D=False):
    """
    Reformats raw input data with the following expected output:
    If 2-D -> torch.tensor with shape [channels, x_dim, y_dim]
    If 3-D -> torch.tensor with shape [channels, x_dim, y_dim, z_dim]
    """
    if not do_3D:
        if x.dim() == 2:
            x = x.view(1, x.shape[0], x.shape[1])
            x = torch.cat((x, torch.zeros((n_chan - 1, x.shape[1], x.shape[2]))))
        elif x.dim() == 3:
            if x.shape[2] > x.shape[0]:
                x = x.permute(1, 2, 0)
            info_chans = [len(torch.unique(x[:, :, i])) > 1 for i in range(x.shape[2])]
            x = x[:, :, info_chans]
            x = torch.tensor(np.transpose(x.numpy(), (2, 0, 1))[:n_chan])  # Remove any additional channels
            # Concatenate empty channels if image has fewer than the specified number of channels
            if x.shape[0] < n_chan:
                zeros = torch.zeros((n_chan - x.shape[0]), x.shape[1], x.shape[2])
                x = torch.cat((x, zeros))
    # else: handle 3-D scenario here
    return x


def normalize1stto99th(x):
    """
    Normalize each channel of input image so that 0.0 corresponds to 1st percentile and 1.0 corresponds to 99th -
    Made to mimic Cellpose's normalization implementation
    """
    sample = x.clone()
    for chan in range(len(sample)):
        if len(torch.unique(sample[chan])) != 1:
            sample[chan] = (sample[chan] - np.percentile(sample[chan], 1))\
                           / (np.percentile(sample[chan], 99) - np.percentile(sample[chan], 1))
    return sample


class Resize(object):
    def __init__(self, default_med, target_labels=None):
        self.default_med = default_med
        self.target_labels = target_labels

    def __call__(self, x, y, pf=None, random_scale=1.0, diameter=None):
        original_dims = y.shape[1], y.shape[2]
        x, y = resize_from_labels(x, y, self.default_med, pf, random_scale=random_scale,
                                  diameter=diameter, target_labels=self.target_labels)
        return x, y, original_dims


def resize_from_labels(x, y, default_med, pf=None, random_scale=1.0, diameter=None, target_labels=None):
    unq = torch.unique(y)
    if len(unq) == 1 and unq == 0:
        return x, y

    # calculate diameters, using only full cells in the given image or images
    if target_labels is None:
        y_cf = copy.deepcopy(torch.squeeze(y, dim=0))
        y_cf = remove_cut_cells(y_cf)
        diams = diam_range(y_cf)
        med = np.percentile(np.array(diams), 75)*random_scale
        med = med if med > 12 else 12  # Note: following work from TissueNet
    else:
        diams = []
        for t_label in target_labels:
            t_cf = copy.deepcopy(torch.squeeze(t_label, dim=0))
            t_cf = remove_cut_cells(t_cf)
            diams = diams + diam_range(t_cf)
        med = np.percentile(np.array(diams), 75)*random_scale
        med = med if med > 12 else 12

    if med > 0:
        rescale_w, rescale_h = default_med[0] / med, default_med[1] / med
        x = np.transpose(x.numpy(), (1, 2, 0))
        x = cv2.resize(x, (int(x.shape[1] * rescale_w), int(x.shape[0] * rescale_h)),
                       interpolation=cv2.INTER_LINEAR)
        if x.ndim == 2:
            x = x[np.newaxis, :]
        else:
            x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y.numpy(), (1, 2, 0))
        y = cv2.resize(y, (int(y.shape[1] * rescale_w), int(y.shape[0] * rescale_h)),
                       interpolation=cv2.INTER_NEAREST)[np.newaxis, :]
        if pf is not None:
            pf = np.transpose(pf[0], (1, 2, 0))
            pf = cv2.resize(pf, (int(pf.shape[1] * rescale_w), int(pf.shape[0] * rescale_h)),
                            interpolation=cv2.INTER_LINEAR)
            pf = np.transpose(pf, (2, 0, 1))
            pf[0] = (pf[0] > 0.5).astype(np.float32)
            return torch.tensor(x), torch.tensor(pf)
        return torch.tensor(x), torch.tensor(y)
    else:
        return x, y


def random_horizontal_flip(x, y):
    if np.random.rand() > .5:
        x = TF.hflip(x)
        y = TF.hflip(y)
    return x, y


def random_rotate(x, y):
    angle = random.random() * 360
    return TF.rotate(x, angle), TF.rotate(y, angle)


def labels_to_flows(label):
    """
    Converts labels to flows for training and validation - Interfaces with Cellpose's masks_to_flows dynamics
    Returns:
        flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell probability, flows[k][2] is Y flow, and flows[k][3] is X flow
    """
    flows = masks_to_flows(label.astype(int))[0]
    label = (label[np.newaxis, :] > 0.5).astype(np.float32)
    return np.concatenate((label, flows))


def followflows(flows):
    """
    Combines follow_flows, get_masks, and fill_holes_and_remove_small_masks from Cellpose implementation
    """
    niter = 400; interp = True; use_gpu = True; cellprob_threshold = 0.0; flow_threshold = 0.4; min_size = 15
    masks = torch.zeros((flows.shape[0], flows.shape[-2], flows.shape[-1]))
    for i, flow in enumerate(flows):
        cellprob = flow[0].cpu().numpy()
        dP = flow[1:].cpu().numpy()
        
        p = follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5., niter, interp, use_gpu)
        maski = get_masks(p, iscell=(cellprob > cellprob_threshold), flows=dP, threshold=flow_threshold)
        maski = fill_holes_and_remove_small_masks(maski, min_size=min_size)
        masks[i] = torch.tensor(maski)
    return masks

def followflows3D(dP,cellprob):
    """
    Combines follow_flows, get_masks, and fill_holes_and_remove_small_masks from Cellpose implementation
    """
    niter = 400; interp = True; use_gpu = True; cellprob_threshold = 0.0; flow_threshold = 0.4; min_size = 16000
    masks = compute_masks(dP, cellprob, niter=niter, interp=interp, use_gpu=use_gpu, mask_threshold=cellprob_threshold,
                          flow_threshold=flow_threshold, min_size=min_size)
    return masks


# Generate patches of input to be passed into model. Currently set to 64x64 patches with at least 32x32 overlap
# - image should also already be resized such that median cell diameter is 32
def generate_patches(data, label=None, patch=(96, 96), min_overlap=(64, 64), lbl_flows=False):
    num_x_patches = math.ceil((data.shape[3] - min_overlap[0]) / (patch[0] - min_overlap[0]))
    x_patches = np.linspace(0, data.shape[3] - patch[0], num_x_patches, dtype=int)
    num_y_patches = math.ceil((data.shape[2] - min_overlap[1]) / (patch[1] - min_overlap[1]))
    y_patches = np.linspace(0, data.shape[2] - patch[1], num_y_patches, dtype=int)

    patch_data = torch.empty((data.shape[0] * num_x_patches * num_y_patches, data.shape[1], patch[0], patch[1]))
    if lbl_flows:
        patch_label = torch.empty((num_x_patches * num_y_patches, 3, patch[0], patch[1]))
    else:
        patch_label = torch.empty((label.shape[0] * num_x_patches * num_y_patches, patch[0], patch[1]))

    for i in range(num_x_patches):
        for j in range(num_y_patches):
            d_patch = data[0, :, y_patches[j]:y_patches[j] + patch[1], x_patches[i]:x_patches[i] + patch[0]]
            patch_data[num_y_patches * i + j] = d_patch
            if lbl_flows:
                l_patch = label[:, y_patches[j]:y_patches[j] + patch[1], x_patches[i]:x_patches[i] + patch[0]]
            else:
                l_patch = label[0, y_patches[j]:y_patches[j] + patch[1], x_patches[i]:x_patches[i] + patch[0]]
            patch_label[num_y_patches * i + j] = l_patch
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
                label_mask[b] = 0
                label_flows1[b] = 0
                label_flows2[b] = 0

            labels[i, 0] = label_mask
            labels[i, 1] = label_flows1
            labels[i, 2] = label_flows2
    else:
        cc = sorted(np.unique(np.concatenate((np.unique(labels[0]), np.unique(labels[:, 0]),
                                              np.unique(labels[-1]), np.unique(labels[:, -1])))))
        for i in range(1, len(cc)):
            labels[labels == cc[i]] = 0
    return labels


# Removes a fraction of labeled samples without cells, such that the ratio of zero to non-zero is approximately 0.1
def remove_empty_label_patches(data, labels):
    num_labels = labels.shape[0]
    nonzero_samples = 0
    for i in range(num_labels):
        if not torch.equal(labels[i], torch.zeros((labels.shape[1:]))):
            nonzero_samples += 1
    num_zeros = num_labels - nonzero_samples
    ratio_zeros = 0.1  # Maximum ratio of zero-label samples to non-zero-label samples
    keep_zeros_percentage = ratio_zeros / (num_zeros / num_labels)  # If > 1, retain all zero-label samples
    if keep_zeros_percentage < 1:
        keep_samples = []
        for i in range(num_labels):
            if not torch.equal(labels[i], torch.zeros((labels.shape[1:]))):
                keep_samples.append(i)
            elif random.random() < keep_zeros_percentage:
                keep_samples.append(i)
        data = data[keep_samples, :]
        labels = labels[keep_samples, :]
    return data, labels


# Creates recombined images after averaging together
def recombine_patches(labels, im_dims, min_overlap):
    num_x_patches = math.ceil((im_dims[1] - min_overlap[0]) / (labels.shape[3] - min_overlap[0]))
    x_patches = np.linspace(0, im_dims[1] - labels.shape[3], num_x_patches, dtype=int)
    num_y_patches = math.ceil((im_dims[0] - min_overlap[1]) / (labels.shape[2] - min_overlap[1]))
    y_patches = np.linspace(0, im_dims[0] - labels.shape[2], num_y_patches, dtype=int)

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


def diam_range(masks):
    masks = np.int32(masks)
    masks = remove_cut_cells(masks)
    x_ranges = []
    y_ranges = []
    diams = []
    uniques = np.unique(masks)[1:]
    for u in uniques:
        inds = np.where(masks == u)
        x_ranges.append(np.amax(inds[1]) - np.amin(inds[1]))
        y_ranges.append(np.amax(inds[0]) - np.amin(inds[0]))
        diams.append(int(math.sqrt(x_ranges[-1] * y_ranges[-1])))
    return diams


def diam_range_3D(masks):
    masks = np.int32(masks)
    x_ranges = []
    y_ranges = []
    z_ranges = []
    diams = []
    uniques = np.unique(masks)[1:]
    for u in uniques:
        inds = np.where(masks == u)
        x_ranges.append(np.amax(inds[2]) - np.amin(inds[2]))
        y_ranges.append(np.amax(inds[1]) - np.amin(inds[1]))
        z_ranges.append(np.amax(inds[0]) - np.amin(inds[0]))
        diams.append(int((x_ranges[-1] * y_ranges[-1] * z_ranges[-1]) ** (1 / 3)))
    return diams


def cell_range(masks, mask_val):
    inds = np.where(masks == mask_val)
    x_range = np.amax(inds[1]) - np.amin(inds[1])
    y_range = np.amax(inds[0]) - np.amin(inds[0])
    return int(math.sqrt(x_range * y_range))
