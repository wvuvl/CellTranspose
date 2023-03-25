import math
import torch
import cv2
import numpy as np
from tqdm import tqdm
from cellpose_src.dynamics import masks_to_flows, follow_flows, get_masks, compute_masks
from cellpose_src.utils import fill_holes_and_remove_small_masks
from cellpose_src.transforms import _taper_mask
import random
import torchvision.transforms.functional as TF

# cellpose source, changed for celltranspose
def random_rotate_and_resize(X, Y=None, scale_range=1., xy = (224,224), 
                             do_flip=True, rescale=None, unet=False):
    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)

        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]

        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow]. 
            If unet, second channel is dist_to_bound.

        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()

        xy: tuple, int (optional, default (224,224))
            size of transformed images to return

        do_flip: bool (optional, default True)
            whether or not to flip images horizontally

        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations

        unet: bool (optional, default False)

        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]

        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]

        scale: array, float
            amount each image was resized by

    """
    scale_range = max(0, min(2, float(scale_range)))

    if X.ndim>2:
        nchan = X.shape[0]
    else:
        nchan = 1
        
    imgi  = np.zeros((nchan, xy[0], xy[1]), np.float32)

    
    if Y is not None:
        if Y.ndim>2:
            nt = Y.shape[0]
        else:
            nt = 1
        lbl = np.zeros((nt, xy[0], xy[1]), np.float32)

   
    
    Ly, Lx = X.shape[-2:]

    # generate random augmentation parameters
    flip = np.random.rand()>.5
    theta = np.random.rand() * np.pi * 2
    scale = (1-scale_range/2) + scale_range * np.random.rand()
    if rescale is not None:
        scale*= 1. / rescale
    dxy = np.maximum(0, np.array([Lx*scale-xy[1],Ly*scale-xy[0]]))
    dxy = (np.random.rand(2,) - .5) * dxy

    # create affine transform
    cc = np.array([Lx/2, Ly/2])
    cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
    pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
    pts2 = np.float32([cc1,
            cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
            cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
    M = cv2.getAffineTransform(pts1,pts2)

    img = X.copy()
    if Y is not None:
        labels = Y.copy()
        if labels.ndim<3:
            labels = labels[np.newaxis,:,:]

    if flip and do_flip:
        img = img[..., ::-1]
        if Y is not None:
            labels = labels[..., ::-1]
            if nt > 1 and not unet:
                labels[2] = -labels[2]

    for k in range(nchan):
        I = cv2.warpAffine(img[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
        imgi[k] = I

    if Y is not None:
        for k in range(nt):
            if k==0:
                lbl[k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=cv2.INTER_NEAREST)
            else:
                lbl[k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)

        if nt > 1 and not unet:
            v1 = lbl[2].copy()
            v2 = lbl[1].copy()
            lbl[1] = (-v1 * np.sin(-theta) + v2*np.cos(-theta))
            lbl[2] = (v1 * np.cos(-theta) + v2*np.sin(-theta))

    return imgi, lbl


def reformat(x, n_chan=1, do_3D=False):
    """
    Reformats raw input data with the following expected output:
    If 2-D ->  ndarray with shape [channels, y_dim, x_dim] or [channels, z_dim, y_dim, x_dim]
    """
    if do_3D:
        if len(x.shape) == 3:
            x = x[np.newaxis,:,:,:]
        
        elif len(x.shape) == 4:
            if x.shape[3] > x.shape[0]:
                x = x.transpose(1, 2, 3, 0)
            info_chans = [len(np.unique(x[:, :, :, i])) > 1 for i in range(x.shape[3])]
            x = x[:, :, :, info_chans]
            x = np.transpose(x, (3, 0, 1, 2))[:n_chan]  # Remove any additional channels

            
        # Concatenate copies of other channels if image has fewer than the specified number of channels
        if x.shape[0] < n_chan:
            x = np.tile(x, (math.ceil(n_chan/x.shape[0]), 1, 1, 1))
            x = x[:n_chan]
    else:    
        if len(x.shape) == 2:
            x = x[np.newaxis,:,:]
        elif len(x.shape) == 3:
            if x.shape[2] > x.shape[0]:
                x = x.transpose(1, 2, 0)
            info_chans = [len(np.unique(x[:, :, i])) > 1 for i in range(x.shape[2])]
            x = x[:, :, info_chans]
            x = np.transpose(x, (2, 0, 1))[:n_chan]  # Remove any additional channels
        # Concatenate copies of other channels if image has fewer than the specified number of channels
        if x.shape[0] < n_chan:
            x = np.tile(x, (math.ceil(n_chan/x.shape[0]), 1, 1))
            x = x[:n_chan]
    return x

# changed cellpose src code for nchan, y, x format
def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False):
    """ resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [nchan x Y x X ] or [nchan x Lz x Y x X] or [Lz x Y x X]

    Ly: int, optional

    Lx: int, optional

    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used

    interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)

    Returns
    --------------

    imgs: ND-array 
        image of size [nchan x Ly x Lx] or [nchan x Lz x Ly x Lx]

    """
    if Ly is None and rsz is None:        
        raise ValueError('must give size to resize to or factor to use for resizing')

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        # no need to check for channels as the channles are always in the front at index 0    
        Ly = int(img0.shape[-2] * rsz[-2])
        Lx = int(img0.shape[-1] * rsz[-1])
     
    
    # no_channels useful for z-stacks, so the third dimension is not treated as a channel
    # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but 
    if (img0.ndim>2 and no_channels) or (img0.ndim==4 and not no_channels):
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            # for 4D data, consider the input stack is Z x ch x Y x X format
            imgs = np.zeros((img0.shape[0],img0.shape[1], Ly, Lx), np.float32)
        for i,img in enumerate(img0):
            imgs[i] = cv2.resize(img.transpose(1, 2, 0), (Lx, Ly), interpolation=interpolation).transpose(2,0,1)
    else:
        # 2D image with channels in the front
        if img0.shape[0] == 1:
            imgs = cv2.resize(img0.transpose(1, 2, 0), (Lx, Ly), interpolation=interpolation)[np.newaxis,:,:]
        else:
            imgs = cv2.resize(img0.transpose(1, 2, 0), (Lx, Ly), interpolation=interpolation).transpose(2,0,1)
        
    return imgs


def normalize1stto99th(x):
    """
    Normalize each channel of input image so that 0.0 corresponds to 1st percentile and 1.0 corresponds to 99th g-
    Made to mimic Cellpose's normalization implementation
    """
    sample = x.clone() if torch.is_tensor(x) else x.copy()
    for chan in range(len(sample)):
        if len(torch.unique(sample[chan]) if torch.is_tensor(sample) else np.unique(sample[chan])) != 1:
            sample[chan] = (sample[chan] - np.percentile(sample[chan], 1))\
                           / (np.percentile(sample[chan], 99) - np.percentile(sample[chan], 1))
    return sample

def train_generate_rand_crop(data, label, crop=112, min_masks=1):
    while 1:
        x_max = data.shape[2] - crop
        y_max = data.shape[1] - crop
            
        i = random.randint(0, x_max)
        j = random.randint(0, y_max)
        
        d_patch = data[:, j:j + crop, i:i + crop]
        l_patch = label[:, j:j + crop, i:i + crop]
        
        if len(np.unique(l_patch)[1:])>=min_masks: break
        else: print(f'Masks in this crop found less than {min_masks}')
    
    return d_patch, l_patch

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
    
    cellprob = flows[0]
    dP = flows[1:]
    
    p = follow_flows(dP * (cellprob > cellprob_threshold) / 5., niter, interp, use_gpu)
    masks = get_masks(p, iscell=(cellprob > cellprob_threshold), flows=dP, threshold=flow_threshold)
    masks = fill_holes_and_remove_small_masks(masks, min_size=min_size)
    
    return masks


def followflows3D(dP, cellprob, cell_metric=12, device=None):
    """
    Combines follow_flows, get_masks, and fill_holes_and_remove_small_masks from Cellpose implementation
    """
    niter = 200; interp = True; use_gpu = True; cellprob_threshold = 0.0; flow_threshold = 0.4
    # Smallest size of calculated mask allowed; masks lower than this value will be removed
    min_size = (cell_metric ** 3) / 125
    masks = compute_masks(dP, cellprob, niter=niter, interp=interp, use_gpu=use_gpu, mask_threshold=cellprob_threshold,
                          flow_threshold=flow_threshold, min_size=min_size, device=device)
    return masks
                   
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


# in:  Ch x Y x X
# out: padded Ch x Y x X
def padding_2D(sample_data,  patch_size):
    unpadded_dims = (sample_data.shape[1], sample_data.shape[2])
    if (sample_data.shape[1] < patch_size or sample_data.shape[2] < patch_size):
        
        sd = np.zeros((sample_data.shape[0], max(patch_size, sample_data.shape[1]), max(patch_size, sample_data.shape[2])))
        
        set_corner = (max(0, (patch_size - sample_data.shape[1]) // 2), max(0, (patch_size - sample_data.shape[2]) // 2))
        
        sd[:, set_corner[0]:set_corner[0] + sample_data.shape[1],
            set_corner[1]:set_corner[1] + sample_data.shape[2]] = sample_data
        
        resized_dims = (sample_data.shape[1], sample_data.shape[2])
        
        return sd, set_corner, unpadded_dims, resized_dims
    else:
        return sample_data, [0,0], unpadded_dims, unpadded_dims
    
# in:  Z x Ch x Y x X
# out: padded Z x Ch x Y x X
def padding_3D(sample_data,  patch_size):
    unpadded_dims = (sample_data.shape[2], sample_data.shape[3])
    if sample_data.shape[2] < patch_size or sample_data.shape[3] < patch_size:
        
        sd = np.zeros((sample_data.shape[0], sample_data.shape[1], max(patch_size, sample_data.shape[2]),
                    max(patch_size, sample_data.shape[3])))
        
        set_corner = (max(0, (patch_size - sample_data.shape[2]) // 2),
                        max(0, (patch_size - sample_data.shape[3]) // 2))
        
        sd[:, :, set_corner[0]:set_corner[0] + sample_data.shape[2],
            set_corner[1]:set_corner[1] + sample_data.shape[3]] = sample_data
        
        resized_dims = (sample_data.shape[2], sample_data.shape[3])
        
        return sd, set_corner, unpadded_dims, resized_dims
    else:
        return sample_data, [0,0], unpadded_dims, unpadded_dims
    
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


def cell_range(masks, mask_val):
    inds = np.where(masks == mask_val)
    x_range = np.amax(inds[1]) - np.amin(inds[1])
    y_range = np.amax(inds[0]) - np.amin(inds[0])
    return int(math.sqrt(x_range * y_range))

def diameters(masks):
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5) / 2
    return md, counts**0.5

def calc_median_dim(masks):
    median_list = []
    count_list = []

    for mask in tqdm(masks):
        md, counts = diameters(mask)
        median_list = np.append(median_list, md)
        count_list = np.append(count_list, counts)
        
    return math.ceil(np.median(median_list))