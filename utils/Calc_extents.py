import numpy as np
import math


def diam_range(masks, percentile=75):
    masks = np.int32(masks)  # Cast as np for convenience
    diams = []
    uniques = np.unique(masks)[1:]
    for u in uniques:
        diams.append(cell_range(masks, u))
    return np.percentile(np.array(diams), percentile)


def cell_range(masks, mask_val):
    inds = np.where(masks == mask_val)
    x_range = np.amax(inds[1]) - np.amin(inds[1])
    y_range = np.amax(inds[0]) - np.amin(inds[0])
    return int(math.sqrt(x_range * y_range))
