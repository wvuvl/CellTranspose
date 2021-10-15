import numpy as np
import math


def diam_range(masks, percentile=75):
    masks = np.int32(masks)  # Cast as np for convenience
    x_ranges = []
    y_ranges = []
    diams = []
    uniques = np.unique(masks)[1:]
    for u in uniques:
        inds = np.where(masks == u)
        x_ranges.append(np.amax(inds[1]) - np.amin(inds[1]))
        y_ranges.append(np.amax(inds[0]) - np.amin(inds[0]))
        diams.append(int(math.sqrt(x_ranges[-1] * y_ranges[-1])))
    return np.percentile(np.array(diams), percentile)
