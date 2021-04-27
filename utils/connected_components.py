"""
Takes mono-labelled .tif files and returns with a distinct label for each segmented instance
"""

import os
import numpy as np
from scipy.ndimage.measurements import label
import tifffile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input-folder', help='The directory containing the data to be segmented.')
parser.add_argument('--output-folder', help='The directory which will contain segmented data')
args = parser.parse_args()

os.mkdir(args.output_folder)

os.chdir(args.input_folder)
num_files = len(os.listdir())
for i, file in enumerate(os.listdir()):
    if file.lower().endswith('.tif'):
        raw = tifffile.imread(file)
        raw = np.ceil(raw).astype(np.int16)
        seg, n_comp = label(raw)
        print('{}/{}: File: {}; components: {}'.format(i, num_files, file, n_comp))
        tifffile.imwrite(os.path.join(args.output_folder, file), seg)

