import os
import tifffile as tiff
import numpy as np
from deepcell.applications import Mesmer
from tqdm import tqdm


path = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/full_data_lbl_0/test/data'

data = []
im_list = os.listdir(path)
for im in tqdm(im_list):
    img = tiff.imread(os.path.join(path,im))
    
    data.append(img.transpose(1,2,0))

data = np.array(data)
print(data.shape)
    
app = Mesmer()

print('Training Resolution:', app.model_mpp, 'microns per pixel')
for i in range(1,5):
    dest = f'/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/tissuenet_v1.0/tissuenet_1.0/mesmer_results/{str(i)}/tiff_results'
    if os.path.exists(dest) == False: os.makedirs(dest)
    segmentation_predictions = app.predict(data, image_mpp=0.5, compartment='whole-cell')

    for index, i in enumerate(segmentation_predictions):
        tiff.imwrite(os.path.join(dest,im_list[index]),i.transpose(2,0,1)[0])