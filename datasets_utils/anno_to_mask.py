import numpy as np
import os
from glob import glob

def remove_overlaps(masks, medians, overlap_threshold=0.75):
    """ replace overlapping mask pixels with mask id of closest mask
        if mask fully within another mask, remove it
        masks = Nmasks x Ly x Lx
    """
    cellpix = masks.sum(axis=0)
    igood = np.ones(masks.shape[0], 'bool')
    for i in masks.sum(axis=(1,2)).argsort():
        npix = float(masks[i].sum())
        noverlap = float(masks[i][cellpix > 1].sum())
        if noverlap / npix >= overlap_threshold:
            igood[i] = False
            cellpix[masks[i]>0] -= 1
            #print(cellpix.min())
    print(f'removing {(~igood).sum()} masks')
    masks = masks[igood]
    medians = medians[igood]
    cellpix = masks.sum(axis=0)
    overlaps = np.array(np.nonzero(cellpix>1.0)).T
    dists = ((overlaps[:,:,np.newaxis] - medians.T)**2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:,0], overlaps[:,1]] = 0
    masks[tocell, overlaps[:,0], overlaps[:,1]] = 1

    # labels should be 1 to mask.shape[0]
    masks = masks.astype(int) * np.arange(1,masks.shape[0]+1,1,int)[:,np.newaxis,np.newaxis]
    masks = masks.sum(axis=0)
    return masks

def ann_to_masks(annotations, anns, overlap_threshold=0.75):
    """ list of coco-format annotations with masks to single image"""
    masks = []
    k=0
    medians=[]
    for ann in anns:
        mask = annotations.annToMask(ann)
        masks.append(mask)
        ypix, xpix = mask.nonzero()
        medians.append(np.array([ypix.mean(), xpix.mean()]))
        k+=1
    masks=np.array(masks).astype('int')
    medians=np.array(medians)
    masks = remove_overlaps(masks, medians, overlap_threshold=overlap_threshold)
    return masks

def livecell_ann_to_masks(img_dir, annotation_file):
    from pycocotools.coco import COCO
    from tifffile import imsave
    img_dir_classes = glob(img_dir + '*/')
    classes = [img_dir_class.split(os.sep)[-2] for img_dir_class in img_dir_classes]
    print(classes)

    train_files = []
    train_class_files = []
    for cclass, img_dir_class in zip(classes, img_dir_classes):
        train_files.extend(glob(img_dir_class + '*.tif'))
        train_class_files.append(glob(img_dir_class + '*.tif'))

    annotations = COCO(annotation_file)
    imgIds = list(annotations.imgs.keys())

    for train_class_file in train_class_files:
        for i in range(len(train_class_file)):
            filename = train_class_file[i]
            fname = os.path.split(filename)[-1]
            loc = np.array([annotations.imgs[imgId]['file_name']==fname for imgId in imgIds]).nonzero()[0]
            if len(loc) > 0:
                imgId = imgIds[loc[0]]
                annIds = annotations.getAnnIds(imgIds=[imgId], iscrowd=None)
                anns = annotations.loadAnns(annIds)
                masks = ann_to_masks(annotations, anns, overlap_threshold=0.75)
                masks = masks.astype(np.uint16)
                maskname = os.path.splitext(filename)[0] + '_masks.tif'
                imsave(maskname, masks)
                print(f'saved masks at {maskname}')

img_dir = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/LiveCell/images/livecell_test_images'
annotation_file = '/media/ramzaveri/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/LiveCell/images/livecell_coco_test.json'
livecell_ann_to_masks(img_dir, annotation_file)