# CellTranspose: Few-shot Domain Adaptation for Cellular Instance Segmentation

<div align="center">
[![Static Badge](https://img.shields.io/badge/WACV-2023-blue?link=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FWACV2023%2Fpapers%2FKeaton_CellTranspose_Few-Shot_Domain_Adaptation_for_Cellular_Instance_Segmentation_WACV_2023_paper.pdf)](https://openaccess.thecvf.com/content/WACV2023/papers/Keaton_CellTranspose_Few-Shot_Domain_Adaptation_for_Cellular_Instance_Segmentation_WACV_2023_paper.pdf)
[![Static Badge](https://img.shields.io/badge/Project_website-green?link=https%3A%2F%2Fgithub.com%2Fwvuvl%2FCellTranspose)](https://github.com/wvuvl/CellTranspose)

[Matthew Keaton](https://www.linkedin.com/in/matthew-keaton/),
[Ram Zaveri](https://ramzaveri.com/),
[Gianfranco Doretto](https://vision.csee.wvu.edu/people/gianfranco-doretto/)

WACV 2023
</div>

This is the official repository of [CellTranspose](https://openaccess.thecvf.com/content/WACV2023/papers/Keaton_CellTranspose_Few-Shot_Domain_Adaptation_for_Cellular_Instance_Segmentation_WACV_2023_paper.pdf), an adaptation-based framework for achieving accurate automated cellular instance segmentation on any type of tissue data using a minimal amount of expert annotations (typically 3 to 5 cell instances). This work is built on [Cellpose](https://github.com/MouseLand/cellpose), a state-of-the-art method developed by Janelia for cellular instance segmentation without user annotation.

This code base can be utilized in a number of ways, including for initial model training, adaptation, and evaluation on both 2-D and 3-D data.

All code examples below assume the default parameters to be used (contained within <code>CellTranspose.py</code> – if new default
values are needed, they can be updated here. Additionally, these can be passed in as parameter arguments).

## Dataset structure
This code has been implemented to handle both png and tiff data. It is expected that each dataset is comprised of two folders, named "data" and "labels," set in the root directory. All images should be set in the "data" folder while all masks should be set in the "labels" folder; any subdirectories will be ignored. Visually, the dataset should be organized as follows:

    - </dataset/folder>
        - /"data"
            - data_1.tiff
            ...
            - data_n.tiff
        - /"labels"
            - lbl_1.tiff
            ...
            - lbl_n.tiff

## Training an Initial Model/Direct Training

In order to produce a model prior to adaptation, the following command template should be utilized.

<code> python3 CellTranspose.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--train-dataset <\path\to\training\dataset> [--val-dataset <\path\to\validation\dataset>] --test-dataset
<\path\to\test\dataset> [--calculate-ap]</code>

For 3-D training data, dataset containing 2-D slices should be passed in and for 3-D testing data, the following template should be used.

<code> python3 CellTranspose.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--train-dataset <\path\to\training\dataset> [--val-dataset <\path\to\validation\dataset>]
--test-dataset <\path\to\test\dataset> --test-from-3D</code>

Note that due to memory constraints, AP calculation cannot occur for 3-D data. Instead, once the initial code has been run, it should be followed by <code>Calc_AP.py</code>.

It should also be noted that training can be broken into separate training and evaluation steps (via
<code>--train-only</code> and <code>--eval-only</code>), but since this is not likely to be used, we only show options
for the full training/evaluation procedure here.

## Adapting the Model

Adaptation, specifically only for training and not evaluation, should be completed using the following code:

<code> python3 CellTranspose.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--pretrained-model <\path\to\pretrained\model> --do-adaptation --train-only --train-dataset <\path\to\training\dataset>
--target-dataset <\path\to\target\dataset> [--val-dataset <\path\to\validation\dataset>]</code>

## Performing Segmentation Using a Pretrained Model

With a newly trained/adapted model, segmentations can be obtained directly:

<code> python3 CellTranspose.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--pretrained-model <\path\to\pretrained\model> --eval-only --test-dataset <\path\to\test\dataset> [--calculate-ap]</code>

### Additional Parameters

Each of the above processes can be augmented using additional parameters, which can be found in <code>CellTranspose.py</code>.
These include saving the training dataset in .npy format in order to save time on future experiments, loading such a
dataset in, and performing preprocessing at each epoch instead of only once (this may slightly increase accuracy, but
is computationally expensive).

### This code has been completed for particular needs

As a final remark, the code as is currently does not implement certain components that may be useful in the future.
Some additions that were originally considered (such as a 3-D volumetric approach) should be fairly easy to incorporate
into the codebase if it is decided that these additions would be valuable.
