#Few-Shot Domain-Adaptive Cell Segmentation with Cellpose

This code base can be utilized in a number of ways, including for initial model training, adaptation, and evaluation on both 2-D and 3-D data.

All code examples below assume the default parameters to be used (contained within <code>main.py</code> – if new default
values are needed, they can be updated here. Additionally, these can be passed in as parameter arguments).

##Dataset structure
This code has been implemented to handle both png and tiff data. It is expected that each dataset is comprised of two folders, named "data" and "labels", set in the root directory. All images should be set in the "data" folder while all masks should be set in the "labels" folder; any subdirectories will be ignored. Visually, the dataset should be organized as follows:

    - </root/folder>
        - /data
            - vol1.tiff
            ...
            - voln.tiff
        - /labels
            - lbl1.tiff
            ...
            - lbln.tiff

##Training an Initial Model/Direct Training

In order to produce a model prior to adaptation, the following command template should be utilized.

<code> python3 main.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--train-dataset <\path\to\training\dataset> [--val-dataset <\path\to\validation\dataset>] --test-dataset
<\path\to\test\dataset> [--calculate-ap]</code>

For 3-D training data: the dataset containing 2-D slices should be passed in
For 3-D testing data: the following template should be used.

<code> python3 main.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--train-dataset <\path\to\training\dataset> [--val-dataset <\path\to\validation\dataset>]
--test-dataset <\path\to\test\dataset> --test-from-3D</code>

Note that due to memory constraints, AP calculation cannot occur for 3-D data. Instead, once the initial code has been run, it should be followed by <code>Calc_AP.py</code>.

It should also be noted that training can be broken into separate training and evaluation steps (via
<code>--train-only</code> and <code>--eval-only</code>), but since this is not likely to be used, we only show options
for the full training/evaluation procedure here.

##Adapting the Model

Adaptation, specifically only for training and not evaluation, should be completed using the following code:

<code> python3 main.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--pretrained-model <\path\to\pretrained\model> --do-adaptation --train-only --train-dataset <\path\to\training\dataset>
--target-dataset <\path\to\target\dataset> [--val-dataset <\path\to\validation\dataset>]</code>

##Performing Segmentation Using a Pretrained Model

With a newly trained/adapted model, segmentations can be obtained directly:

<code> python3 main.py --dataset-name <"dataset_name"> --results-dir <\path\to\results\dir>
--pretrained-model <\path\to\pretrained\model> --eval-only --test-dataset <\path\to\test\dataset> [--calculate-ap]</code>

###Additional Parameters

Each of the above processes can be augmented using additional parameters, which can be found in <code>main.py</code>.
These include saving the training dataset in .npy format in order to save time on future experiments, loading such a
dataset in, and performing preprocessing at each epoch instead of only once (this may slightly increase accuracy, but
is computationally expensive).

###This code has been completed for particular needs

As a final remark, the code as is currently does not implement certain components that may be useful in the future.
Some additions that were originally considered (such as a 3-D volumetric approach) should be fairly easy to incorporate
into the codebase if it is decided that these additions would be valuable.