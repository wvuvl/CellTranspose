# CellTranspose: Pre-adaptation sample

Usage of CellTranspose is fairly straightforward: all parameters can be adjusted via command line arguments, which can
be viewed in main.py or by calling "python3 main.py -h"

_A few notes:_

1. This version of CellTranspose only works on 2D data - please omit the "--do-3D" and "--from-3D" arguments and
refrain from using non-2D data (multiple channels are acceptable: see "--n-chan")
2. We intend to redact this in a future version, but please include --val-use-labels or --test-use-labels when
training or testing, respectively, and ignore the --cellpose-model, --size-model, and --refine-prediction args for the
time being. As the original Cellpose predicts cell size every time with a small regression model, we followed their
direction. However, the small amount of annotation we will request from the user will serve as a proxy for cellular size
prediction, and should be more robust.

_Standard hyperparameters:_
--learning-rate
0.2
--momentum
0.9
--weight-decay
1e-5
--batch-size
8
--epochs
500
--median-diams
30
--patch-size
112
--min-overlap
0
--test-overlap
84
--n-chan
2

_Sample run:_

(cellpose) matthew@axon:~/Documents/Cell_Segmentation_Code/domain-adaptive_cellular_instance-seg$ `python3 main.py
--learning-rate 0.005 --momentum 0.9 --weight-decay 1e-5 --batch-size 16 --epochs 30 --median-diams 60 --patch-size
224 --min-overlap 56 --test-overlap 168 --n-chan 2 --dataset-name "Cellpose Generalized Dataset" --results-dir
"~/cellpose_results/Cellpose_results_1" --train-dataset "~/Neuro_Proj1_Data/Cellpose_Dataset/Generalized/train"
--val-dataset "~/Neuro_Proj1_Data/Cellpose_Dataset/Specialized/test" --val-use-labels --test-dataset
"~/Neuro_Proj1_Data/Cellpose_Dataset/Generalized/test" --test-use-labels --calculate-ap`

_Sample eval-only run:_

(cellpose) matthew@axon:~/Documents/Cell_Segmentation_Code/domain-adaptive_cellular_instance-seg$ `python3 main.py
--batch-size 256 --median-diams 30 --patch-size 112 --test-overlap 56 --n-chan 2 --eval-only --dataset-name
"Cellpose Specialized Dataset" --results-dir "~/cellpose_results/Cellpose_results_2" --pretrained-model
"~/cellpose_results/Cellpose_resuls_1/trained_model.pt" --test-dataset
"~/Neuro_Proj1_Data/Cellpose_Dataset/Specialized/test" --test-use-labels --calculate-ap`

## Installation

First, please follow the installation instructions for the original Cellpose work - see their
[installation instructions](https://github.com/MouseLand/cellpose#local-installation) on their repository, and follow
 the steps (1-6). Then, install the required packages for this repository with the cellpose conda environment activated.