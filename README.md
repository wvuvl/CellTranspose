# CellTranspose: Pre-adaptation sample

Usage of CellTranspose is fairly straightforward: all parameters can be adjusted via command line arguments, which can
be viewed in main.py or by calling "python3 main.py -h"

_A few notes:_

1. This version of CellTranspose only works on 2D data - please omit the "--do-3D" and "--from-3D" arguments and
refrain from using non-2D data (multiple channels are acceptable: see "--n-chan")
2. We intend to redact this in a future version, but please include --val-use-labels or --test-use-labels when
training or testing, respectively, for the time being. As the original Cellpose predicts cell size every time with a
small regression model, we followed their direction. However, the small amount of annotation we will request from the
user will serve as a proxy for cellular size prediction, and should be more robust.

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

## Installation

First, please follow the installation instructions for the original Cellpose work - see their
[installation instructions](https://github.com/MouseLand/cellpose#local-installation) on their repository, and follow
 the steps (1-6). Then, clone this repository and install the required packages.