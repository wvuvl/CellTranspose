# for version in 01 02 03 04 05
#     do
#     python3 "CellTranspose.py" \
#         --learning-rate 0.01 \
#         --num-workers 4 \
#         --epochs 500 \
#         --n-chan 1 \
#         --batch-size 16 \
#         --eval-batch-size 48 \
#         --dataset-name "Cellpose" \
#         --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_1/$version-celltranspose_model/" \
#         --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#         --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#         --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#         --calculate-ap
    
#     python3 "CellTranspose.py" \
#         --learning-rate 0.01 \
#         --num-workers 4 \
#         --epochs 500 \
#         --n-chan 2 \
#         --batch-size 16 \
#         --eval-batch-size 48 \
#         --dataset-name "Cellpose" \
#         --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_2/$version-celltranspose_model/" \
#         --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#         --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#         --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#         --calculate-ap
    
# done

python3 "CellTranspose.py" \
        --learning-rate 0.01 \
        --num-workers 4 \
        --epochs 500 \
        --n-chan 1 \
        --batch-size 16 \
        --eval-batch-size 48 \
        --dataset-name "Cellpose" \
        --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_1/celltranspose_model/" \
        --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
        --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
        --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
        --calculate-ap