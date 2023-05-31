# for version in 01 02 03 04 05
# do
#     python3 "CellTranspose.py" \
#             --learning-rate 0.01 \
#             --num-workers 4 \
#             --epochs 1000 \
#             --n-chan 2 \
#             --batch-size 8 \
#             --eval-batch-size 48 \
#             --dataset-name "Cellpose" \
#             --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_2_1000_112_8/$version-celltranspose_model/" \
#             --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#             --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --calculate-ap

#     python3 "CellTranspose.py" \
#             --learning-rate 0.01 \
#             --num-workers 4 \
#             --epochs 1000 \
#             --n-chan 1 \
#             --batch-size 8 \
#             --eval-batch-size 48 \
#             --dataset-name "Cellpose" \
#             --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_1_1000_112_8/$version-celltranspose_model/" \
#             --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#             --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --calculate-ap
# done


# python3 "CellTranspose.py" \
#             --learning-rate 0.01 \
#             --num-workers 4 \
#             --epochs 500 \
#             --n-chan 2 \
#             --batch-size 8 \
#             --eval-batch-size 48 \
#             --dataset-name "Cellpose" \
#             --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_2_500_112_8/celltranspose_model_with_pix_contrast_with_0" \
#             --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#             --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --calculate-ap

# python3 "CellTranspose.py" \
#             --learning-rate 0.01 \
#             --num-workers 4 \
#             --epochs 500 \
#             --n-chan 1 \
#             --batch-size 8 \
#             --eval-batch-size 48 \
#             --dataset-name "Cellpose" \
#             --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_1_500_112_8/celltranspose_model_with_pix_contrast_with_0/" \
#             --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#             --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --calculate-ap

# python3 "CellTranspose.py" \
#             --learning-rate 0.01 \
#             --num-workers 4 \
#             --epochs 1000 \
#             --n-chan 2 \
#             --batch-size 8 \
#             --eval-batch-size 48 \
#             --dataset-name "Cellpose" \
#             --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_2_1000_112_8/celltranspose_model_with_pix_contrast_with_0" \
#             --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#             --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --calculate-ap

# python3 "CellTranspose.py" \
#             --learning-rate 0.01 \
#             --num-workers 4 \
#             --epochs 1000 \
#             --n-chan 1 \
#             --batch-size 8 \
#             --eval-batch-size 48 \
#             --dataset-name "Cellpose" \
#             --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_1_1000_112_8/celltranspose_model_with_pix_contrast_with_0/" \
#             --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#             --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#             --calculate-ap

python3 "CellTranspose.py" \
            --learning-rate 0.01 \
            --num-workers 4 \
            --epochs 1000 \
            --n-chan 2 \
            --batch-size 8 \
            --eval-batch-size 48 \
            --dataset-name "BBBC039" \
            --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_1_1000_112_8/celltranspose_model_with_pix_contrast_pretrained_BBBC039/" \
            --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/BBBC039/BBBC039/split/train" \
            --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/BBBC039/BBBC039/split/test" \
            --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/BBBC039/BBBC039/split/test" \
            --calculate-ap