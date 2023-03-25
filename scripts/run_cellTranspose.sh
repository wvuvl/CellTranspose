python3 "CellTranspose.py" \
    --learning-rate 0.1 \
    --num-workers 4 \
    --epochs 500 \
    --n-chan 2 \
    --batch-size 8 \
    --eval-batch-size 48 \
    --dataset-name "Cellpose" \
    --flows-available \
    --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results_our_aug/" \
    --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
    --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
    --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
    --calculate-ap


# python3 "CellTranspose.py" \
#     --eval-only \
#     --n-chan 2 \
#     --batch-size 8 \
#     --eval-batch-size 48 \
#     --dataset-name "Cellpose" \
#     --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results_test" \
#     --pretrained-model /mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results/trained_model.pt \
#     --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#     --calculate-ap
