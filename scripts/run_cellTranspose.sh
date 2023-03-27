# python3 "CellTranspose.py" \
#     --learning-rate 0.01 \
#     --num-workers 4 \
#     --epochs 500 \
#     --n-chan 2 \
#     --batch-size 16 \
#     --eval-batch-size 48 \
#     --dataset-name "Cellpose" \
#     --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results_500_cp_diam_min_12/" \
#     --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
#     --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#     --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#     --calculate-ap

python3 "CellTranspose.py" \
    --min-overlap 0.1 \
    --eval-only \
    --n-chan 2 \
    --eval-batch-size 48 \
    --median-diams 30 \
    --dataset-name "Cellpose" \
    --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results_500_our_diam_min_12_test" \
    --pretrained-model /mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results_500_our_diam_min_12/trained_model.pt \
    --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
    --calculate-ap
