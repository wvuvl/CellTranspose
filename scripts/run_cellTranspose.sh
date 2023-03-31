for version in 01 02 03 04 05
    do
    for use_chan in 0 1
    do
        python3 "CellTranspose.py" \
            --learning-rate 0.01 \
            --num-workers 4 \
            --epochs 500 \
            --n-chan 1 \
            --chan-use $use_chan \
            --batch-size 16 \
            --eval-batch-size 48 \
            --dataset-name "Cellpose" \
            --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_1_$use_chan/$version-celltranspose_model/" \
            --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
            --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
            --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
            --calculate-ap
    done
    
    
    python3 "CellTranspose.py" \
        --learning-rate 0.01 \
        --num-workers 4 \
        --epochs 500 \
        --n-chan 2 \
        --batch-size 16 \
        --eval-batch-size 48 \
        --dataset-name "Cellpose" \
        --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/chan_2/$version-celltranspose_model/" \
        --train-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/train" \
        --val-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
        --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
        --calculate-ap
    
done

# python3 "CellTranspose.py" \
#     --min-overlap 0.1 \
#     --eval-only \
#     --n-chan 2 \
#     --chan-use 0 \
#     --eval-batch-size 48 \
#     --median-diams 30 \
#     --dataset-name "Cellpose" \
#     --results-dir "/mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results_500_224_12_test" \
#     --pretrained-model /mnt/12F9CADD61CB0337/syGlass/inquiry_test/cp_data_results_500_224/trained_model.pt \
#     --test-dataset "/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/generalist_cellpose/Generalized/test" \
#     --calculate-ap
