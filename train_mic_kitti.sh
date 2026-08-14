#!/bin/sh
python ./train_mic_pcc.py \
    -X ./samples/kitti_train_index.txt \
    -T ./samples/kitti_test_index.txt \
    -c 3 6 9 12 -P 12 -S 0 3 6 9 12 -k 64 96 128 96 -w 3 5 7 9 -E 16 -n 0 \
    --chunk 1 --range_coder=tfc --qmode=cornered --rotate=z --derotate --grouping=sequential \
    $@