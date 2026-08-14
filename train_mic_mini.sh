#!/bin/sh
python ./train_mic_pcc.py \
    -X ./samples/mini_index.txt \
    -T ./samples/mini_index.txt \
    -P 12 -S 0 4 8 12 -c 4 8 12 -k 32 -E 16 \
    -e 10 --steps_per_epoch 100 --range_coder=nrc --cpu \
    $@