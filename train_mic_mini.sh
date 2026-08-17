#!/bin/sh
$DOCKER=${DOCKER:-0}
if [ $DOCKER ]; then;
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    LOGS=${LOGS:-"./logs"}

    if [ ! -d "$LOGS" ]; then
        mkdir -p $LOGS
    fi
    docker container rm mic-mini
    docker run \
        --name mic-mini \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        --mount type=bind,src=$LOGS,dst=/home/mic-opcc/logs \
        ivslab/mic-opcc \
        --name mic-mini \
        -X ./samples/mini_index.txt \
        -T ./samples/mini_index.txt \
        -P 12 -S 0 12 -c 12 -k 32 -E 16 \
        -e 10 --steps_per_epoch 100 --range_coder=tfc \
        $@
else
    python ./run_mic_pcc.py \
        --name mic-mini \
        -X ./samples/mini_index.txt \
        -T ./samples/mini_index.txt \
        -P 12 -S 0 12 -c 12 -k 32 -E 16 \
        -e 10 --steps_per_epoch 100 --range_coder=nrc --cpu \
        $@
fi