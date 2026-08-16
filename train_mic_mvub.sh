#!/bin/sh
$DOCKER=${DOCKER:-0}
if [ $DOCKER ]; then;
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    DATA=${DATA:-"./data"}
    LOGS=${LOGS:-"./logs"}

    if [ ! -d "$LOGS" ]; then
        mkdir -p $LOGS
    fi
    docker container rm mic-opcc
    docker run \
        --name mic-opcc \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        --mount type=bind,src=$DATA,dst=/home/mic-opcc/data \
        --mount type=bind,src=$LOGS,dst=/home/mic-opcc/logs \
        ivslab/mic-opcc \
        --name mic-mvub \
        -X ./samples/kitti_train_index.txt \
        -T ./samples/kitti_test_index.txt \
        -c 3 6 9 12 -P 12 -S 0 3 6 9 12 -k 64 96 128 96 -w 3 5 7 9 -E 16 -n 0 \
        --chunk 1 --range_coder=tfc --qmode=cornered --rotate=z --derotate --grouping=sequential \
        $@
else
    python ./run_mic_pcc.py \
        --name mic-mvub \
        -X ./samples/kitti_train_index.txt \
        -T ./samples/kitti_test_index.txt \
        -c 3 6 9 12 -P 12 -S 0 3 6 9 12 -k 64 96 128 96 -w 3 5 7 9 -E 16 -n 0 \
        --chunk 1 --range_coder=tfc --qmode=cornered --rotate=z --derotate --grouping=sequential \
        $@
fi
