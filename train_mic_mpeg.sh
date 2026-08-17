#!/bin/sh
$DOCKER=${DOCKER:-0}
if [ $DOCKER ]; then;
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    DATA=${DATA:-"./data"}
    LOGS=${LOGS:-"./logs"}

    if [ ! -d "$LOGS" ]; then
        mkdir -p $LOGS
    fi
    docker container rm mic-mpeg
    docker run \
        --name mic-mpeg \
        --gpus "device=$CUDA_VISIBLE_DEVICES" \
        --mount type=bind,src=$DATA,dst=/home/mic-opcc/data \
        --mount type=bind,src=$LOGS,dst=/home/mic-opcc/logs \
        ivslab/mic-opcc \
        --name mic-mpeg \
        -X ./samples/mpeg_train_index.txt \
        -T ./samples/mpeg_test_index.txt \
        -e 10 -c 1 2 3 4 5 -P 10 -S 0 2 4 6 8 10 -k 96 -w 5 -E 0 -n 0 \
        --chunk=10 --xformat=ply --xshape -1 3 --scale 1.0 \
        --range_coder=tfc --qmode=cornered --rotate=z --derotate --grouping=sequential \
        $@
else
    python ./run_mic_pcc.py \
        --name mic-mpeg \
        -X ./samples/mpeg_train_index.txt \
        -T ./samples/mpeg_test_index.txt \
        -e 10 -c 1 2 3 4 5 -P 10 -S 0 2 4 6 8 10 -k 96 -w 5 -E 0 -n 0 \
        --chunk=10 --xformat=ply --xshape -1 3 --scale 1.0 \
        --range_coder=tfc --qmode=cornered --rotate=z --derotate --grouping=sequential \
        $@
fi
