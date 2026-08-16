#!/bin/sh
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
    ivslab/mic-opcc $@