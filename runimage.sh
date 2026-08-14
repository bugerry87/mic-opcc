#!/bin/sh
docker run \
    --name mic-opcc \
    --mount type=bind,src=./data,dst=/mic-opcc/data \
    ivslab/mic-opcc $@