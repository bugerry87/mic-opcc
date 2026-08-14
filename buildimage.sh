#!/bin/sh
DOCKER_BUILDKIT=1 docker build \
    --tag ivslab/mic-opcc \
    -f ./Dockerfile \
    --cache-from ivslab/mic-opcc \
    .