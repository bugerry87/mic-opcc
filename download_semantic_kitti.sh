#!/bin/sh
DATA=${$1:-"./data/kitti"}

if [ ! -d "$DATA" ]; then
    mkdir -p $DATA
fi

cd $DATA
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip -o ./
unzip data_odometry_velodyne.zip 