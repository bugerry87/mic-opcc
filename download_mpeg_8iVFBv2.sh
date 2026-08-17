#!/bin/sh
DATA=${1:-"./data/mpeg/8iVFBv2"}

if [ ! -d "$DATA" ]; then
    mkdir -p $DATA
fi

cd $DATA
wget http://plenodb.jpeg.org/pc/8ilabs/longdress.zip
unzip longdress.zip
wget http://plenodb.jpeg.org/pc/8ilabs/loot.zip
unzip loot.zip
wget http://plenodb.jpeg.org/pc/8ilabs/redandblack.zip
unzip redandblack.zip
wget http://plenodb.jpeg.org/pc/8ilabs/soldier.zip
unzip soldier.zip 