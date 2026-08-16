#!/bin/sh
DATA=${$1:-"./data/mpeg/8iVFBv2"}

if [ ! -d "$DATA" ]; then
    mkdir -p $DATA
fi

cd $DATA
wget http://plenodb.jpeg.org/pc/8ilabs/longdress.zip -o ./
wget http://plenodb.jpeg.org/pc/8ilabs/loot.zip -o ./
wget http://plenodb.jpeg.org/pc/8ilabs/redandblack.zip -o ./
wget http://plenodb.jpeg.org/pc/8ilabs/soldier.zip -o ./
unzip longdress.zip
unzip loot.zip 
unzip redandblack.zip 
unzip soldier.zip 