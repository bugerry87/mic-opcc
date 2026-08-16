#!/bin/sh
DATA=${$1:-"./data/mvub"}

if [ ! -d "$DATA" ]; then
    mkdir -p $DATA
fi

cd $DATA
wget http://plenodb.jpeg.org/pc/microsoft/andrew10.zip -o ./
wget http://plenodb.jpeg.org/pc/microsoft/david10.zip -o ./
wget http://plenodb.jpeg.org/pc/microsoft/phil10.zip -o ./
wget http://plenodb.jpeg.org/pc/microsoft/ricardo10.zip -o ./
wget http://plenodb.jpeg.org/pc/microsoft/sarah10.zip -o ./
unzip andrew10.zip
unzip david10.zip 
unzip phil10.zip 
unzip ricardo10.zip
unzip sarah10.zip