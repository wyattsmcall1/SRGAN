#!/bin/bash
image=$1
width=$2
height=$3

if [-z "$image" || -z "$width" || -z "$height"]
then
    echo "Please call script as ./resize_up.sh image_name width height"
else
    width+="X"
    convert -resize $width$height $image out_$image
fi
