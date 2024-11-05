#!/bin/bash
PYTORCH_IMG_DIR=../pytorch-yolo-v3-master/imgs
VAL_IMG_DIR=../data/VOC2012/val

#1) Downsample list of val (HR) Image with filter
#mkdir tmp_1
#for filename in $VAL_IMG_DIR/*.jpg; do
#    filebase=$(basename $filename)
#    convert -resize 125x -filter point $filename tmp_1/out_$filebase #Use 125x because upscale by 4
#done
#2) Run through SRGAN (SR)
#mkdir tmp_2
#for filename in ./tmp_1/*.jpg; do
#    python ../test_image.py --image_name $filename --script_mode Y
#done


#for filename in $PYTORCH_IMG_DIR/*.jpg; do
#    echo "$filename"
#    bash ./resize_down.sh "$filename" "catrom" ;
#done

#rm tmp -rf
