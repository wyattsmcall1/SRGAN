#!/bin/bash
#To see what these filters do click here
#http://www.imagemagick.org/Usage/filter/nicolas/

image=$1
filter=$2

#Get width and divide by 4
width=$(identify -format "%w" "$1")> /dev/null
width=$((width/4))
width+="X"

#This is the no filter case
#Filter defaults to Lanczos/Mitchell, so will need to artifically add the "None" filter
if [[ -z "$filter" || "$filter" == "none" ]]
then
    echo "No filter given, using 'point' as default"
    convert -resize $width -filter point $1 out_$1

else
    #Filter types given here:https://imagemagick.org/script/command-line-options.php#filter
    convert -resize $width -filter $filter $image out_$image
fi

