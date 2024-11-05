for filename in ../pytorch-yolo-v3-master/imgs/*.jpg; do
    echo "$filename"
    bash ./resize_down.sh "$filename" "catrom" ;
done
