#!/bin/bash

img_1=$1
img_2=$2
img_3=$3
shape_pr=~/Downloads/shape_predictor_68_face_landmarks.dat
out_path=$4
name=$5

python3 main_3.py \
    --image_1 $img_1 \
    --image_2 $img_2 \
    --image_3 $img_3 \
    --shape-predictor "$shape_pr" \
    --ratio_1 0.3 \
    --ratio_2 0.4 \
    --output $out_path \
    --name $name
