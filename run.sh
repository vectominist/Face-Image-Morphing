#!/bin/bash

img_1=$1
img_2=$2
shape_pr=~/Downloads/shape_predictor_68_face_landmarks.dat
ratio=$3
out_path=$4

python3 main.py \
    --image_1 $img_1 \
    --image_2 $img_2 \
    --shape-predictor "$shape_pr" \
    --ratio $ratio
    # --output $out_path
