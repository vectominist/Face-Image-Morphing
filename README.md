# Face Image Morphing
Final project of Interactive Computer Graphics 2020 Spring, NTU CSIE

## Introduction
This repository is a simple program for face image morphing implemented in python3. The program can be split into two parts: face feature extraction and 2D image morphing. For the feature extraction part, we used [OpenCV](https://github.com/skvark/opencv-python) and [dlib](https://github.com/davisking/dlib) packages following the instructions from the article [Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/). While the 2D image morphing program is an implementation of the paper "Feature-based image metamorphosis"<sup>[1](#Reference)</sup> in [Numpy](https://numpy.org/).

## Instructions
### Morphing two images
A simple way of using this feature is to run `run.sh`:
```
bash run.sh <path to image 1> <path to image 2> <ratio> <directory of output images> [name of the morphed image]
```
Note that the `ratio` is the interpolation ratio between the two images. `ratio = 0` represents the output image will be the first image, while `ratio = 1` represents the output image will be the second image.

### Morphing three images

## Reference
1. [T. Beier and S. Neely, "Feature-based image metamorphosis", SIGGRAPH, 1992](https://www.cs.princeton.edu/courses/archive/fall00/cs426/papers/beier92.pdf)

