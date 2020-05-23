from collections import OrderedDict
import numpy as np
from src.image_lib import read_image, show_image
import cv2
import argparse
import dlib
import imutils



# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


def shape_to_np(shape, dtype="int"):
    """
    shape: feature coordinates from predictor
    return np.array of coordinates [[x, y]...]
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_feature_dict(shape):
    """
    shape: feature coordinates from shape_to_np()
    return dict of face feature names and coordinates
    facial_features_cordinates = { 'name': array[(coordinates)...] }
    """
    facial_features_cordinates = {}
    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

    print(facial_features_cordinates)
    return facial_features_cordinates


def show_feature_crcles(image, shape):
    """
    image: image read with cv2.imread()
    shape: feature coordinates from shape_to_np()
    return image with feature circles
    """
    # show feature circles
    for i, (x, y) in enumerate(shape):
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(image, '{}'.format(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

    return image


def crop_to_face(image, shape):
    """
    image: image read with cv2.imread()
    shape: feature coordinates from shape_to_np()
    return cropped image & modified coordinates
    """
    # crop image to only the face
    (left, top) = shape[0]
    (right, bottom) = shape[0]
    
    # find border of the face
    for (x, y) in shape:
        if x < left: left = x
        if x > right: right = x
        if y < top: top = y
        if y > bottom: bottom = y

    image = image[top-20:bottom+20, left-20:right+20]
    
    for i in range(68):
        shape[i][0] = shape[i][0] - (left-20)
        shape[i][1] = shape[i][1] - (top-20)

    return image, shape


print('Initializing dlib')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print('Reading image')
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print('Detecting face features')
rects = detector(gray, 1)
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    # get coordinates of features
    shape = shape_to_np(shape)
    
    # put feature in dict
    get_feature_dict(shape)

    # crop image to only the face
    image, shape = crop_to_face(image, shape)

    # show feature circles
    image = show_feature_crcles(image, shape)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
show_image(image)