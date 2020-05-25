import argparse
from os.path import join
from face_morph import FaceImageMorphing
from src.image_lib import show_image, show_arrows, save_image
import matplotlib.pyplot as plt

# Arguments
parser = argparse.ArgumentParser(description='Face image morphing program.')
parser.add_argument('-i1', '--image_1', type=str, help='Path to the first image.')
parser.add_argument('-i2', '--image_2', type=str, help='Path to the second image.')
parser.add_argument('-sp', '--shape-predictor', default='shape_predictor_68_face_landmarks.dat', type=str, help="Path to facial landmark predictor")
parser.add_argument('-r',  '--ratio', default=0.5, type=float, help='morphing ratio of the two images')
parser.add_argument('-a',  '--a', default=0.1, type=float, help='morphing hyper-parameter a')
parser.add_argument('-b',  '--b', default=1.0, type=float, help='morphing hyper-parameter b')
parser.add_argument('-p',  '--p', default=0.5, type=float, help='morphing hyper-parameter p')
parser.add_argument('-o',  '--output', default=None, type=str, help='output image directory')
paras = vars(parser.parse_args())

# E.g. 
# python3 main.py -i1 ~/Desktop/sample_image/keanu_0.jpg -i2 ~/Desktop/sample_image/peng_0.png

# Process face image morphing
model = FaceImageMorphing(paras['a'], paras['b'], paras['p'], paras['shape_predictor'])
img_1, img_2, img_out, P1, Q1, P2, Q2 = model.FaceMorph2D(paras['image_1'], paras['image_2'], paras['ratio'])

# Show results
fig=plt.figure(figsize=(15, 5))
fig.add_subplot(1, 3, 1)
show_arrows(P1, Q1, 'r')
show_image(img_1)
fig.add_subplot(1, 3, 2)
show_arrows(P2, Q2, 'r')
show_image(img_2)
fig.add_subplot(1, 3, 3)
show_image(img_out)
plt.show()

# Save image
if paras['output'] is not None and len(paras['output']) > 0:
    name_1 = '.'.join(paras['image_1'].split('/')[-1].split('.')[:-1]) + '_crop.png'
    name_2 = '.'.join(paras['image_2'].split('/')[-1].split('.')[:-1]) + '_crop.png'
    save_image(join(paras['output'], name_1), img_1)
    save_image(join(paras['output'], name_2), img_2)
    save_image(join(paras['output'], 'morph_{}.png'.format(str(paras['ratio']))), img_out)

exit()

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
    left   = np.min(shape[:,0])
    right  = np.max(shape[:,0])
    top    = np.min(shape[:,1])
    bottom = np.max(shape[:,1])

    width  =  right - left
    height = bottom - top
    size   = int(max(width, height) * 1.1)
    w_mid  = (left + right) // 2
    h_mid  = (top + bottom) // 2
    image = image[h_mid - size // 2: h_mid + size // 2 + 1, w_mid - size // 2: w_mid + size // 2 + 1]
    
    shape[:, 0] -= w_mid - size // 2
    shape[:, 1] -= h_mid - size // 2
    
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