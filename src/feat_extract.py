import os
from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils

from src.image_lib import show_image

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Out_Mouth", (48, 60)),
    ("In_Mouth", (60, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Up_Nose", (27, 31)),
    ("Down_Nose", (31, 36)),
    ("Jaw", (0, 17))
])

def shape_to_np(shape, dtype="float"):
    '''
    shape: feature coordinates from predictor
    return np.array of coordinates [[x, y]...]
    '''
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_feature_dict(shape):
    '''
    shape: feature coordinates from shape_to_np()
    return dict of face feature names and coordinates
    facial_features_cordinates = { 'name': array[(coordinates)...] }
    '''
    facial_features_cordinates = {}
    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

    # print(facial_features_cordinates)
    return facial_features_cordinates

def show_feature_circles(image, shape):
    '''
    image: image read with cv2.imread()
    shape: feature coordinates from shape_to_np()
    return image with feature circles
    '''
    # show feature circles
    for i, (x, y) in enumerate(shape):
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(image, '{}'.format(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

    return image

def crop_to_face(image, shape):
    '''
    image: image read with cv2.imread()
    shape: feature coordinates from shape_to_np()
    return cropped image & modified coordinates
    '''
    # crop image to only the face
    (left, top) = shape[0]
    (right, bottom) = shape[0]
    
    # find border of the face
    left   = int(np.min(shape[:,0]))
    right  = int(np.max(shape[:,0]))
    top    = int(np.min(shape[:,1]))
    bottom = int(np.max(shape[:,1]))

    width  =  right - left
    height = bottom - top
    size   = int(max(width, height) * 1.)
    w_mid  = (left + right) // 2
    h_mid  = (top + bottom) // 2
    image = image[h_mid - size // 2: h_mid + size // 2 + 1, w_mid - size // 2: w_mid + size // 2 + 1]
    
    shape[:, 0] -= w_mid - size // 2
    shape[:, 1] -= h_mid - size // 2
    
    return image, shape

class FaceFeatureExtractor:
    '''
        Extract face features with dlib
        Turn features into dictionary of coordinates
        Crop image only with the part with the face
    '''
    def __init__(self, shape_predictor):
        self.shape_predictor = shape_predictor
        if not os.path.isfile(shape_predictor):
            import wget
            import bz2
            # If cannot find shape_predictor, download it from dlib.net
            url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            download_name = wget.download(url)
            zipfile = bz2.BZ2File(download_name) # open the file
            data = zipfile.read() # get the decompressed data
            newfilepath = download_name[:-4] # assuming the filepath ends with .bz2
            open(newfilepath, 'wb').write(data) # write a uncompressed file
            self.shape_predictor = newfilepath
    
    def ExtractFeature(self, image_path):
        print('Initializing dlib')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.shape_predictor)

        print('Reading image')
        image = cv2.imread(image_path)
        assert image is not None
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print('Detecting face features')
        rects = detector(gray, 1)
        for i, rect in enumerate(rects):
            shape = predictor(gray, rect)
            # get coordinates of features
            shape = shape_to_np(shape)

            # crop image to only the face
            image, shape = crop_to_face(image, shape)

            # show feature circles
            # image_with_feat = show_feature_circles(image, shape.astype(int))

            # assume only one face is in the image
            break
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, shape


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    model = FaceFeatureExtractor(args['shape_predictor'])
    image, shape = model.ExtractFeature(args['image'])

    # show feature circles
    image = show_feature_circles(image, shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_image(image)
