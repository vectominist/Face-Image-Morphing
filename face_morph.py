import numpy as np
import cv2
from src.image_lib import read_image, show_image, show_arrows, resize
from src.morphing_np import Morphing
from src.feat_extract import FaceFeatureExtractor, get_feature_dict

def FeaturePostprocess(shape):
    '''
        Contruct feature vectors from extracted facial landmarks.
    '''
    P, Q = [], []
    feat_dict = get_feature_dict(shape)
    for i, name in enumerate(feat_dict):
        l = feat_dict[name]
        for j in range(len(l) - 1):
            P.append([l[j, 1], l[j, 0]])
            Q.append([l[j+1, 1], l[j+1, 0]])
        if name in ['Right_Eye', 'Left_Eye', 'Out_Mouth', 'In_Mouth']:
            P.append([l[-1, 1], l[-1, 0]])
            Q.append([l[0, 1], l[0, 0]])
    
    return np.array(P), np.array(Q)

class FaceImageMorphing:
    '''
        The class of the whole face image morphing pipeline,
        including preprocessing, face feature extraction, and
        image morphing.
    '''
    def __init__(self, a=0.1, b=1.0, p=0.5, predictor='shape_predictor_68_face_landmarks.dat'):
        # TODO: feature extraction
        self.Morph = Morphing(a, b, p)
        self.FaceFeatureExtractor = FaceFeatureExtractor(predictor)
    
    def FaceMorph2D(self, img_1_path, img_2_path, ratio):
        '''
            Combination of all steps of face image morphing.
            img_1 : input image 1
            img_2 : input image 2
            ratio : ratio between the two images
                    0.0 means the output image will be img_1
                    1.0 means the output image will be img_2
        '''
        # Feature extraction
        img_1, shape_1 = self.FaceFeatureExtractor.ExtractFeature(img_1_path)
        img_2, shape_2 = self.FaceFeatureExtractor.ExtractFeature(img_2_path)

        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

        # Resize the two images to the same (both images were cropped to square already)
        if img_1.shape[0] > img_2.shape[0]:
            shape_2 *= img_1.shape[0] / img_2.shape[0]
            img_2 = resize(img_2, img_1.shape[0], img_1.shape[1])
        elif img_1.shape[0] < img_2.shape[0]:
            shape_1 *= img_2.shape[0] / img_1.shape[0]
            img_1 = resize(img_1, img_2.shape[0], img_2.shape[1])
        
        if img_1.max() > 1.:
            img_1 = img_1.astype(float) / 255.
        if img_2.max() > 1.:
            img_2 = img_2.astype(float) / 255.

        # Process feature
        P1, Q1 = FeaturePostprocess(shape_1)
        P2, Q2 = FeaturePostprocess(shape_2)

        # Image morphing
        img_out = self.Morph.TwoImageMorphing(img_1, img_2, 
                                              P1, Q1, P2, Q2, ratio)

        return img_1, img_2, img_out, P1, Q1, P2, Q2
