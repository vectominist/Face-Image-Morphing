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
            img_1_path : path of input image 1
            img_2_path : path of input image 2
            ratio : ratio between the two images
                    0.0 means the output image will be img_1
                    1.0 means the output image will be img_2
        '''
        # Feature extraction
        img_1, shape_1 = self.FaceFeatureExtractor.ExtractFeature(img_1_path)
        img_2, shape_2 = self.FaceFeatureExtractor.ExtractFeature(img_2_path)

        # Resize the two images to the same (both images were cropped to square already)
        size_max = 200
        shape_1 *= size_max / img_1.shape[0]
        img_1 = resize(img_1, size_max, size_max)
        shape_2 *= size_max / img_2.shape[0]
        img_2 = resize(img_2, size_max, size_max)

        # Process feature
        P1, Q1 = FeaturePostprocess(shape_1)
        P2, Q2 = FeaturePostprocess(shape_2)

        # Image morphing
        img_out, Pd, Qd = self.Morph.TwoImageMorphing(img_1, img_2, 
                                              P1, Q1, P2, Q2, ratio)

        return img_1, img_2, img_out, P1, Q1, P2, Q2

    def FaceMorph2D3Img(self, img_1_path, img_2_path, img_3_path, ratio_1, ratio_2):
        '''
            Combination of all steps of face image morphing.
            img_1 : path of input image 1
            img_2 : path of input image 2
            img_3 : path of input image 3
            ratio : ratio between the two images
                    0.0 means the output image will be img_1
                    1.0 means the output image will be img_2
        '''
        # Feature extraction
        img_1, shape_1 = self.FaceFeatureExtractor.ExtractFeature(img_1_path)
        img_2, shape_2 = self.FaceFeatureExtractor.ExtractFeature(img_2_path)
        img_3, shape_3 = self.FaceFeatureExtractor.ExtractFeature(img_3_path)

        # Resize the three images to the same (both images were cropped to square already)
        size_max = 200
        shape_1 *= size_max / img_1.shape[0]
        img_1 = resize(img_1, size_max, size_max)
        shape_2 *= size_max / img_2.shape[0]
        img_2 = resize(img_2, size_max, size_max)
        shape_3 *= size_max / img_3.shape[0]
        img_3 = resize(img_3, size_max, size_max)
        
        # Process feature
        P1, Q1 = FeaturePostprocess(shape_1)
        P2, Q2 = FeaturePostprocess(shape_2)
        P3, Q3 = FeaturePostprocess(shape_3)

        # Calculate ratio:
        ratio_1, ratio_2 = ratio_2 / (ratio_1 + ratio_2), (1. - ratio_1 - ratio_2) / (ratio_1 + ratio_2)

        # Image morphing
        img_out, Pd, Qd = self.Morph.TwoImageMorphing(img_1, img_2, 
                                              P1, Q1, P2, Q2, ratio_1)
        img_out, Pd, Qd = self.Morph.TwoImageMorphing(img_out, img_3, 
                                              Pd, Qd, P3, Q3, ratio_2)

        return img_1, img_2, img_3, img_out, P1, Q1, P2, Q2, P3, Q3
