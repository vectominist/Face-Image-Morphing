import numpy as np

from src.image_lib import resize_to_same, read_image, show_image, show_arrows
from src.morphing_np import Morphing
from src.feat_extract import FaceFeatureExtractor

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
        raise NotImplementedError
    
    def FaceMorph2D(self, img_1_path, img_2_path, ratio):
        '''
            Combination of all steps of face image morphing.
            img_1 : input image 1
            img_2 : input image 2
            ratio : ratio between the two images
                    0.0 means the output image will be img_1
                    1.0 means the output image will be img_2
        '''
        raise NotImplementedError

        # Feature extraction
        shape_1, img_1 = self.FaceFeatureExtractor.ExtractFeature(img_1_path)
        shape_2, img_2 = self.FaceFeatureExtractor.ExtractFeature(img_2_path)
        
        # Process feature
        # TODO
        P1, Q1 = None, None
        P2, Q2 = None, None

        # Image prepocessing (crop face using features)
        # Note: feature vectors need to be modified after cropping faces
        # TODO
        
        # Image morphing
        img_out = self.Morph.TwoImageMorphing(img_1, img_2, 
                                              P1, Q1, P2, Q2, ratio)

        return img_out
