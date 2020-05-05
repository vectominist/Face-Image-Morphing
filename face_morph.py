import numpy as np

from src.image_lib import resize_to_same, read_image, show_image, show_arrows
from src.morphing_np import Morphing

class FaceImageMorphing:
    '''
        The class of the whole face image morphing pipeline,
        including preprocessing, face feature extraction, and
        image morphing.
    '''
    def __init__(self, a=0.1, b=1.0, p=0.5):
        # TODO: feature extraction
        self.Morph = Morphing(a, b, p)
        raise NotImplementedError
    
    def FaceMorph2D(self, img_1, img_2, ratio):
        '''
            Combination of all steps of face image morphing.
            img_1 : input image 1
            img_2 : input image 2
            ratio : ratio between the two images
                    0.0 means the output image will be img_1
                    1.0 means the output image will be img_2
        '''
        raise NotImplementedError
        # Image prepocessing
        # TODO

        # Feature extraction
        # TODO
        P1, Q1 = None, None
        P2, Q2 = None, None
        
        # Image morphing
        img_out = self.Morph.TwoImageMorphing(img_1, img_2, 
                                              P1, Q1, P2, Q2, ratio)

        return img_out
