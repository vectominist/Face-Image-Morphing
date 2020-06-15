import numpy as np

def Perpendicular(v):
    # v : N x 2
    return np.stack([v[:,1], -v[:,0]],axis=-1)

def PointInterpolation(v1, v2, ratio=0.5):
    # v1, v2 : N x 2
    # interpolation of points by finding the midpoint
    return v1 * (1. - ratio) + v2 * ratio

class Morphing:
    '''
        An implementation of the morphing algorithm from the 
        paper "Feature-Based Image Metamorphosis".
    '''
    def __init__(self, a=0.1, b=2.0, p=0.5):
        self.a = a
        self.b = b # recommended range: [0.5, 2]
        self.p = p # recommended range: [0  , 1]

    def MultiFeatMorphing(self, X, P, Q, Pp, Qp):
        '''
            Calculate morphing with multiple feature vectors on 
            destination coordinate X. Using numpy as instead of 
            for loops saves time.
            X  : Destination coordinate         H x W
            P  : Feature vector start (desti)   N x 2
            Q  : Feature vector end (desti)     N x 2
            Pp : Feature vector start (source)  N x 2
            Qp : Feature vector end (source)    N x 2
        '''
        X      = np.repeat(np.expand_dims(X, 2), P.shape[0], axis=2) # HxWxNx2
        u      = np.sum((X - P) * (Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2, axis=-1)**2 # HxWxN
        v      = np.sum((X - P) * Perpendicular(Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2, axis=-1) # HxWxN
        Xp     = Pp + np.repeat(np.expand_dims(u, axis=-1), 2, axis=-1) * (Qp - Pp) \
                 + np.repeat(np.expand_dims(v, axis=-1), 2, axis=-1) * Perpendicular(Qp - Pp) \
                 / np.repeat(np.linalg.norm(Qp - Pp, ord=2, axis=-1, keepdims=True), 2, axis=-1)
        D      = Xp - X # HxWxNx2
        dist   = np.absolute(v) * (np.greater_equal(u, 0, dtype=float) * np.less_equal(u, 1, dtype=float)) \
                 + np.linalg.norm(X - P, ord=2, axis=-1) * (np.less(u, 0, dtype=float)) \
                 + np.linalg.norm(X - Q, ord=2, axis=-1) * (np.greater(u, 1, dtype=float))
        weight = np.power(np.power(np.linalg.norm(Q - P, axis=-1), self.p) / (self.a + dist), self.b)
        Xp     = X[:,:,0,:] + np.sum(D * np.repeat(np.expand_dims(weight, axis=3), 2, axis=-1), axis=2) \
                 / np.repeat(np.expand_dims(np.sum(weight, axis=-1), axis=2), 2, axis=-1)
        Xp = np.around(Xp).astype(int)
        Xp[:,:,0] = np.clip(Xp[:,:,0], 0, X.shape[0] - 1)
        Xp[:,:,1] = np.clip(Xp[:,:,1], 0, X.shape[1] - 1)
        return Xp # HxWx2

    def TwoImageMorphing(self, 
            img_1, img_2, 
            P1, Q1, P2, Q2, 
            ratio=0.5):
        assert img_1.shape == img_2.shape
        H, W = img_1.shape[0], img_1.shape[1]
        # Interpolate features (destination)
        Pd, Qd = PointInterpolation(P1, P2, ratio), PointInterpolation(Q1, Q2, ratio)
        # Calculate morphing
        X = np.zeros((H, W, 2), dtype=float)
        X[:,:,0] = np.expand_dims(np.arange(0, H, dtype=float), axis=1)
        X[:,:,1] = np.expand_dims(np.arange(0, W, dtype=float), axis=0)
        Xp_1 = self.MultiFeatMorphing(X.copy(), Pd, Qd, P1, Q1)
        Xp_2 = self.MultiFeatMorphing(X.copy(), Pd, Qd, P2, Q2)
        result = img_1[Xp_1[:,:,0],Xp_1[:,:,1],:] * (1. - ratio) + \
                 img_2[Xp_2[:,:,0],Xp_2[:,:,1],:] * ratio
        
        return result, Pd, Qd
