import numpy as np

def Perpendicular(V):
    # V : N x 2
    return np.stack([V[:,1], -V[:,0]],axis=-1)

class Morphing:
    def __init__(self, a=0.1, b=1.0, p=0.5):
        self.a = a
        self.b = b
        self.p = p

    def MultiFeatMorphing(self, X, P, Q, Pp, Qp):
        # X : H x W x 2
        # P : N x 2
        # Q : N x 2
        X = np.repeat(np.expand_dims(X, 2), P.shape[0], axis=2) # H x W x N x 2
        print('X : {}, P : {}, Q : {}, Pp : {}, Qp : {}'.format(X.shape, P.shape, Q.shape, Pp.shape, Qp.shape))
        print(np.sum((X-P)*(Q-P), axis=-1).shape)
        u      = np.sum((X - P)*(Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2)**2 # HxWxN
        print('u : {}'.format(u.shape))
        v      = np.sum((X - P)*Perpendicular(Q - P), axis=-1) / np.linalg.norm(Q - P, ord=2) # HxWxN
        print('v : {}'.format(v.shape))
        Xp     = Pp + np.repeat(np.expand_dims(u, axis=-1), 2, axis=-1) * (Qp - Pp) \
                 + np.repeat(np.expand_dims(v, axis=-1), 2, axis=-1) * Perpendicular(Qp - Pp) / np.linalg.norm(Qp - Pp, ord=2)
        print('Xp : {}'.format(Xp.shape))
        D      = Xp - X
        print('D : {}'.format(D.shape))
        dist   = np.absolute(v) * (np.greater_equal(u, 0)*np.less_equal(u, 1)) \
                 + np.linalg.norm(X - P, ord=2, axis=-1) * (np.less(u, 0)) \
                 + np.linalg.norm(X - Q, ord=2, axis=-1) * (np.greater(u, 1))
        # print(dist)
        print('dist : {}'.format(dist.shape))
        weight = np.power(np.power(np.linalg.norm(Q - P), self.p) / (self.a + dist), self.b)
        print('weight : {}'.format(weight.shape))
        Xp     = X[:,:,0,:] + np.sum(D * np.repeat(np.expand_dims(weight, axis=-1), 2, axis=-1), axis=2) \
                 / np.repeat(np.expand_dims(np.sum(weight, axis=-1), axis=-1), 2, axis=-1)
        print('Xp : {}'.format(Xp.shape))
        return Xp

    def image_morphing(self, 
            img_1, img_2, 
            P1, Q1, P2, Q2, 
            ratio=0.5, a=0.1, b=1.0, p=0.5):
        # Interpolate features

        # Calculate morphing
        pass

if __name__ == '__main__':
    M = Morphing()

    H, W = 10, 20
    X = np.zeros((H, W, 2), dtype=float)
    X[:,:,0] = np.expand_dims(np.arange(0, H, dtype=float), axis=1)
    X[:,:,1] = np.expand_dims(np.arange(0, W, dtype=float), axis=0)
    # print(X)
    P = np.array([[1., 5.], [7., 8.], [9., 3.]])
    Q = np.array([[3., 10.], [8., 10.], [10., 7.]])
    Pp = np.array([[2., 6.], [9., 7.], [10., 5.]])
    Qp = np.array([[5., 17.], [12., 11.], [10., 6.]])
    Xp = M.MultiFeatMorphing(X, P, Q, Pp, Qp)
    # print(Xp)