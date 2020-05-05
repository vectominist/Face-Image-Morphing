import numpy as np
from skimage import io, filters, color, transform, feature, exposure, external
import matplotlib.pyplot as plt

# Basic Operations
def normalize(image):
    image -= np.min(image)
    image = image / np.max(image)
    return image

def thresholding(image, threshold):
    return (image > threshold).astype(np.float64)

def resize(image, shape_0, shape_1):
    return transform.resize(image, (shape_0, shape_1))

def resize_to_same(img_1, img_2):
    swap = False
    if img_1.shape == img_2.shape:
        return img_1, img_2
    if img_1.shape[0] > img_2.shape[1]:
        img_1, img_2 = img_2, img_1 # img_1 has smaller height
        swap = True
    H = img_1.shape[0]
    img_2 = resize(img_2, H, int(img_2.shape[1] * H / img_2.shape[0]))
    W = min(img_1.shape[1], img_2.shape[1])
    img_1, img_2 = img_1[:, :W], img_2[:, :W]

    if swap: return img_2, img_1
    else:    return img_1, img_2

# Basic IO
def read_image(filename, as_gray=False):
    if filename.split('.')[-1] in ['tif', 'tiff']:
        image = external.tifffile.imread(filename)
    else:
        image = io.imread(filename, as_gray=as_gray) # numpy array
    # color image : M x N x 3
    # gray image  : M x N
    # M : height
    # N : width
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    if image.max() > 1:
        image = image.astype(float) / 255.
    return image

def show_image(image, mode=None):
    if mode is None:
        if image.shape[2] == 1:
            image = np.concatenate((image, image, image), axis=2)
        plt.imshow(image)
    elif mode == 'gray':
        plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    # plt.clf()

def save_image(filename, image, norm=True):
    if norm:
        image = normalize(image.astype(np.float64))
    image *= 255
    io.imsave(filename, image.astype(np.uint8))

def show_arrows(P, Q, color='r'):
    for i in range(len(P)):
        plt.arrow(P[i, 1], 
                  P[i, 0], 
                  Q[i, 1] - P[i, 1], 
                  Q[i, 0] - P[i, 0], 
                  color=color, width=5)
