import argparse
from os.path import join
from face_morph import FaceImageMorphing
from src.image_lib import show_image, show_arrows, save_image, butterworth_shade
import matplotlib.pyplot as plt

SHOW_RESULTS = False

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
parser.add_argument('-n',  '--name', default=None, type=str, help='output image file name')
parser.add_argument('-s',  '--save', default=False, action='store_true', help='output image file name')
paras = vars(parser.parse_args())

# E.g. 
# python3 main.py -i1 ~/Desktop/sample_image/keanu_0.jpg -i2 ~/Desktop/sample_image/peng_0.png

# Process face image morphing
model = FaceImageMorphing(paras['a'], paras['b'], paras['p'], paras['shape_predictor'])
img_1, img_2, img_out, P1, Q1, P2, Q2 = model.FaceMorph2D(paras['image_1'], paras['image_2'], paras['ratio'])

img_out = butterworth_shade(img_out, 1.)

# Show results
if SHOW_RESULTS:
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
    name_out = 'morph_{}.png'.format(str(paras['ratio'])) \
                if paras['name'] is None and len(paras['name']) == 0 \
                else paras['name']
    name_1 = '.'.join(name_out.split('.')[:-1]) + '_c1.png'
    name_2 = '.'.join(name_out.split('.')[:-1]) + '_c2.png'

    if paras['save']:
        save_image(join(paras['output'], name_1), img_1)
        save_image(join(paras['output'], name_2), img_2)
    save_image(join(paras['output'], name_out), img_out)
