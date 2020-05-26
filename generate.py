import numpy as np
import subprocess
import sys
from os.path import join
import os

from src.image_lib import read_image, resize, save_image

DATA_NUM = [15, 15] # number of image data (0: male, 1: female)
IMG_SIZE = 200

def read_list(filename):
    with open(filename, 'r') as fp:
        male_list, female_list = [], []
        for f in fp:
            if f[-1] == '\n': f = f[:-1]
            if f[0] == 'm':
                male_list.append(f)
            elif f[0] == 'f':
                female_list.append(f)
        male_list   = sorted(male_list)
        female_list = sorted(female_list)
        return male_list, female_list

def gen_rand(total, n):
    choose = np.arange(total)
    np.random.shuffle(choose)
    return choose[:n]

if __name__ == '__main__':
    target   = sys.argv[1] # path to target image
    target_g = sys.argv[2] # gender of target image
    img_dir  = sys.argv[3] # directory of image dataset
    img_list = sys.argv[4] # list image dataset
    out_dir  = sys.argv[5] # output directory

    m_list, f_list = read_list(img_list)
    DATA_NUM[0] = len(m_list)
    DATA_NUM[1] = len(f_list)
    all_list = [m_list, f_list]

    target_g_idx = 0 if target_g.lower() == 'm' else 1
    other_g_idx  = target_g_idx ^ 1

    same_gender_list  = gen_rand(DATA_NUM[target_g_idx], 6)
    other_gender_list = gen_rand(DATA_NUM[other_g_idx], 6)
    print(same_gender_list)
    print(other_gender_list)

    out_dir = join(out_dir, target.split('/')[-1].split('.')[0])
    os.makedirs(out_dir, exist_ok=True)

    tar_img = []
    com_img = []

    for i in range(0, 6, 2):
        name = 'tar_{}.png'.format(i // 2)
        subprocess.run(['bash', 'run3.sh',
                        target,
                        join(img_dir, all_list[other_g_idx][other_gender_list[i]]),
                        join(img_dir, all_list[other_g_idx][other_gender_list[i+1]]),
                        out_dir,
                        name])
        new_img = read_image(join(out_dir, name))
        tar_img.append(resize(new_img, IMG_SIZE, IMG_SIZE))
    
    com_img = [[], [], []]
    for i in range(0, 6, 2):
        name = 'com_{}.png'.format(i)
        subprocess.run(['bash', 'run3.sh',
                        join(img_dir, all_list[target_g_idx][same_gender_list[i]]),
                        join(img_dir, all_list[other_g_idx][other_gender_list[i]]),
                        join(img_dir, all_list[other_g_idx][other_gender_list[i+1]]),
                        out_dir,
                        name])
        new_img = read_image(join(out_dir, name))
        com_img[i // 2].append(resize(new_img, IMG_SIZE, IMG_SIZE))
        name = 'com_{}.png'.format(i + 1)
        subprocess.run(['bash', 'run3.sh',
                        join(img_dir, all_list[target_g_idx][same_gender_list[i+1]]),
                        join(img_dir, all_list[other_g_idx][other_gender_list[i]]),
                        join(img_dir, all_list[other_g_idx][other_gender_list[i+1]]),
                        out_dir,
                        name])
        new_img = read_image(join(out_dir, name))
        com_img[i // 2].append(resize(new_img, IMG_SIZE, IMG_SIZE))
    
    test_1 = np.concatenate([tar_img[0], com_img[0][0], com_img[0][1]], axis=1)
    test_2 = np.concatenate([com_img[1][0], tar_img[1], com_img[1][1]], axis=1)
    test_3 = np.concatenate([com_img[2][0], com_img[2][1], tar_img[2]], axis=1)
    
    save_image(join(out_dir, 'test_1.png'), test_1)
    save_image(join(out_dir, 'test_2.png'), test_2)
    save_image(join(out_dir, 'test_3.png'), test_3)