import argparse
import sys
import os

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--targetroot', type=str, default='datasets/horse2zebra/', help='')
parser.add_argument('--gtroot', type=str, default='datasets/horse2zebra/', help='')
parser.add_argument('--img_format', type=str, default='.png', help='')
parser.add_argument('--size', type=int, default=100, help='size of the data (squared assumed)')
opt = parser.parse_args()

img_list = [os.path.splitext(f)[0] for f in os.listdir(opt.targetroot) if f.endswith(opt.img_format)]

mse_sum = 0.0

for idx, img_name in enumerate(img_list):
    print('processing... {} / {}'.format(idx+1, len(img_list)))

    target = np.array(Image.open(os.path.join(opt.targetroot, img_name + opt.img_format)).convert('RGB').resize((opt.size, opt.size)), dtype=np.float)
    gt = np.array(Image.open(os.path.join(opt.gtroot, img_name + opt.img_format)).convert('RGB').resize((opt.size, opt.size)), dtype=np.float)

    mse = np.linalg.norm(gt-target)

    mse_sum += mse

avg_mse = mse_sum / len(img_list)

print('avg_mse: {}'.format(avg_mse))
