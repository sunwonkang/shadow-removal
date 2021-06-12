#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np

from models import Generator
from utils import mask_generator
from utils import QueueMask

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--mask_nc', type=int, default=1, help='number of channels of mask data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=400, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()

opt.im_suf_A = '.png'

#Data configuration
opt.dataroot = '/root/USRGAN_test/'
opt.size = 100

opt.dataroot_A = opt.dataroot + 'A'
opt.dataroot_M = opt.dataroot + 'B'

#H/W configuration
opt.n_cpu = 4
if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')

print(opt)


###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc + opt.mask_nc, opt.output_nc)

if opt.cuda:
    netG_A2B.to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B), strict=False)

# Set model's test mode
netG_A2B.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
img_transform = transforms.Compose([
    transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

mask_transform = transforms.Compose([
    transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

###################################
to_pil = transforms.ToPILImage()

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/B'):
    os.makedirs('output/B')

##################################### A to B // shadow to shadow-free
gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

mask_queue = QueueMask(gt_list.__len__())

mask_non_shadow = Variable(Tensor(1, 1, opt.size, opt.size).fill_(-1.0), requires_grad=False)

for idx, img_name in enumerate(gt_list):
    print('predicting: %d / %d' % (idx + 1, len(gt_list)))

    # Set model input
    img = Image.open(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)).convert('RGB')
    w, h = img.size
    mask = Image.open(os.path.join(opt.dataroot_M, img_name + opt.im_suf_A)).convert('L')

    img_var = (img_transform(img).unsqueeze(0)).to(device)
    mask_var = (mask_transform(mask).unsqueeze(0)).to(device)

    # Generate output

    input_var = torch.cat([img_var, mask_var], dim=1).to(device)

    temp_B = netG_A2B(input_var)

    fake_B = 0.5*(temp_B.data + 1.0)
    mask_queue.insert(mask_generator(img_var, temp_B))
    fake_B = np.array(transforms.Resize((h, w))(to_pil(fake_B.data.squeeze(0).cpu())))
    Image.fromarray(fake_B).save('output/B/%s' % img_name + opt.im_suf_A)

    mask_last = mask_queue.last_item()

    print('Generated images %04d of %04d' % (idx+1, len(gt_list)))
