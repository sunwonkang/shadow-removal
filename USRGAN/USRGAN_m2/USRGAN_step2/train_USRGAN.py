#!/usr/bin/python3

from __future__ import print_function
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
# from utils import Logger
from utils import weights_init_normal
#from datasets_SRD import ImageDataset
from datasets_S3 import ImageNMaskDataset
import matplotlib.pyplot as plt
# import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=200, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--mask_nc', type=int, default=1, help='number of channels of mask data') # for MASK
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--resume', action='store_true', help='resume')
opt = parser.parse_args()

#Data configuration
opt.dataroot = '/root/USRGAN_train/'
opt.size = 100

#Learning configuration
opt.batchSize = 1 
opt.epoch = 0
opt.n_epochs = 100
opt.lr = 0.0002
opt.decay_epoch = 50
opt.snapshot_epochs = 25

#H/W configuration
opt.n_cpu = 4
if torch.cuda.is_available():
    opt.cuda = True

opt.resume = False

print(opt)


###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc + opt.mask_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc + opt.mask_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss() #lsgan
#criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


####### resume the training process
if opt.resume:
    print
    'resume training:'
    netG_A2B.load_state_dict(torch.load('output/netG_A2B.pth'))
    netG_B2A.load_state_dict(torch.load('output/netG_B2A.pth'))
    netD_A.load_state_dict(torch.load('output/netD_A.pth'))
    netD_B.load_state_dict(torch.load('output/netD_B.pth'))

    optimizer_G.load_state_dict(torch.load('output/optimizer_G.pth'))
    optimizer_D_A.load_state_dict(torch.load('output/optimizer_D_A.pth'))
    optimizer_D_B.load_state_dict(torch.load('output/optimizer_D_B.pth'))

    lr_scheduler_G.load_state_dict(torch.load('output/lr_scheduler_G.pth'))
    lr_scheduler_D_A.load_state_dict(torch.load('output/lr_scheduler_D_A.pth'))
    lr_scheduler_D_B.load_state_dict(torch.load('output/lr_scheduler_D_B.pth'))


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
input_M = Tensor(opt.batchSize, opt.mask_nc, opt.size, opt.size)

target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
print('Preparing data...')
transforms_image = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
                transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transforms_mask = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
                transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])]
dataloader = DataLoader(ImageNMaskDataset(opt.dataroot, transforms_image=transforms_image, transforms_mask=transforms_mask),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

#inpt_M = sorted(glob.glob(os.path.join(root, 'shadow_mask') + '/*.*'))
# Loss plot
#logger = Logger(opt.n_epochs, len(dataloader), server='http://137.189.90.150', http_proxy_host='http://proxy.cse.cuhk.edu.hk/', env = 'main')
###################################

plt.ioff()
curr_iter = 0
G_losses = []
D_A_losses = []
D_B_losses = []
to_pil = transforms.ToPILImage()

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        real_M = Variable(input_M.copy_(batch['M']))
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        real_BM = torch.cat([real_B, real_M], dim=1).cuda()
        same_B = netG_A2B(real_BM)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0  #||Gb(b)-b||1
        # G_B2A(A) should equal A if real A is fed
        real_AM = torch.cat([real_A, real_M], dim=1).cuda()
        same_A = netG_B2A(real_AM)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0  #||Ga(a)-a||1
        # GAN loss

        fake_B = netG_A2B(real_AM)
        fake_BM = torch.cat([fake_B, real_M], dim=1).cuda()
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real) #log(Db(Gb(a)))

        fake_A = netG_B2A(real_BM)
        fake_AM = torch.cat([fake_A, real_M], dim=1).cuda()
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real) #log(Da(Ga(b)))

        # Cycle loss
        recovered_A = netG_B2A(fake_BM)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0 #||Ga(Gb(a))-a||1

        recovered_B = netG_A2B(fake_AM)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0 #||Gb(Ga(b))-b||1

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        G_losses.append(loss_G.item())

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real) #log(Da(a))

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)  #log(1-Da(G(b)))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        D_A_losses.append(loss_D_A.item())

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real) #log(Db(b))

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake) #log(1-Db(G(a)))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        D_B_losses.append(loss_D_B.item())

        optimizer_D_B.step()
        ###################################

        curr_iter += 1

        if i % 1 == 0:
            log = '[iter %d], [loss_G %.5f], [loss_G_identity %.5f], [loss_G_GAN %.5f],' \
                  '[loss_G_cycle %.5f], [loss_D %.5f]' % \
                  (curr_iter, loss_G, (loss_identity_A + loss_identity_B), (loss_GAN_A2B + loss_GAN_B2A),
                   (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B))
            print(log)

            img_fake_A = 0.5 * (fake_A.detach().data + 1.0)
            img_fake_A = (to_pil(img_fake_A.data[0].squeeze(0).cpu()))
            img_fake_A.save('output/fake_A.png')

            img_fake_B = 0.5 * (fake_B.detach().data + 1.0)
            img_fake_B = (to_pil(img_fake_B.data[0].squeeze(0).cpu()))
            img_fake_B.save('output/fake_B.png')

        # Progress report (http://137.189.90.150:8097)
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
        #             images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')

    torch.save(optimizer_G.state_dict(), 'output/optimizer_G.pth')
    torch.save(optimizer_D_A.state_dict(), 'output/optimizer_D_A.pth')
    torch.save(optimizer_D_B.state_dict(), 'output/optimizer_D_B.pth')

    torch.save(lr_scheduler_G.state_dict(), 'output/lr_scheduler_G.pth')
    torch.save(lr_scheduler_D_A.state_dict(), 'output/lr_scheduler_D_A.pth')
    torch.save(lr_scheduler_D_B.state_dict(), 'output/lr_scheduler_D_B.pth')

    if (epoch+1) % opt.snapshot_epochs == 0:
        torch.save(netG_A2B.state_dict(), ('output/netG_A2B_%d.pth' % (epoch+1)))
        torch.save(netG_B2A.state_dict(), ('output/netG_B2A_%d.pth' % (epoch+1)))
        # torch.save(netD_A.state_dict(), ('output/netD_A_%d.pth' % (epoch+1)))
        # torch.save(netD_B.state_dict(), ('output/netD_B_%d.pth' % (epoch+1)))

    print('Epoch:{}'.format(epoch))


    if (epoch+1) % opt.snapshot_epochs == 0:
        plt.figure(figsize=(10, 5))
        plt.title("Generator Loss During Training")
        plt.plot(G_losses, label="G loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig('./output/generator.png')
        # plt.show(block=False)

        plt.figure(figsize=(10, 5))
        plt.title("Discriminator Loss During Training")
        plt.plot(D_A_losses, label="D_A loss")
        plt.plot(D_B_losses, label="D_B loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig('./output/discriminator.png')
        # plt.show(block=False)
        plt.close('all')

###################################
