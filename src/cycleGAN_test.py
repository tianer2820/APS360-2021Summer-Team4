######################FROM COLAB ####################################################
from multiprocessing import Process, freeze_support
def run():
    torch.multiprocessing.freeze_support()
    print('loop')
if __name__ == '__main__':
    run()

from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

       # print(input_nc)
        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print('shape', x.shape)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print("before:", x.shape)
        x =  self.model(x)
       # print("after:", x.shape)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    import random
    import time
    import datetime
    import sys
    
    from torch.autograd import Variable
    import torch
    #from visdom import Visdom
    import numpy as np
    
    def tensor2image(tensor):
        image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
        if image.shape[0] == 1:
            image = np.tile(image, (3,1,1))
        return image.astype(np.uint8)
    
    class Logger():
        def __init__(self, n_epochs, batches_epoch):
            #self.viz = Visdom()
            self.n_epochs = n_epochs
            self.batches_epoch = batches_epoch
            self.epoch = 1
            self.batch = 1
            self.prev_time = time.time()
            self.mean_period = 0
            self.losses = {}
            self.loss_windows = {}
            self.image_windows = {}
    
    
        def log(self, losses=None, images=None):
            self.mean_period += (time.time() - self.prev_time)
            self.prev_time = time.time()
    
            sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
    
            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    #self.losses[loss_name] = losses[loss_name].data[0]
                    self.losses[loss_name] = losses[loss_name].data
                else:
                   # self.losses[loss_name] += losses[loss_name].data[0]
                    self.losses[loss_name] += losses[loss_name].data
    
                if (i+1) == len(losses.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))
    
            batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
            batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
            sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))
    
            # Draw images
            '''
            for image_name, tensor in images.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
                else:
                    self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})
                    '''
    
            # End of epoch
            if (self.batch % self.batches_epoch) == 0:
                # Plot losses
                for loss_name, loss in self.losses.items():
                    # Reset losses for next epoch
                      self.losses[loss_name] = 0.0
    
                self.epoch += 1
                self.batch = 1
                sys.stdout.write('\n')
            else:
                self.batch += 1
    
            
    
    class ReplayBuffer():
        def __init__(self, max_size=50):
            assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
            self.max_size = max_size
            self.data = []
    
        def push_and_pop(self, data):
            to_return = []
            for element in data.data:
                element = torch.unsqueeze(element, 0)
                if len(self.data) < self.max_size:
                    self.data.append(element)
                    to_return.append(element)
                else:
                    if random.uniform(0,1) > 0.5:
                        i = random.randint(0, self.max_size-1)
                        to_return.append(self.data[i].clone())
                        self.data[i] = element
                    else:
                        to_return.append(element)
            return Variable(torch.cat(to_return))
    
    class LambdaLR():
        def __init__(self, n_epochs, offset, decay_start_epoch):
            assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
            self.n_epochs = n_epochs
            self.offset = offset
            self.decay_start_epoch = decay_start_epoch
    
        def step(self, epoch):
            return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant(m.bias.data, 0.0)

import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = torchvision.datasets.ImageFolder(root=root, transform=transforms_)#sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = torchvision.datasets.ImageFolder(root=root, transform=transforms_)#sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
import glob
import random
import os
    
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision
    
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = torchvision.datasets.ImageFolder(root=root, transform=transforms_)#sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = torchvision.datasets.ImageFolder(root=root, transform=transforms_)#sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
    
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
    
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
    
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
def get_relevant_indices(dataset):
  indices = []
  for i in range(len(dataset)):
      indices.append(i)
  return indices

  #all of my imports:

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import argparse

def get_data_loader ( bs=60, transform = None):

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    raw_data = torchvision.datasets.ImageFolder(root = r'C:\Users\wangy\Desktop\CycleGAN_test\PyTorch-CycleGAN-master\dataset', transform = transform)


  # get relevant indicies to sample from:
    relevant_indices = get_relevant_indices(raw_data)
    np.random.seed(1000)
    np.random.shuffle(relevant_indices)
  
    split = int(len(relevant_indices) * 0.85) # this is the train_validation  & test data split
    train_val_indices, test_indices = relevant_indices [:split], relevant_indices [split:]
  
    split_2 = int(len(relevant_indices) * 0.7) # this is the train & validation split
    train_indices, validation_indices = train_val_indices [:split_2], train_val_indices [split_2:]
  
    # getting the sampler:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)
  
    raw_sampler = SubsetRandomSampler(relevant_indices)
  
    #splitting the data:
    train_loader =  torch.utils.data.DataLoader(raw_data, batch_size=bs, num_workers=1, sampler = train_sampler)#shuffle = True)
    val_loader =  torch.utils.data.DataLoader(raw_data, batch_size=bs, num_workers=1, sampler=val_sampler)
    test_loader =  torch.utils.data.DataLoader(raw_data, batch_size=bs, num_workers=1, sampler=test_sampler)
  
    raw_loader = torch.utils.data.DataLoader(raw_data, batch_size=bs, num_workers=1, sampler=raw_sampler)
  
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = get_data_loader(bs = 1);

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=7, help='size of the batches')
parser.add_argument('--dataroot', type=str, default= r'C:\Users\wangy\Desktop\CycleGAN_test\PyTorch-CycleGAN-master\dataset', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=2, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()

def train(G, D, lr=0.002, batch_size=64, num_epochs=20):

    rand_size = 100;

    # optimizers for generator and discriminator
    d_optimizer = optim.Adam(D.parameters(), lr)
    g_optimizer = optim.Adam(G.parameters(), lr)
 
    # define loss function
    criterion = nn.BCEWithLogitsLoss()

    # get the training datasets
    #train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

    # prepare data loader
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    train_loader = get_data_loader()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # fixed data for testing
    sample_size=16
    test_noise = np.random.uniform(-1, 1, size=(batch_size,3,256,256))
    test_noise = torch.from_numpy(test_noise).float()
    #print(test_noise.shape)
    #x = input("w")

    for epoch in range(num_epochs):
        D.train()
        G.train()
        
        for batch_i, (real_images, _) in enumerate(train_loader[0]):
                    
            batch_size = real_images.size(0)
            
            # rescale images to range -1 to 1
            real_images = real_images*2 - 1
            
            # === Train the Discriminator ===
            
            d_optimizer.zero_grad()

            # discriminator losses on real images 
            D_real = D(real_images)
            labels = torch.ones(batch_size)
            d_real_loss = criterion(D_real.squeeze(), labels)
            
            # discriminator losses on fake images
            z = np.random.uniform(-1, 1, size=(batch_size,3,256,256)) # size of the input to generator
            # the 1 is the number of channels
            #print("z:", z.shape)
            z = torch.from_numpy(z).float()
            
            fake_images = G(z)

            D_fake = D(fake_images)
            labels = torch.zeros(batch_size) # fake labels = 0
            d_fake_loss = criterion(D_fake.squeeze(), labels)
            
            # add up losses and update parameters
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            

            # === Train the Generator ===
            g_optimizer.zero_grad()
            
            # generator losses on fake images
            z = np.random.uniform(-1, 1, size=(batch_size,3,256,256))
            #print("z", z.shape)
            z = torch.from_numpy(z).float()
            fake_images = G(z)
          
            D_fake = D(fake_images)
            labels = torch.ones(batch_size) #flipped labels

            # compute loss and update parameters
            g_loss = criterion(D_fake.squeeze(), labels)
            g_loss.backward()
            g_optimizer.step()

        # print loss
        print('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
              % (epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))
        #print("\n\n\n\n\n\n\no \n\n\n")
        # plot images
        G.eval()
        D.eval()
        test_images = G(test_noise)
        print(test_images[1:])
        #x = input("wait")

        plt.figure(figsize=(9, 3))
        for k in range(16):
            plt.subplot(2, 8, k+1)
            #print(test_images[k,:].shape)
            #reconstructing the shape of images:
            plt.imshow(test_images[k,:].data.numpy().swapaxes(0,2))
        plt.show()
    
    return losses

Gen = Generator(opt.input_nc, opt.output_nc)
Dis = Discriminator(opt.input_nc)
print("the training start")
train(Gen, Dis, lr=0.002, batch_size=32, num_epochs=20)