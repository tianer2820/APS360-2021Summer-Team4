import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=32, nb=6):
        super(Encoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = nb
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        return x


class Generator(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=32, nb=6):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = nb

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1 = nn.ConvTranspose2d(ngf * 4 + 1, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm = nn.InstanceNorm2d(ngf * 2)
        self.deconv2 = nn.ConvTranspose2d(ngf * 2 + 1, ngf, 3, 2, 1, 1)
        self.deconv2_norm = nn.InstanceNorm2d(ngf)
        self.deconv3 = nn.Conv2d(ngf + 1, output_nc, 7, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.resnet_blocks(input)
        x = self._concat_noise(x)

        x = F.relu(self.deconv1_norm(self.deconv1(x)))
        x = self._concat_noise(x)

        x = F.relu(self.deconv2_norm(self.deconv2(x)))
        x = F.pad(x, (3, 3, 3, 3), 'reflect')

        x = self._concat_noise(x)
        
        o = torch.tanh(self.deconv3(x))
        return o
    
    @staticmethod
    def _concat_noise(x):
        shape = list(x.shape)
        shape[1] = 1
        device = x.device
        rand = torch.randn(shape).to(device)
        x = torch.cat([x, rand], 1)
        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ndf=64):
        super(discriminator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = self.conv5(x)
        # no pooling layer???
        return x

# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel + 1, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        shape = list(x.shape)
        shape[1] = 1
        rand = torch.randn(shape).to(x.device)
        x = torch.cat([x, rand], 1)
        x = self.conv2_norm(self.conv2(x))

        return input + x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
