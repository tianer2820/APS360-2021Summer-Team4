'''import torch
from torch import nn
import torch.nn.functional as F
import tensorflow as tf
from adain import AdaIN

#requirements:
#using CNN, AdaIN structure, attention mechanism: https://arxiv.org/pdf/1812.04948.pdf
#https://github.com/xunhuang1995/AdaIN-style
#main reference: https://github.com/elleryqueenhomels/arbitrary_style_transfer
#idk if this is the right Adain layer that we should use
#idea: encod the image into a latent space
# pass it through the AdaIN network
# pass it through the decoder
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.generator_1 = nn.Sequntial( #generator itself can be a autoencoder
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3 , stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 6, out_channels = 20, kernel_size = 3 , stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 20, out_channels = 45, kernel_size = 3),
            nn.ConvTranspose2d(in_channels = 45, out_channels = 20, kernel_size = 5), #the decoder part
            nn.ReLU(),
            nn.ConvTranspose2d(20, 6, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # the discriminator is a classifier. distinguish real data from data created by the generator
        self.discriminator = nn.Sequential (

        )

    def forward(self, real):#Adain for later
        #FC layers/MLP inserted here 
        #style = self.generator_encoder(style)
        #content = self.generator_encoder(content)
        #self.target_features = AdaIN(content,style)
        #do we need to upsample the data?
       # self.generated_images = self.discriminator(self.target_features)
        #return self.generated_images
        fake = self.generator_encoder(real.copy())
        fake = self.generator_decoder(fake)

   
if __name__ == "__main__":
    # test
    mymodel = TestModel()
'''

#residual block from the reference:
# they added this part to the generator
#modify this to print out the size of the images.....
import torch
from torch import nn
import torch.nn.functional as F
import tensorflow as tf
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

#class generator(nn.Module):

class gen(nn.Module):
    def __init__(self, in_ch):
        super(gen, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 16*in_ch, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16*in_ch, 64*in_ch, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64*in_ch, 128*in_ch, 7)
        self.tconv3 = nn.ConvTranspose2d(128*in_ch, 64*in_ch, 7) #the decoder part
        self.tconv2 = nn.ConvTranspose2d(64*in_ch, 16*in_ch, 3, stride=2, padding=1, output_padding=1)
        self.tconv1 =  nn.ConvTranspose2d(16*in_ch, in_ch, 3, stride=2, padding=1, output_padding=1)
        #self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2);
        self.Residual = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, in_ch, 3),
            nn.InstanceNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, in_ch, 3),
            nn.InstanceNorm2d(in_ch) 
        )
        
        # the discriminator is a classifier. distinguish real data from data created by the generator
        #discriminator uses a leaky relu which has a small slope for negative values and is usually used for avoid saturation/dominance of noise -> do we need??

    def forward(self, x):#Adain for later
        print("before:", x.shape)
        x = F.relu (self.conv1(x))
        print("1:", x.shape)
        x = F.relu (self.conv2(x))
        print("2:", x.shape)
        x = F.relu (self.conv3(x))
        print("3:", x.shape)
        x = F.relu (self.tconv3(x))
        print("4:", x.shape)
        x = F.relu (self.tconv2(x))
        print("5:", x.shape)
        x = F.relu (self.tconv1(x))
        print("6:", x.shape)
        #residual block was added here:
        #x = self.Residual(x)
        #print("res & after:", x.shape)
       
        return x
class dis(nn.Module):
    def __init__(self, in_ch):
        super(dis, self).__init__()
        self.conv1= nn.Conv2d(in_channels = in_ch, out_channels = 64, kernel_size = 3 , stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 120, kernel_size = 3 , stride = 2, padding = 1)
        self.conv3 =  nn.Conv2d(in_channels = 120, out_channels = 348, kernel_size = 3 , stride = 2, padding = 1)
        self.conv4 =  nn.Conv2d(in_channels = 348, out_channels = 546, kernel_size = 3 , stride = 2, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 546, out_channels = 1, kernel_size = 3 , stride = 2, padding = 1)# the classification layer
        self.dropout = nn.Dropout(0.3)
        
          
    def forward(self, x):
      #print("before:", x.shape)
      x = F.leaky_relu(self.conv1(x), 0.2)
      x = self.dropout(x)
      #print("1:", x.shape)
      x = F.leaky_relu(self.conv2(x), 0.2)
      x = self.dropout(x)
      #print("2:", x.shape)
      x = F.leaky_relu(self.conv3(x), 0.2)
      x = self.dropout(x)
     # print("3:", x.shape)
      x = F.leaky_relu(self.conv4(x), 0.2)
      x = self.dropout(x)
     # print("4:", x.shape)
      x = F.leaky_relu(self.conv5(x), 0.2)
      x =  F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
     # print("after:", x.shape)
      return x
#testmodel = Model(3)
