import torch
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
        self.generator = nn.Sequntial(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3 , stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 6, out_channels = 20, kernel_size = 3 , stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 20, out_channels = 45, kernel_size = 3)
        )
        
        self.discriminator = nn.Sequential (
            #upsampling??
            nn.ConvTranspose2d(in_channels = 45, out_channels = 20, kernel_size = 5),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 6, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        

    def forward(self, style, content):
        #FC layers/MLP inserted here 
        style = self.generator(style)
        content = self.generator(content)
        self.target_features = AdaIN(content,style)
        #do we need to upsample the data?
        self.generated_images = self.discriminator(self.target_features)
   
if __name__ == "__main__":
    # test
    mymodel = TestModel()
