import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, utils
from torch import optim
from PIL import Image
from imageio import imwrite



def gram_matrix(tensor):
    # batch_size is 1
    batch_size, d, h, w = tensor.shape
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

    
def get_feature_output(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'
                 }
    feature_output = {}
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            feature_output[layers[name]] = image
    return feature_output

