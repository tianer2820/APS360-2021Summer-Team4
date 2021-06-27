import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms, utils

from PIL import Image
from imageio import imwrite

from typing import List

from data import image_convert, load_image
from utils import get_feature_output, gram_matrix


def style_transfer(content: torch.Tensor, style: torch.Tensor, vgg: nn.Module, epochs=4000, checkpoint_freq=500) -> List[torch.Tensor]:
    # select GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    content = content.to(device)
    style = style.to(device)

    # Displaying the images
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    # ax1.imshow(image_convert(content))
    # ax2.imshow(image_convert(style))

    # Turning off gradient updates for the parameters of the model
    for parameters in vgg.parameters():
        parameters.requires_grad_(False)
    vgg.to(device)

    # get feature maps of the layers of interest
    content_features = get_feature_output(content, vgg)
    style_features = get_feature_output(style, vgg)

    # Gram matrix for the style features
    style_grams = {layer: gram_matrix(
        style_features[layer]) for layer in style_features}

    # Defining the target mat
    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.8,
                     'conv3_1': 0.5,
                     'conv4_1': 0.3,
                     'conv5_1': 0.1}
    content_weight = 1
    style_weight = 1e5
    
    optimizer = optim.Adam([target], lr=0.001)
    checkpoints = []

    for i in range(epochs):
        current_features = get_feature_output(target, vgg)  # update features

        optimizer.zero_grad()

        # Content loss
        # content_loss = torch.mean(
        #     (target_features["conv4_2"]-content_features["conv4_2"])**2)
        content_loss = F.mse_loss(
            current_features["conv4_2"], content_features["conv4_2"])

        # style loss
        style_loss = 0
        for layer in style_weights:
            current_gram_layer = gram_matrix(current_features[layer])
            layer_loss = style_weights[layer] * \
                F.mse_loss(current_gram_layer, style_grams[layer])
            batch, depth, h, w = target.shape
            style_loss += layer_loss / (depth * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # log progress
        if (i+1) % checkpoint_freq == 0:
            print('Total loss: ', total_loss.item())
            check = target.cpu().detach()
            checkpoints.append(check)

    return checkpoints


if __name__ == "__main__":
    content_path = './images/janelle.png'
    style_path = './images/Starry-Night-by-Vincent-Van-Gogh-painting.jpg'

    content = load_image(content_path)
    style = load_image(style_path, shape=content.shape[-2:])

    vgg = models.vgg19(pretrained=True).features
    epochs = 400

    imgs = style_transfer(content, style, vgg, epochs, checkpoint_freq=100)
    for i, img in enumerate(imgs):
        imwrite('./output/content_image{:0>4}.jpg'.format(i), image_convert(img))

