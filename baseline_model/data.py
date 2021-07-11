import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, utils
from torch import optim
from PIL import Image
from imageio import imwrite


def load_image(image_path, max_size=400, shape=None) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    size = shape if shape is not None else max_size if max(image.size) > max_size else max(image.size) 
    transform = transforms.Compose([transforms.Resize(size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    image = transform(image)[:3,:,:].unsqueeze(0)
    return image    


def image_convert(image) -> np.ndarray:
    image = image.to("cpu").clone().detach().squeeze()
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image