
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.utils.data

from dataset_utils import get_dataset, InfinateLoader


dataset_A = get_dataset('new_data/train/photo/', 256, use_normalize=True)
dataset_B = get_dataset('new_data/train/pixel/', 256, use_normalize=True)

for i in range(50):
    img = dataset_B[i]
    print(torch.max(img), torch.min(img), torch.sum(img) / (img.shape[1] * img.shape[2]))
    img = transforms.F.to_pil_image(img)
    img.save('./print_{:0>4}.png'.format(i))
