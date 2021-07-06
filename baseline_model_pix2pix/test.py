import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.figure
import itertools

from dataset import get_dataset

pixel_set = get_dataset('new_data/train/pixel', 256)
photo_set = get_dataset('new_data/train/photo', 256)


fixed_xs = []
for i in range(6):
    fixed_xs.append(photo_set[i].unsqueeze(0))
fixed_x_ = torch.cat(fixed_xs, dim=0)

fixed_ys = []
for i in range(6):
    fixed_ys.append(pixel_set[i].unsqueeze(0))
fixed_y_ = torch.cat(fixed_ys, dim=0)


size_figure_grid = 2
fig, ax = plt.subplots(6, size_figure_grid, figsize=(size_figure_grid, 6), squeeze=False)

for i, j in itertools.product(range(6), range(size_figure_grid)):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)

for i in range(6):
    ax[i, 0].cla()
    ax[i, 0].imshow((fixed_x_[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    ax[i, 1].cla()
    ax[i, 1].imshow((fixed_y_[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2)

fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
label = 'Epoch {0}'.format(123)
fig.text(0.5, 0, label, ha='center')


plt.show()
