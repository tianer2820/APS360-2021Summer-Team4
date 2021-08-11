from dataset_utils import get_dataset
import torch.utils.data
import torch
import torchvision.transforms.functional as TF
import network_pix2pix
import os

join = os.path.join
from PIL import Image


PHOTO_FOLDER = 'photo'
PIXEL_FOLDER = 'pixel'
OUT_FOLDER = 'pix2pix_out'


def ensure_path(path):
        if not os.path.isdir(path):
            os.makedirs(path)


def show_result(G, x_, epoch):
    G.eval()
    G.cpu()
    x_ = x_.cpu()
    test_images = G(x_)

    
    for i in range(x_.shape[0]):
        # img_np = (test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2
        # img = Image.fromarray(img_np)
        img = TF.to_pil_image(test_images[i])
        img.save(join(OUT_FOLDER, '{:0>4}_{:0>2}_fake.png'.format(epoch, i)))

        img2 = TF.to_pil_image(x_[i])
        img2.save(join(OUT_FOLDER, '{:0>4}_{:0>2}_input.png'.format(epoch, i)))


photo_set = get_dataset(PHOTO_FOLDER, 256, use_normalize=False, blur_size=3)

filter_count = 64
G = network_pix2pix.generator(d=filter_count)
stat_dict = torch.load('./pix2pix_generator_param1.pkl')
G.load_state_dict(stat_dict)
G.eval()

ensure_path(OUT_FOLDER)

for i in range(len(photo_set)):
    x = photo_set[i]
    x = torch.unsqueeze(x, 0)
    show_result(G, x, i)
    print('writing {}'.format(i))
