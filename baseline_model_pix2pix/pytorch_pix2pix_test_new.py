from dataset import get_dataset
import torch.utils.data
import torch
import torchvision.transforms.functional as TF
import network_new
import os

join = os.path.join
from PIL import Image



def show_result(G, x_):
    G.eval()
    G.cpu()
    x_ = x_.cpu()
    test_images = G(x_)

    
    for i in range(x_.shape[0]):
        # img_np = (test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2
        # img = Image.fromarray(img_np)
        img = TF.to_pil_image(test_images[i])
        img.save('./save{:0>4}.png'.format(i))
        img.show()



pixel_set = get_dataset('new_data/train/pixel', 256)
photo_set = get_dataset('new_data/train/photo', 256)

fixed_xs = []
for i in range(6):
    fixed_xs.append(photo_set[i].unsqueeze(0))
fixed_x_ = torch.cat(fixed_xs, dim=0)

filter_count = 64
G = network_new.generator(d=filter_count)

stat_dict = torch.load('./generator_param1.pkl')
G.load_state_dict(stat_dict)

show_result(G, fixed_x_)
