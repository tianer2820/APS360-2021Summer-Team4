import itertools, torch, random
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import os


def save_model_test(images: torch.Tensor, modelA: torch.nn.Module, modelB: torch.nn.Module,
               epoch, device, base_path):
    images = images.to(device)

    realA = images
    genB = modelA(images)
    recA = modelB(genB)

    realA = realA.cpu().detach()
    genB = genB.cpu().detach()
    recA = recA.cpu().detach()

    batch = images.shape[0]

    for i in range(batch):
        path = os.path.join(base_path, 'EP{:0>4}_{:0>2}_input.png'.format(epoch, i))
        img = TF.to_pil_image(realA[i])
        img.save(path)

        path = os.path.join(base_path, 'EP{:0>4}_{:0>2}_Fake.png'.format(epoch, i))
        img = TF.to_pil_image(genB[i])
        img.save(path)

        path = os.path.join(base_path, 'EP{:0>4}_{:0>2}_Recon.png'.format(epoch, i))
        img = TF.to_pil_image(recA[i])
        img.save(path)

def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    test_images = G(x_)

    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5), squeeze=False)
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class image_store():
    def __init__(self, store_size=50):
        self.store_size = store_size
        self.num_img = 0
        self.images = []

    def query(self, image):
        select_imgs = []
        for i in range(image.size()[0]):
            if self.num_img < self.store_size:
                self.images.append(image)
                select_imgs.append(image)
                self.num_img += 1
            else:
                prob = np.random.uniform(0, 1)
                if prob > 0.5:
                    ind = np.random.randint(0, self.store_size - 1)
                    select_imgs.append(self.images[ind])
                    self.images[ind] = image
                else:
                    select_imgs.append(image)

        return Variable(torch.cat(select_imgs, 0))

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images