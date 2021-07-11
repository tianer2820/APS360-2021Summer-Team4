from typing import Callable
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

import torch
import torchvision

from PIL import Image
import os
join = os.path.join


class ImageDataset:
    def __init__(self, folder, loader: Callable, transform: Callable, ext=('png', 'jpg')) -> None:
        folder = os.path.abspath(folder)
        self.loader = loader
        self.transform = transform

        # check folder
        if not os.path.isdir(folder):
            raise ValueError('cant find folder {}'.format(folder))

        # format the extention list
        ext = list(ext)
        for i in range(len(ext)):
            ext[i] = ext[i].strip(' .')

        # save attributes
        self.source_folder = folder
        self.exts = ext

        # scan folder
        self.file_list = []
        ls = os.listdir(folder)
        for file in ls:
            if not os.path.isfile(join(folder, file)):
                continue  # ignore folders

            name, extention = os.path.splitext(file)
            extention = extention.strip(' .')
            if extention in self.exts:
                # supported file
                self.file_list.append(file)
        # check file count
        if len(self.file_list) == 0:
            raise FileNotFoundError(
                'No valid file in folder {}'.format(folder))

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, i):
        file = self.file_list[i]
        full_path = join(self.source_folder, file)
        loaded = self.loader(full_path)
        transformed = self.transform(loaded)
        return transformed


def get_dataset(root_folder, target_size, use_normalize=False):

    random_crop = torchvision.transforms.RandomCrop(
        target_size, pad_if_needed=True)
    random_flip = torchvision.transforms.RandomHorizontalFlip()
    normalize = torchvision.transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def loader(path: str):
        img = Image.open(path)
        img: Image.Image
        img = img.convert('RGB')
        if use_normalize:
            return normalize(TF.to_tensor(img))
        else:
            return TF.to_tensor(img)

    def transform(tensor: torch.Tensor):
        tensor = random_crop(tensor)
        tensor = random_flip(tensor)
        return tensor

    # dataset = torchvision.datasets.ImageFolder(
    #     root_folder, transform, loader=loader)
    dataset = ImageDataset(root_folder, loader=loader, transform=transform)
    return dataset


if __name__ == '__main__':
    import torch.utils.data
    dataset = get_dataset('new_data/train/pixel', 256)
    loader = torch.utils.data.DataLoader(dataset, batch_size=7)
    

    for data in loader:
        print(data)
        break
        # img = data[0]
        # img: torch.Tensor
        # img = TF.to_pil_image(img)
        # img: Image.Image
        # img.save('./save{}.png'.format(i))
