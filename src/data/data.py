import torch
import torch.utils.data

from .dataset import ImageDataset


def get_raw_datasets():
    comic_data = ImageDataset('../data/trainA')
    pixel_data = ImageDataset('../data/trainB')
    return comic_data, pixel_data


def split_dataset(dataset):
    total = len(dataset)
    train = int(total * 0.7)
    valid = int(total * 0.15)
    test = total - train - valid

    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, (train, valid, test))
    return train_set, valid_set, test_set

