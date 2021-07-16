from data import get_raw_datasets, split_dataset, ImageDataset
from model import ResidualBlock, gen, dis
from training_from_colab import train
import torch
import torch.utils.data

# test code
comic, pixel = get_raw_datasets()
comic_train, comic_valid, comic_test = split_dataset(comic)
pixel_train, pixel_valid, pixel_test = split_dataset(pixel)

comic_loader = torch.utils.data.DataLoader(comic_train, batch_size=32)
pixel_loader = torch.utils.data.DataLoader(comic_train, batch_size=32)

'''for batch in loader:
    print(batch.shape)
    break'''
Gen_1 = gen(3)
Dis_1 = dis(3)
Gen_2 = gen(3)
Dis_2 = dis(3)
train(Gen_1, Dis_1,Gen_2, Dis_2, comic_loader,pixel_loader, num_epochs = 5, batch_size = 32)

