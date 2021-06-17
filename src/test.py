from data import get_raw_datasets, split_dataset, ImageDataset

import torch
import torch.utils.data

# test code
comic, pixel = get_raw_datasets()
comic_train, comic_valid, comic_test = split_dataset(comic)

loader = torch.utils.data.DataLoader(comic_train, batch_size=32)

for batch in loader:
    print(batch.shape)
    break
