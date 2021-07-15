import torch
import torchvision

from PIL import Image
import os
join = os.path.join


class ImageDataset:
    def __init__(self, folder, ext=('png', 'jpg')) -> None:
        folder = os.path.abspath(folder)

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
            raise FileNotFoundError('No valid file in folder {}'.format(folder))
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, i):
        file = self.file_list[i]
        full_path = join(self.source_folder, file)
        img = Image.open(full_path)
        img = img.convert('RGB')
        #????
        tensor = torchvision.transforms.transforms.F.to_tensor(img)
        return tensor


if __name__ == '__main__':
    # test code
    import torch.utils.data

    dataset = ImageDataset('../data/trainA')
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for batch in loader:
        print(batch.shape)
        break
