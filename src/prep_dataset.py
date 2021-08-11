import os
join = os.path.join
import shutil
from PIL import Image

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rename_file(original, new):
    name, ext = os.path.splitext(original)
    return new + '.png'


def prep_image(path, target_size, scale_ratio) -> Image.Image:
    img = Image.open(path)
    img = img.convert('RGB')
    w, h = img.size
    min_edge = min(w, h)
    target_min = target_size * scale_ratio
    if min_edge < target_size:
        scale = target_size / min_edge
    else:
        scale = target_min / min_edge
    w2 = int(w * scale)
    h2 = int(h * scale)
    img = img.resize((w2, h2), resample=Image.NEAREST)
    return img


def prep_dataset(data_root, out_root, target_size, scale_ratio, train_only=False):
    files = os.listdir(data_root)

    is_dir = lambda file: os.path.isdir(join(data_root, file))
    categories = list(filter(is_dir, files))
    i = 0

    for category in categories:
        data_folder = join(data_root, category)

        files = os.listdir(data_folder)
        supported_ext = ('.jpg', '.png')
        is_supported = lambda file: os.path.splitext(file)[1].lower() in supported_ext
        good_files = list(filter(is_supported, files))

        name_len = 6
        if not train_only:
            total_num = len(good_files)
            train_num = int(total_num * 0.7)
            val_num = int(total_num * 0.15)
            test_num = total_num - train_num - val_num
            
            train_files = good_files[0:int(total_num*0.7)]
            val_files = good_files[int(total_num*0.7):int(total_num*0.85)]
            test_files = good_files[int(total_num*0.85):]

            dest_folder = join(out_root, 'train', category)
            ensure_folder(dest_folder)
            for file in train_files:
                name = rename_file(file, '{:0>6}'.format(i))

                img = prep_image(join(data_folder, file), target_size, scale_ratio)
                img.save(join(dest_folder, name))

                # shutil.copy(join(data_folder, file), join(dest_folder, name))
                i += 1
                print('#{}: copying {}'.format(i, file))
            
            dest_folder = join(out_root, 'val', category)
            ensure_folder(dest_folder)
            for file in val_files:
                name = rename_file(file, '{:0>6}'.format(i))

                img = prep_image(join(data_folder, file), target_size, scale_ratio)
                img.save(join(dest_folder, name))

                # shutil.copy(join(data_folder, file), join(dest_folder, name))
                i += 1
                print('#{}: copying {}'.format(i, file))

            dest_folder = join(out_root, 'test', category)
            ensure_folder(dest_folder)
            for file in test_files:
                name = rename_file(file, '{:0>6}'.format(i))

                img = prep_image(join(data_folder, file), target_size, scale_ratio)
                img.save(join(dest_folder, name))

                # shutil.copy(join(data_folder, file), join(dest_folder, name))
                i += 1
                print('#{}: copying {}'.format(i, file))
        else:
            dest_folder = join(out_root, 'train', category)
            ensure_folder(dest_folder)
            for file in good_files:
                name = rename_file(file, '{:0>6}'.format(i))

                img = prep_image(join(data_folder, file), target_size, scale_ratio)
                img.save(join(dest_folder, name))

                # shutil.copy(join(data_folder, file), join(dest_folder, name))
                i += 1
                print('#{}: copying {}'.format(i, file))


if __name__ == "__main__":
    data_root = './eval_data/'
    out_root = './eval_data2/'
    target_size = 256
    scale_ratio = 1
    prep_dataset(data_root, out_root, target_size, scale_ratio, train_only=True)
