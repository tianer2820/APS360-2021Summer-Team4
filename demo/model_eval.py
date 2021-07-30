import os
import torch
import torch.utils.data
import torchvision.transforms.functional as TF

import network
from dataset_utils import get_dataset


MODEL_FOLDER = 'models'
PHOTO_FOLDER = 'photo'
PIXEL_FOLDER = 'pixel'


def ensure_path(path):
        if not os.path.isdir(path):
            os.makedirs(path)


def save_model_test(images: torch.Tensor,
                    E_A: torch.nn.Module,
                    G_B: torch.nn.Module,
                    epoch, device, save_folder):
    with torch.no_grad():
        images = images.to(device)

        realA = images
        latentA = E_A(images)
        genB = G_B(latentA)

        realA = realA.cpu().detach()
        genB = genB.cpu().detach()

        batch = images.shape[0]

    for i in range(batch):
        path = os.path.join(save_folder, '{:0>4}_{:0>2}_input.png'.format(epoch, i))
        img = TF.to_pil_image(realA[i])
        img.save(path)

        path = os.path.join(save_folder, '{:0>4}_{:0>2}_Fake.png'.format(epoch, i))
        img = TF.to_pil_image(genB[i])
        img.save(path)


def eval(normalize_input=True, use_cuda=True, only_first_N=None):

    config = {
    'n_resnet': 5,
    'g_features': 32,
    'd_features': 64,
    'img_size': 256
    }

    # results save path
    a2b_path = 'A2B'
    b2a_path = 'B2A'
    ensure_path(a2b_path)
    ensure_path(b2a_path)

    # data_loader
    dataset_A = get_dataset(PHOTO_FOLDER, config['img_size'], use_normalize=normalize_input)
    dataset_B = get_dataset(PIXEL_FOLDER, config['img_size'], use_normalize=normalize_input)


    # network
    E_A = network.Encoder(input_nc=3, output_nc=3, ngf=config['g_features'], nb=config['n_resnet'])
    E_B = network.Encoder(input_nc=3, output_nc=3, ngf=config['g_features'], nb=config['n_resnet'])
    G_A = network.Generator(input_nc=3, output_nc=3, ngf=config['g_features'], nb=config['n_resnet'])
    G_B = network.Generator(input_nc=3, output_nc=3, ngf=config['g_features'], nb=config['n_resnet'])

    E_A_state_dict = torch.load(os.path.join(MODEL_FOLDER, 'encoderA_param.pkl'))
    E_B_state_dict = torch.load(os.path.join(MODEL_FOLDER, 'encoderB_param.pkl'))
    G_A_state_dict = torch.load(os.path.join(MODEL_FOLDER, 'generatorA_param.pkl'))
    G_B_state_dict = torch.load(os.path.join(MODEL_FOLDER, 'generatorB_param.pkl'))

    E_A.load_state_dict(E_A_state_dict)
    E_B.load_state_dict(E_B_state_dict)
    G_A.load_state_dict(G_A_state_dict)
    G_B.load_state_dict(G_B_state_dict)

    E_A.eval()
    E_B.eval()
    G_A.eval()
    G_B.eval()

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('using CUDA')
    else:
        device = torch.device('cpu')
        print('using CPU')

    E_A.to(device)
    E_B.to(device)
    G_A.to(device)
    G_B.to(device)

    print('---------- Networks Loaded -------------')

    # test A to B
    print('evaluating A2B...')
    loader_A = torch.utils.data.DataLoader(dataset_A, batch_size=1, shuffle=False)
    for i, imgsA in enumerate(loader_A):
        save_model_test(imgsA, E_A, G_B, i, device, a2b_path)
        print('generated {}'.format(i))

        if only_first_N is not None:
            if i > only_first_N:
                break

    # test B to A
    print('evaluating B2A...')
    loader_B = torch.utils.data.DataLoader(dataset_B, batch_size=1, shuffle=False)
    for i, imgsB in enumerate(loader_B):
        save_model_test(imgsB, E_B, G_A, i, device, b2a_path)
        print('generated {}'.format(i))

        if only_first_N is not None:
            if i > only_first_N:
                break

if __name__ == "__main__":
    eval(normalize_input=False, use_cuda=False, only_first_N=20)
