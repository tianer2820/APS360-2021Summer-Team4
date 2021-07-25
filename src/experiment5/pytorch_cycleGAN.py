import os
import time
import network
import util
import itertools
import wandb
import subprocess
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.utils.data

from dataset_utils import get_dataset, InfinateLoader

SAVE_ROOT = 'save_root'
ZIP_PATH = 'save_root.zip'

def train(configures, proj_name, proj_group, test_per_epoch=10, save_per_epoch=50):

    with wandb.init(project=proj_name, group=proj_group, config=configures):
        config = wandb.config

        # results save path
        def ensure_path(path):
            if not os.path.isdir(path):
                os.makedirs(path)

        
        save_root = SAVE_ROOT
        
        model_path = os.path.join(save_root, 'model')
        a2b_path = os.path.join(save_root, 'A2B')
        b2a_path = os.path.join(save_root, 'B2A')
        pre_a2b_path = os.path.join(save_root, 'pretrain', 'A2B')
        pre_b2a_path = os.path.join(save_root, 'pretrain', 'B2A')

        ensure_path(save_root)
        ensure_path(model_path)
        ensure_path(a2b_path)
        ensure_path(b2a_path)
        ensure_path(pre_a2b_path)
        ensure_path(pre_b2a_path)

        # data_loader

        dataset_A = get_dataset('new_data/train/photo/', config.img_size, use_normalize=config.normalize_img)
        dataset_B = get_dataset('new_data/train/pixel/', config.img_size, use_normalize=config.normalize_img)

        loader_A = InfinateLoader(torch.utils.data.DataLoader(
            dataset_A, batch_size=config.batch_size, shuffle=True))
        loader_B = InfinateLoader(torch.utils.data.DataLoader(
            dataset_B, batch_size=config.batch_size, shuffle=True))


        # network
        E_A = network.Encoder(input_nc=3, output_nc=3, ngf=config.g_features, nb=config.n_resnet)
        E_B = network.Encoder(input_nc=3, output_nc=3, ngf=config.g_features, nb=config.n_resnet)
        G_A = network.Generator(input_nc=3, output_nc=3, ngf=config.g_features, nb=config.n_resnet)
        G_B = network.Generator(input_nc=3, output_nc=3, ngf=config.g_features, nb=config.n_resnet)
        D_A = network.discriminator(input_nc=3, output_nc=1, ndf=config.d_features)
        D_B = network.discriminator(input_nc=3, output_nc=1, ndf=config.d_features)
        
        E_A.weight_init(mean=0.0, std=0.02)
        E_B.weight_init(mean=0.0, std=0.02) 
        G_A.weight_init(mean=0.0, std=0.02)
        G_B.weight_init(mean=0.0, std=0.02)
        D_A.weight_init(mean=0.0, std=0.02)
        D_B.weight_init(mean=0.0, std=0.02)

        E_A.train()
        E_B.train()
        G_A.train()
        G_B.train()
        D_A.train()
        D_B.train()

        if config.cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        E_A.to(device)
        E_B.to(device)
        G_A.to(device)
        G_B.to(device)
        D_A.to(device)
        D_B.to(device)

        print('---------- Networks initialized -------------')
        util.print_network(E_A)
        util.print_network(E_B)
        util.print_network(G_A)
        util.print_network(G_B)
        util.print_network(D_A)
        util.print_network(D_B)
        print('-----------------------------------------------')

        # loss
        BCE_loss = nn.BCELoss().to(device)
        MSE_loss = nn.MSELoss().to(device)
        L1_loss = nn.L1Loss().to(device)

        # Adam optimizer
        E_optimizer = optim.Adam(itertools.chain(E_A.parameters(), E_B.parameters()), lr=config.lrG)
        G_optimizer = optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=config.lrG)
        D_A_optimizer = optim.Adam(D_A.parameters(), lr=config.lrD)
        D_B_optimizer = optim.Adam(D_B.parameters(), lr=config.lrD)

        # image store
        fakeA_store = util.ImagePool(50)
        fakeB_store = util.ImagePool(50)

        wandb.watch((E_A, E_B, G_A, G_B, D_A, D_B), log_freq=30)

        print('training start!')
        start_time = time.time()

        for epoch in range(config.epochs):
            D_A_losses = []
            D_B_losses = []
            G_A_losses = []
            G_B_losses = []
            A_cycle_losses = []
            B_cycle_losses = []

            epoch_start_time = time.time()
            # learning rate decay
            if (epoch+1) > config.decay_start:
                D_A_optimizer.param_groups[0]['lr'] -= config.lrD / (config.epochs - config.decay_start)
                D_B_optimizer.param_groups[0]['lr'] -= config.lrD / (config.epochs - config.decay_start)
                G_optimizer.param_groups[0]['lr'] -= config.lrG / (config.epochs - config.decay_start)
                E_optimizer.param_groups[0]['lr'] -= config.lrG / (config.epochs - config.decay_start)

            # train epoch
            for iteration in range(config.epoch_size // config.batch_size):
                # read data
                realA = loader_A.next()
                realB = loader_B.next()

                realA = realA.to(device)
                realB = realB.to(device)

                # train generator G
                G_optimizer.zero_grad()
                E_optimizer.zero_grad()

                # generate real A to fake B; D_A(G_A(A))
                latentA = E_A(realA)
                fakeB = G_B(latentA)
                D_B_result = D_B(fakeB)
                G_B_loss = MSE_loss(D_B_result, torch.ones(D_B_result.size()).to(device))

                # reconstruct fake B to rec A; G_B(G_A(A))
                recA = G_A(latentA)
                A_cycle_loss = L1_loss(recA, realA) * config.lambdaA
                if config.full_cycle_loss:
                    recA_full = G_A(E_B(fakeB))
                    A_cycle_loss_full = L1_loss(recA_full, realA) * config.lambdaA
                    A_cycle_loss += A_cycle_loss_full


                # generate real B to fake A; D_A(G_B(B))
                latentB = E_B(realB)
                fakeA = G_A(latentB)
                D_A_result = D_A(fakeA)
                G_A_loss = MSE_loss(D_A_result, torch.ones(D_A_result.size()).to(device))

                # reconstruct fake A to rec B G_A(G_B(B))
                recB = G_B(latentB)
                B_cycle_loss = L1_loss(recB, realB) * config.lambdaB
                if config.full_cycle_loss:
                    recB_full = G_B(E_A(fakeA))
                    B_cycle_loss_full = L1_loss(recB_full, realB) * config.lambdaA
                    B_cycle_loss += B_cycle_loss_full

                # optimizer step
                G_loss = G_A_loss + G_B_loss + A_cycle_loss + B_cycle_loss
                G_loss.backward()
                G_optimizer.step()
                E_optimizer.step()

                G_A_losses.append(G_A_loss.item())
                G_B_losses.append(G_B_loss.item())
                A_cycle_losses.append(A_cycle_loss.item())
                B_cycle_losses.append(B_cycle_loss.item())

                # train discriminator D_A
                D_A_optimizer.zero_grad()

                D_A_real = D_A(realA)
                D_A_real_loss = MSE_loss(D_A_real, torch.ones(D_A_real.size()).to(device))

                fakeA = fakeA_store.query(fakeA)
                D_A_fake = D_A(fakeA)
                D_A_fake_loss = MSE_loss(D_A_fake, torch.zeros(D_A_fake.size()).to(device))

                D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
                D_A_loss.backward()
                D_A_optimizer.step()

                D_A_losses.append(D_A_loss.item())

                # train discriminator D_B
                D_B_optimizer.zero_grad()

                D_B_real = D_B(realB)
                D_B_real_loss = MSE_loss(D_B_real, torch.ones(D_B_real.size()).to(device))

                # fakeA = fakeA_store.query(fakeA.data)
                fakeB = fakeB_store.query(fakeB)
                D_B_fake = D_B(fakeB)
                D_B_fake_loss = MSE_loss(D_B_fake, torch.zeros(D_B_fake.size()).to(device))

                D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
                D_B_loss.backward()
                D_B_optimizer.step()

                D_B_losses.append(D_B_loss.item())


            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            print(
                '[%d/%d] - ptime: %.2f, loss_D_A: %.3f, loss_D_B: %.3f, loss_G_A: %.3f, loss_G_B: %.3f, loss_A_cycle: %.3f, loss_B_cycle: %.3f' % (
                    (epoch + 1), config.epochs, per_epoch_ptime,
                    torch.mean(torch.FloatTensor(D_A_losses)),
                    torch.mean(torch.FloatTensor(D_B_losses)),
                    torch.mean(torch.FloatTensor(G_A_losses)),
                    torch.mean(torch.FloatTensor(G_B_losses)),
                    torch.mean(torch.FloatTensor(A_cycle_losses)),
                    torch.mean(torch.FloatTensor(B_cycle_losses))))

            log_items = {
                'loss_D_A': torch.mean(torch.FloatTensor(D_A_losses)),
                'loss_D_B': torch.mean(torch.FloatTensor(D_B_losses)),
                'loss_G_A': torch.mean(torch.FloatTensor(G_A_losses)),
                'loss_G_B': torch.mean(torch.FloatTensor(G_B_losses)),
                'loss_cycle_A': torch.mean(torch.FloatTensor(A_cycle_losses)),
                'loss_cycle_B': torch.mean(torch.FloatTensor(B_cycle_losses))
            }
            wandb.log(log_items)

            if (epoch+1) % test_per_epoch == 0:
                # test A to B
                imagesA = []
                for i in range(10):
                    img = dataset_A[i]
                    img: torch.Tensor
                    img = img.unsqueeze(0).cpu().detach()
                    imagesA.append(img)
                imagesA = torch.cat(imagesA, 0)
                util.save_model_test(imagesA, E_A, E_B, G_A, G_B, epoch+1, device, a2b_path)
                # test B to A
                imagesB = []
                for i in range(10):
                    img = dataset_B[i]
                    img: torch.Tensor
                    img = img.unsqueeze(0).cpu().detach()
                    imagesB.append(img)
                imagesB = torch.cat(imagesB, 0)
                util.save_model_test(imagesB, E_B, E_A, G_B, G_A, epoch+1, device, b2a_path)
            
            if (epoch+1) % save_per_epoch == 0:
                torch.save(E_A.state_dict(), os.path.join(model_path, '{:0>4}_encoderA_param.pkl'.format(epoch+1)))
                torch.save(E_B.state_dict(), os.path.join(model_path, '{:0>4}_encoderB_param.pkl'.format(epoch+1)))
                torch.save(G_A.state_dict(), os.path.join(model_path, '{:0>4}_generatorA_param.pkl'.format(epoch+1)))
                torch.save(G_B.state_dict(), os.path.join(model_path, '{:0>4}_generatorB_param.pkl'.format(epoch+1)))
                torch.save(D_A.state_dict(), os.path.join(model_path, '{:0>4}_discriminatorA_param.pkl'.format(epoch+1)))
                torch.save(D_B.state_dict(), os.path.join(model_path, '{:0>4}_discriminatorB_param.pkl'.format(epoch+1)))

        end_time = time.time()
        total_ptime = end_time - start_time

        print("total time: {:.2f}".format(total_ptime))

        print("Training finish!... save training results")

        torch.save(E_A.state_dict(), os.path.join(model_path, 'encoderA_param.pkl'))
        torch.save(E_B.state_dict(), os.path.join(model_path, 'encoderB_param.pkl'))
        torch.save(G_A.state_dict(), os.path.join(model_path, 'generatorA_param.pkl'))
        torch.save(G_B.state_dict(), os.path.join(model_path, 'generatorB_param.pkl'))
        torch.save(D_A.state_dict(), os.path.join(model_path, 'discriminatorA_param.pkl'))
        torch.save(D_B.state_dict(), os.path.join(model_path, 'discriminatorB_param.pkl'))

        # save zip file
        process = subprocess.run(['zip', ZIP_PATH, SAVE_ROOT, '-r'])
        if process.returncode == 0:
            # successed
            if wandb.run.dir != '':
                shutil.copyfile(ZIP_PATH, os.path.join(wandb.run.dir, 'save_root.zip'))
        else:
            wandb.alert('Zip Failed!', 'wandb run zip command failed.')

if __name__ == "__main__":
    config = {
        'lrD': 0.0002,
        'lrG': 0.0002,
        'lambdaA': 10,
        'lambdaB': 10,
        'n_resnet': 6, # 6, for each side(encoder and decoder)
        'epochs': 200, # 200
        'epoch_size': 500, # 500
        'decay_start': 200, # 100
        'g_features': 32, # 32
        'd_features': 72, # 72
        'img_size': 256,
        'batch_size': 6, # 6
        'normalize_img': True,
        'full_cycle_loss': True, # False
        'cuda': False
    }

    train(configures=config, proj_name='cycleGan_test',
          proj_group='experiment5', test_per_epoch=10, save_per_epoch=50)
