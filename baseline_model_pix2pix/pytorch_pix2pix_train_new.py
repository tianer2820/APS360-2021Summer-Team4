from dataset import get_dataset
import wandb
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch
import util_new
import network_new
import os
import time
import pickle
join = os.path.join


config = {
    'lr_D': 0.00001,
    'lr_G': 0.00003,
    'batch_size': 16,
    'filter_count': 64,  # 64
    'epochs': 150,
    'save_freq': 50,
    'use_cuda': True,
    'epoch_size': 300
}

# wandb
with wandb.init(project='pix2pix_baseline', group='parameter_search', config=config):
    config = wandb.config

    # prep save folders
    root = join(wandb.run.dir, 'save_root')
    if wandb.run.dir == '':
        root = './save_root'
    print('saving to: {}'.format(root))
    result_path = join(root, 'results')
    save_path = join(root, 'saves')

    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # data_loader
    pixel_set = get_dataset('new_data/train/pixel', 256)
    photo_set = get_dataset('new_data/train/photo', 256)

    pixel_loader = torch.utils.data.DataLoader(
        pixel_set, batch_size=config.batch_size, shuffle=True)
    photo_loader = torch.utils.data.DataLoader(
        photo_set, batch_size=config.batch_size, shuffle=True)

    pixel_loader_cycle = util_new.CycleIter(pixel_loader)
    photo_loader_cycle = util_new.CycleIter(photo_loader)

    fixed_xs = []
    for i in range(6):
        fixed_xs.append(photo_set[i].unsqueeze(0))
    fixed_x_ = torch.cat(fixed_xs, dim=0)

    # GPU
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # network
    G = network_new.generator(d=config.filter_count)
    D = network_new.discriminator(d=config.filter_count)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.to(device)
    D.to(device)
    G.train()
    D.train()

    # loss
    BCE_loss = nn.BCELoss().to(device)
    L1_loss = nn.L1Loss().to(device)

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=config.lr_G)
    D_optimizer = optim.Adam(D.parameters(), lr=config.lr_D)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    #######################
    print('training start!')

    start_time = time.time()
    wandb.watch((G, D), log_freq=100)

    for epoch in range(config.epochs):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        num_iter = 0

        for batch_num in range(config.epoch_size // config.batch_size):
            x_ = photo_loader_cycle.get()
            y_ = pixel_loader_cycle.get()
            # x is photo, y is pixel art, generator will convert x to y

            # train discriminator D
            D.zero_grad()

            x_, y_ = x_.to(device), y_.to(device)

            D_result = D(y_).squeeze()
            D_real_loss = BCE_loss(
                D_result, torch.ones(D_result.size()).to(device))

            G_result = G(x_)
            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(
                D_result, torch.zeros(D_result.size()).to(device))

            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            D_optimizer.step()

            train_hist['D_losses'].append(D_train_loss.item())

            D_losses.append(D_train_loss.item())

            # train generator G
            G.zero_grad()

            G_result = G(x_)
            D_result = D(G_result).squeeze()

            G_train_loss = BCE_loss(
                D_result, torch.ones(D_result.size()).to(device))
            G_train_loss.backward()
            G_optimizer.step()

            train_hist['G_losses'].append(G_train_loss.item())

            G_losses.append(G_train_loss.item())

            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), config.epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                    torch.mean(torch.FloatTensor(G_losses))))
        wandb.log({
            'loss_d': torch.mean(torch.FloatTensor(D_losses)),
            'loss_G': torch.mean(torch.FloatTensor(G_losses))
        })
        with torch.no_grad():
            util_new.show_result(G, fixed_x_.to(device), epoch+1,
                                save=True,
                                path=join(result_path, '{:0>4}.png'.format(epoch + 1)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        # periodic saves
        if (epoch+1) % config.save_freq == 0:
            torch.save(G.state_dict(),
                    join(save_path, 'generator_param_{}.pkl'.format(epoch+1)))
            torch.save(D.state_dict(), 
                    join(save_path, 'discriminator_param_{}.pkl'.format(epoch+1)))

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(
        torch.FloatTensor(train_hist['per_epoch_ptimes'])), config.epochs, total_ptime))

    print("Training finish!... save training results")
    torch.save(G.state_dict(), join(root, 'generator_param.pkl'))
    torch.save(D.state_dict(), join(root, 'discriminator_param.pkl'))

    with open(join(root, 'train_hist.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)

    util_new.show_train_hist(train_hist, save=True,
                            path=join(root, 'train_hist.png'))

