from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data
from torch import nn
import itertools

def train(G_p2c, D_p2c, G_c2p, D_c2p, loader_Comic, loader_Pixel, lr=0.002, batch_size=64, num_epochs=20):

    rand_size = 100;

    # optimizers for generator and discriminator

    #chain the parameters of the discriminators together:
    d_p2c_optimizer = optim.Adam(D_p2c.parameters(), lr)
    d_c2p_optimizer = optim.Adam(D_c2p.parameters(), lr)
    g_optimizer = optim.Adam(itertools.chain(D_p2c.parameters(), D_c2p.parameters()), lr)
    

    # define losses function
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_Cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()  #this is optional??

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # fixed data for testing
    sample_size=16
    test_noise = np.random.uniform(-1, 1, size=(batch_size,3,256,256))
    test_noise = torch.from_numpy(test_noise).float()
    #print(test_noise.shape)
    #x = input("w")

    for epoch in range(num_epochs):
        #D.train()
        #G.train()
        #training mode
        G_p2c.train()
        D_p2c.train()
        G_c2p.train()
        D_c2p.train()
        pixel_iterator = iter(loader_Pixel)
        for batch_i, comic_images in enumerate(loader_Comic):
                    
            batch_size = comic_images.size(0)
            
            # rescale images to range -1 to 1
            comic_images = comic_images*2 - 1

            #getting the pixel images:https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
            pixel_images = next(pixel_iterator)
            
            
            # === Train the Discriminator === 
            ######################################discriminator c2p#######################################
            d_c2p_optimizer.zero_grad()

            # discriminator losses on real images 
            D_real_comic = D_c2p(comic_images)
            labels = torch.ones(batch_size)  ##??????
            print("\nfirst\n")
            d_real_loss_c = criterion_GAN(D_real_comic.squeeze(), labels)
            
            # discriminator losses on fake images
            z = np.random.uniform(-1, 1, size=(batch_size,3,256,256)) # size of the input to generator
            # the 1 is the number of channels
            #print("z:", z.shape)
            z = torch.from_numpy(z).float()
            
            fake_comic_images = G_p2c(z)

            D_fake_comic = D_c2p(fake_comic_images)
            labels = torch.zeros(batch_size) # fake labels = 0
            print("\nfirst\n")
            d_fake_loss_c = criterion_GAN(D_fake_comic.squeeze(), labels)
            
            # add up losses and update parameters
            d_c2p_loss = d_real_loss_c + d_fake_loss_c
            d_c2p_loss.backward()
            d_c2p_optimizer.step()


            ######################################discriminator p2c#######################################
            d_p2c_optimizer.zero_grad()

            # discriminator losses on real images 
            D_real_pixel = D_c2p(pixel_images)
            labels = torch.ones(batch_size)  ##??????
            print("\nsecond\n")
            d_real_loss_p = criterion_GAN(D_real_pixel.squeeze(), labels)
            
            # discriminator losses on fake images
            z = np.random.uniform(-1, 1, size=(batch_size,3,256,256)) # size of the input to generator
            # the 1 is the number of channels
            #print("z:", z.shape)
            z = torch.from_numpy(z).float()
            
            fake_pixel_images = G_c2p(z)

            D_fake_pixel = D_p2c(fake_pixel_images)
            labels = torch.zeros(batch_size) # fake labels = 0
            print("\nsecond\n")
            d_fake_loss_p = criterion_GAN(D_fake_pixel.squeeze(), labels)
            
            # add up losses and update parameters
            d_p2c_loss = d_real_loss_p + d_fake_loss_p
            d_p2c_loss.backward()
            d_p2c_optimizer.step()
            
            ######################################################generator training###########################################
            # === Train the Generator ===
            
            g_optimizer.zero_grad()
            
            # generator losses on fake images
            z = np.random.uniform(-1, 1, size=(batch_size,3,256,256))
            #print("z", z.shape)
            z = torch.from_numpy(z).float()

            ##########################GAN losses#################################3

            ###########G_c2p()############
            fake_comic = G_p2c(pixel_images)
            prediction_c = D_c2p(fake_comic)
            label = torch.ones(batch_size) ###why are we using ones here??
            print("\nthird\n")
            loss_gan_p2c = criterion_GAN(prediction_c.squeeze(), label)
            ###########G_p2c()#############
            fake_pixel = G_c2p(comic_images)
            prediction_p = D_p2c(fake_pixel)
            label = torch.ones(batch_size) ###why are we using ones here?
            print("\nthird\n")#?
            loss_gan_c2p = criterion_GAN(prediction_p.squeeze(), label)

            ##########################cycle losses#################################
            cycled_pixel = G_c2p(fake_comic)
            print("\nthird\n")
            print(cycled_pixel.shape)
            print(pixel_images.shape)
            loss_cycle_pcp = criterion_Cycle(cycled_pixel, pixel_images) * 10.0
            cycled_comic = G_p2c(fake_pixel)
            print("\nthird\n")
            loss_cycle_cpc = criterion_Cycle(cycled_comic, comic_images) * 10.0

            ##########################identity losses##############################
            same_comic = G_p2c(comic_images)
            loss_identity_comic = criterion_identity(same_comic, comic_images) * 5.0 #参数
            same_pixel = G_c2p(pixel_images)
            loss_identity_pixel = criterion_identity(same_pixel, pixel_images) *5.0

            # compute loss and update parameters
            g_loss = loss_gan_p2c + loss_gan_c2p + loss_cycle_cpc + loss_cycle_pcp + loss_identity_comic + loss_identity_pixel
            g_loss.backward()
            g_optimizer.step()







        ###################################33# print loss##############################333
        print('Epoch [%d/%d], d_p2c_loss: %.4f, d_c2p_loss: %.4f, g_loss: %.4f, ' 
              % (epoch + 1, num_epochs, d_p2c_loss.item(), d_c2p_loss.item(), g_loss.item()))

        # append discriminator loss and generator loss
        losses.append((d_p2c_loss.item(), d_c2p_loss.item(), g_loss.item()))
        #print("\n\n\n\n\n\n\no \n\n\n")
        # plot images
        G_c2p.eval()
        D_c2p.eval()
        G_p2c.eval()
        D_p2c.eval()
        test_images = G_c2p(comic_images)  ## why are we plotting noise??????
        #print(test_images[1:])
        #x = input("wait")

        plt.figure(figsize=(9, 3))
        for k in range(16):
            plt.subplot(2, 8, k+1)
            #print(test_images[k,:].shape)
            #reconstructing the shape of images:
            print(test_images.shape)
            haha = test_images[k,:].data.numpy().swapaxes(0,2)
            print(haha.shape)
            plt.imshow(test_images[k,:].data.numpy().swapaxes(0,2))
        plt.show()
    
    return losses


