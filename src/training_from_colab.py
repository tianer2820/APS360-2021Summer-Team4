from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data
from torch import nn

def train(G, D, train_loader, lr=0.002, batch_size=64, num_epochs=20):

    rand_size = 100;

    # optimizers for generator and discriminator
    d_optimizer = optim.Adam(D.parameters(), lr)
    g_optimizer = optim.Adam(G.parameters(), lr)
 
    # define loss function
    criterion = nn.BCEWithLogitsLoss()

    # get the training datasets
    #train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

    # prepare data loader
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    #train_loader = get_data_loader()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # fixed data for testing
    sample_size=16
    test_noise = np.random.uniform(-1, 1, size=(batch_size,3,256,256))
    test_noise = torch.from_numpy(test_noise).float()
    print(test_noise.shape)
    #x = input("w")

    for epoch in range(num_epochs):
        D.train()
        G.train()
        
        for batch_i, real_images in enumerate(train_loader):
                    
            batch_size = real_images.size(0)
            
            # rescale images to range -1 to 1
            real_images = real_images*2 - 1
            
            # === Train the Discriminator ===
            
            d_optimizer.zero_grad()

            # discriminator losses on real images 
            D_real = D(real_images)
            labels = torch.ones(batch_size)
            d_real_loss = criterion(D_real.squeeze(), labels)
            
            # discriminator losses on fake images
            z = np.random.uniform(-1, 1, size=(batch_size,3,256,256)) # size of the input to generator
            # the 1 is the number of channels
            #print("z:", z.shape)
            z = torch.from_numpy(z).float()
            
            fake_images = G(z)

            D_fake = D(fake_images)
            labels = torch.zeros(batch_size) # fake labels = 0
            d_fake_loss = criterion(D_fake.squeeze(), labels)
            
            # add up losses and update parameters
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            

            # === Train the Generator ===
            g_optimizer.zero_grad()
            
            # generator losses on fake images
            z = np.random.uniform(-1, 1, size=(batch_size,3,256,256))
            #print("z", z.shape)
            z = torch.from_numpy(z).float()
            fake_images = G(z)
          
            D_fake = D(fake_images)
            labels = torch.ones(batch_size) #flipped labels

            # compute loss and update parameters
            g_loss = criterion(D_fake.squeeze(), labels)
            g_loss.backward()
            g_optimizer.step()

        # print loss
        print('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
              % (epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))
        #print("\n\n\n\n\n\n\no \n\n\n")
        # plot images
        G.eval()
        D.eval()
        test_images = G(test_noise)
        print(test_images[1:])
        #x = input("wait")

        plt.figure(figsize=(9, 3))
        for k in range(16):
            plt.subplot(2, 8, k+1)
            #print(test_images[k,:].shape)
            #reconstructing the shape of images:
            plt.imshow(test_images[k,:].data.numpy().swapaxes(0,2))
        plt.show()
    
    return losses


