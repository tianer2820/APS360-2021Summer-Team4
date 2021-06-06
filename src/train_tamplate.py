import os
import wandb
import torch
import torchvision

from torch import nn
import torch.nn.functional as F
import torch.utils.data


USE_CUDA = True
SINGLE_BATCH_ONLY = False
CONTINUE_TRAINING = True
SAVE_FREQ = 100

# 1. Start a W&B run
run = wandb.init(project='TestProject')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01
config.batch_size = 20
config.loss = 'MSE'
config.optimizer = 'Adam'

# 3. prepare data and model
data_set = torchvision.datasets.DatasetFolder("some_folder")
data_loader = torch.utils.data.DataLoader(
    data_set, config.batch_size, shuffle=False)
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
# maybe load previous weights here
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criteria = F.mse_loss

# 4. Test CUDA
if USE_CUDA and torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA available, using GPU")
else:
    device = torch.device('cpu')
    print("using CPU")

# 5. Train
for epoch in range(100):
    # store training loss and accuracy
    train_losses = []

    # iterate batchs
    for batch in data_loader:
        datas = batch[0].to(device)
        lables = batch[1].to(device)

        outs = model(datas)
        loss = criteria(outs, lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store losses
        train_losses.append(loss.item())

        if SINGLE_BATCH_ONLY:
            break # for debug purpose
    
    # end of one epoch, print information (maybe)
    # log to wandb
    average_loss = sum(train_losses) / len(train_losses)
    wandb.log({"loss": loss})

    # save snapshot
    if (epoch + 1) % SAVE_FREQ == 0:
        # save the file here
        torch.save(model.state_dict(), "./snapshot/path/file.pkl")

# save snapshots to wandb
files = os.listdir("./snapshot/path/")
for file in files:
    wandb.save("./snapshot/path/{}".format(file))

run.finish(0)
print("Done")
