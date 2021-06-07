import os
import wandb
import torch
import torchvision

from torch import nn
import torch.nn.functional as F
import torch.utils.data


def train(confs: dict, USE_CUDA=True, SINGLE_BATCH_ONLY=False, CONTINUE_TRAINING=True, SAVE_FREQ=100):

    # 1. Start a W&B run
    with wandb.init(project='TestProject', config=confs) as run:
        config = run.config

        # 2. prepare data and model
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

        # 3. load previous weights
        if CONTINUE_TRAINING:
            # load previous weights here
            pass

        # 4. prepare optimizer and loss
        if config.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criteria = F.mse_loss

        # 5. detect CUDA
        if USE_CUDA and torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA available, using GPU")
        else:
            device = torch.device('cpu')
            print("using CPU")
        model = model.to(device)

        # 6. Train
        wandb.watch(model, log_freq=100)
        print("start training...")
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
                    break  # for debug purpose

            # end of one epoch, print information and log to wandb
            average_loss = sum(train_losses) / len(train_losses)
            wandb.log({"loss": average_loss})
            print("Epoch {}: loss: {}".format(epoch, average_loss))

            # save snapshots
            if (epoch + 1) % SAVE_FREQ == 0:
                snapshots_dir = os.path.join(wandb.run.dir, 'snapshots/')
                snapshot_name = 'snapshot_{:0>4}.pkl'.format(epoch)

                if not os.path.isdir(snapshots_dir):
                    os.makedirs(snapshots_dir)

                torch.save(model.state_dict(), os.path.join(
                    snapshots_dir, snapshot_name))

                print("snapshot saved to: {}".format(snapshot_name))

        print("Done")


if __name__ == "__main__":
    conf_dict = {
        'lr': 0.01,
        'batch_size': 20,
        'loss': 'MSE',
        'optimizer': 'Adam'
    }

    train(conf_dict)
