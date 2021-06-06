import wandb
import os

run = wandb.init(project='TestProject', entity="summer2021-aps360-team4")
try:
    config = wandb.config
    config.learning_rate = 0.01
    config.batch_size = 20
    config.loss = 'MSE'
    config.optimizer = 'Adam'

    for i in range(64):
        wandb.log({"loss": i**2})


    for i in range(3):
        path = os.path.join(wandb.run.dir, "testfile{}.txt".format(i))
        path = os.path.abspath(path)
        with open(path, 'w', encoding='utf8') as f:
            f.write("Hello world!")
            print(path)

    print("done")
    run.finish(0)
except:
    run.finish(1)
    raise
