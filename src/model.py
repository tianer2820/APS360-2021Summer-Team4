import torch
from torch import nn
import torch.nn.functional as F


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, x):
        return x


if __name__ == "__main__":
    # test
    mymodel = TestModel()
    