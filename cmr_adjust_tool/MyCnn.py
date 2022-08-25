import torch
from torch import nn


# 搭建神经网络
class MyCnn(nn.Module):
    def __init__(self):
        super(MyCnn, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*4*4, 32),
            nn.Linear(32, 8)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    mymodel = MyCnn()
    input = torch.ones((64, 1, 128, 128))
    output = mymodel(input)
    print(output.shape)
