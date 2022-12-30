from torch import nn


class Network(nn.Module):
    def __init__(self, x):
        super(Network, self).__init__()
        self.x = x
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, x),
        )

    def forward(self, x):
        x = self.model(x)
        return x
