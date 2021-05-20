import torch.nn as nn
from collections import OrderedDict

class Flatten_(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def get_LaNet(reduction_value=10):
    LaNet = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1, 6 , 5, padding=2)),
          ('relu1', nn.ReLU()),
          ('maxpool1', nn.MaxPool2d((2, 2), stride=2)),
          ('conv2', nn.Conv2d(6, 16, 5)),
          ('relu2', nn.ReLU()),
          ('maxpool2', nn.MaxPool2d((2, 2), stride=2)),
          ('flatten', Flatten_()),
          ('fc1', nn.Linear(400, 120)),
          ('relu3', nn.ReLU()),
          ('fc2', nn.Linear(120, 84)),
          ('relu3', nn.ReLU()),
          ('fc3', nn.Linear(84, reduction_value)),
        ]))
    return LaNet

def get_CelebA(reduction_value=10):
    return nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),

        nn.Conv2d(64, 128, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),

        nn.Conv2d(128, 256, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),

        nn.Conv2d(256, 512, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),

        Flatten_(),
        nn.Dropout(0.2),
        nn.Linear(512 * 14 * 14, 1024),
        nn.Linear(1024, 256),
        nn.Linear(256, reduction_value)
    )
