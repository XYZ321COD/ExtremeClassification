"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.
"""
import torch.nn as nn
from collections import OrderedDict
from aggregation_layer import Reduction_Layer
class Flatten_(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def define_model():
    network_MNIST = nn.Sequential(OrderedDict([
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
          ('fc3', nn.Linear(84, 10)),
          ('sigmoid1', nn.Sigmoid())
        ]))

    return network_MNIST

def add_aggregation_to_model(model, input_size, output_size):
    return nn.Sequential(model, Reduction_Layer(input_size, output_size))