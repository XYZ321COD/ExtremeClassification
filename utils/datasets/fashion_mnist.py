import os
from torchvision import datasets
from torchvision import transforms
import torch


def get_fashion_mnist(batch_size):
    BATCHSIZE = batch_size
    DIR = os.getcwd()
    transform = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transform),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transform),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader
