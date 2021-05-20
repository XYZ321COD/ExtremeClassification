import os
import urllib
from torchvision import datasets
from torchvision import transforms
import torch


def get_mnist(batch_size):
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    BATCHSIZE = batch_size
    DIR = os.getcwd()


    transform = transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize(
                                    (0.1307,), (0.3081,))])

    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transform),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, transform=transform),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader

