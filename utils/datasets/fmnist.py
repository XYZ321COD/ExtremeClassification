
import os
import urllib
from torchvision import datasets
from torchvision import transforms
import torch

def get_fashion_mnist(batch_size):

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    BATCHSIZE = batch_size
    DIR = os.getcwd()


    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transform_train),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transform_test),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader