
import os
import urllib
from torchvision import datasets
from torchvision import transforms
import torch

def get_cifar(batch_size):

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    BATCHSIZE = batch_size
    DIR = os.getcwd()


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DIR, train=True, download=True, transform=transform_train),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DIR, train=False, transform=transform_test),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader


def get_cifar100(batch_size):

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    BATCHSIZE = batch_size
    DIR = os.getcwd()


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DIR, train=True, download=True, transform=transform_train),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DIR, train=False, transform=transform_test),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader