from utils.datasets import mnist, cifar, fmnist
import yaml
name_to_loader = {"MNIST": mnist.get_mnist, "CIFAR10": cifar.get_cifar, "FMNIST": fmnist.get_fashion_mnist, "CIFAR100": cifar.get_cifar100}
file = open(r'config.yaml')
cfg = yaml.load(file, Loader=yaml.FullLoader)
def get_dataset(trial):
    if trial is None:
        batch_size = 128
    else:
        batch_size = trial.suggest_int("batchsize", min(cfg['hyperparameters']['batchsize']), max(cfg['hyperparameters']['batchsize']))
    return name_to_loader[cfg['dataset']['name']](batch_size), batch_size

def get_datasets(trial, name):
    if trial is None:
        batch_size = 128
    if name=='MNIST-FMNIST':
        return name_to_loader['MNIST'](batch_size), name_to_loader['FMNIST'](batch_size)
    if name=='CIFAR10-CIFAR100':
        return name_to_loader['CIFAR10'](batch_size), name_to_loader['CIFAR100'](batch_size)
