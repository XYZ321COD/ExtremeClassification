import yaml
from utils.datasets import mnist
from utils.datasets import fashion_mnist


name_to_loader = {
    "MNIST": mnist.get_mnist,
    "FashionMNIST": fashion_mnist.get_fashion_mnist
}
file = open(r'config.yaml')
cfg = yaml.load(file, Loader=yaml.FullLoader)


def get_train_dataset(trial):
    if trial is None:
        batch_size = 128
    else:
        batch_size = trial.suggest_int("batchsize", min(cfg['hyperparameters']['batchsize']), max(cfg['hyperparameters']['batchsize']))
    return name_to_loader[cfg['dataset']['name']](batch_size), batch_size


def get_eval_datasets(trial):
    if trial is None:
        batch_size = 128
    else:
        batch_size = trial.suggest_int("batchsize", min(cfg['hyperparameters']['batchsize']), max(cfg['hyperparameters']['batchsize']))
    return [name_to_loader[name](batch_size) for name in cfg['dataset']['eval_datasets']], batch_size
