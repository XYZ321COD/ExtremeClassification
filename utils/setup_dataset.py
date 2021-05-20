import yaml
from utils.datasets import mnist, celeba


name_to_loader = {
    "MNIST": mnist.get_mnist,
    "CELEBA": celeba.get_celeba
}
file = open(r'config.yaml')
cfg = yaml.load(file, Loader=yaml.FullLoader)


def get_dataset(trial):
    batch_size = trial.suggest_int("batchsize", min(cfg['hyperparameters']['batchsize']), max(cfg['hyperparameters']['batchsize']))
    return name_to_loader[cfg['dataset']['name']](batch_size), batch_size
