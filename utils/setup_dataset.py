from utils.datasets import mnist
import yaml
name_to_loader = {"MNIST": mnist.get_mnist}
file = open(r'config.yaml')
cfg = yaml.load(file, Loader=yaml.FullLoader)
def get_dataset(trial):
    if trial is None:
        batch_size = 128
    else:
        batch_size = trial.suggest_int("batchsize", min(cfg['hyperparameters']['batchsize']), max(cfg['hyperparameters']['batchsize']))
    return name_to_loader[cfg['dataset']['name']](batch_size), batch_size