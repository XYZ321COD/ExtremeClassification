from utils.models import LaNet
import yaml
import torch.nn as nn
from utils.aggregation_layer import MaxReductionLayer, ProdReductionLayer
name_to_model = {
    "LaNet": LaNet.get_LaNet,
    "CelebA": LaNet.get_CelebA
}

file = open(r'config.yaml')
cfg = yaml.load(file, Loader=yaml.FullLoader)


def add_aggregation_to_model(model, input_size, output_size):
    reduction_layer = MaxReductionLayer if cfg['hyperparameters']['aggregation_method'] == 'max' else None
    reduction_layer = ProdReductionLayer if cfg['hyperparameters']['aggregation_method'] == 'prod' else None
    if reduction_layer is None:
        reduction_layer = nn.Linear(input_size, output_size)
        model = nn.Sequential(model, reduction_layer)
        model = nn.Sequential(model, nn.Sigmoid())
        return model
    return nn.Sequential(model, reduction_layer(input_size, output_size))


def get_model(trial):
    reduction_value = trial.suggest_int("reduction_value", min(cfg['hyperparameters']['reduction_value']), max(cfg['hyperparameters']['reduction_value']))
    num_of_classes = cfg['dataset']['number_of_classes']
    if cfg['options']['add_reduction_layer']:        
        model = name_to_model[cfg['options']['model']](reduction_value)
        return add_aggregation_to_model(model, reduction_value, num_of_classes), reduction_value

    return name_to_model[cfg['options']['model']](num_of_classes), reduction_value
