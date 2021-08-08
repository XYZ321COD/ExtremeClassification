import torch.nn as nn
import torch.nn as nn

def str_to_loss(string_loss):
    mapping = {"BCELoss": nn.BCELoss(reduce='sum'), "CrossEntropyLoss" : nn.CrossEntropyLoss(reduce='mean')}
    return mapping[string_loss]