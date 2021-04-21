import torch
from typing import Tuple


def freeze_all_except_first(model: torch.nn.Module, device: torch.device, init_fn=None) -> Tuple[torch.nn.Module, torch.Tensor]:
    """
    Freeze all layers except last one, and swap the weights using init_fn.
    """
    if init_fn:
        model[1].A.data = init_fn(model[1].A.size()).to(device)
        model[1].A.data.requires_grad = True
    W: torch.Tensor = model[-1].A.data.clone()
    for param in model[1].parameters():
        param.requires_grad = True

    for param in model[0].parameters():
        param.requires_grad = False

    return model, W


def freeze_first(model: torch.nn.Module, device: torch.device, init_fn=None) -> Tuple[torch.nn.Module, torch.Tensor]:
    """
    Freeze last layer, and swap the weights using init_fn.
    """
    if init_fn:
        model[1].A.data = init_fn(model[1].A.size()).to(device)
        model[1].A.data.requires_grad = True
    W: torch.Tensor = model[-1].A.data.clone()
    for param in model[1].parameters():
        param.requires_grad = False
    for param in model[0].parameters():
        param.requires_grad = True

    return model, W
