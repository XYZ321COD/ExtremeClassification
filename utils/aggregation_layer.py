import abc

import torch


def max_aggregate(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Performs aggregation using max operator for given weights and input tensors.
    """
    return (W * x.unsqueeze(dim=-1)).max(dim=1)[0]


def prod_aggregate(W: torch.Tensor, x: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    """
    Performs aggregation using product of probabilities for given weights and input tensors.
    """
    out = 1.0 - torch.pow((W * x.unsqueeze(dim=-1)), alpha)
    return torch.pow(1.0 - out.prod(dim=1), 1 / alpha)


class ReductionLayer(abc.ABC, torch.nn.Module):
    """
    Abstract custom reduction layer class containing logic
    for weight initialization.
    """

    def __init__(self, size_in: int, size_out: int) -> None:
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        A = torch.full((size_in, size_out), -1000)
        A = A.fill_diagonal_(1000).to(torch.float32)
        self.A = torch.nn.Parameter(A)
        # self.A = torch.nn.init.uniform_(self.A, -1, 1)


class MaxReductionLayer(ReductionLayer):
    """
    Custom layer which performs max aggregation of weights.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = torch.sigmoid(self.A)
        return max_aggregate(W, x)


class ProdReductionLayer(ReductionLayer):
    """
    Custom layer which performs aggregation using product
    of probabilities.
    """

    def __init__(self, size_in: int, size_out: int, *, alpha: float = 1.0) -> None:
        super().__init__(size_in, size_out)
        self.alpha = alpha

    def forward(self, x) -> torch.Tensor:
        W = torch.sigmoid(self.A)
        return prod_aggregate(W, x, alpha=self.alpha)
