import torch


def init_identity(size_in: int, size_out: int) -> torch.Tensor:
    """
    Initialize matrix `A` in order to receive
    identity after sigmoid is applied.
        1000 -> sigmoid -> 1
        -1000 -> sigmoid -> 0
    """
    A: torch.Tensor = torch.full((size_in, size_out), -1000)
    A = A.fill_diagonal_(1000).to(torch.float32)
    return A


def init_identity_permutation(size_in: int, size_out: int) -> torch.Tensor:
    """
    Initialize matrix `A` in order to receive permuted
    identity after sigmoid is applied.
    """
    A: torch.Tensor = init_identity(size_in, size_out)
    return A[torch.randperm(A.size(0)), :]


def init_with_ones(size_in: int, size_out: int, *, p: int = 8) -> torch.Tensor:
    """
    Initialize matrix `A` in order to get a matrix
    composed of 0 and 1 (on random positions with prob == 1 / p) after sigmoid is applied.
    """
    A: torch.Tensor = torch.full((size_in, size_out), -1000).to(torch.float32)
    A += torch.bernoulli(torch.ones(A.size()) / p) * 2000
    return A


def init_xavier(size_in: int, size_out: int) -> torch.Tensor:
    A: torch.Tensor = torch.zeros((size_in, size_out))
    torch.nn.init.xavier_uniform_(A)
    return A
