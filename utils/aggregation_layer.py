import numpy as np
import torch


def aggregate(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Performs aggregation for given weights and input tensors.
    """
    seq = []
    for i in range(len(x)):
        sub_seq = []
        for j in range(W.size()[0]):
            sub_seq.append(W[j, :] * x[i, j])

        z = torch.stack(sub_seq, dim=1)
        z = z.max(dim=1)[0]
        seq.append(z)

    out = torch.stack(seq, 0)
    return out


class Reduction_Layer(torch.nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        A = torch.full((size_in, size_out), -1000)
        A = A.fill_diagonal_(1000).to(torch.float32)
        self.A = torch.nn.Parameter(A)  # nn.Parameter is a Tensor that's a module parameter.

    def forward(self, x):
        W = torch.nn.Sigmoid()(self.A)  # Applying sigmoid to weight
        return aggregate(W, x)
        # return (W * x.unsqueeze(1)).max(dim=-1)[0]