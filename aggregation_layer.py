import numpy as np
import torch


class Reduction_Layer(torch.nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        A = torch.Tensor(size_in, size_out)
        self.A = torch.nn.Parameter(A)  # nn.Parameter is a Tensor that's a module parameter.
        # torch.nn.init.eye_(self.A)
        y = 1.0/np.sqrt(size_in)
        torch.nn.init.uniform_(self.A, -y, y)

    def forward(self, x):
        W = torch.nn.Sigmoid()(self.A)  # Applying sigmoid to weight
        # tensors = []
        # for elem in W.t():
        #  max = torch.max(x * elem, 1)[0]
        #  tensors.append(max)
        # output = torch.stack(tensors, 1)
        seq = []
        for i in range(len(x)):
            sub_seq = []
            for j in range(len(W)):
                sub_seq.append(W[j, :] * x[i, j])

            seq.append(torch.stack(sub_seq, dim=0).max(dim=1)[0])

        out = torch.stack(seq, 0)
        # out = out.max(dim=1)[0]

        assert x.size() == out.size()
        return out
        # return (W * x.unsqueeze(1)).max(dim=-1)[0]
