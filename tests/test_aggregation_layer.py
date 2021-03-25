import unittest

import ddt
import torch

from utils.aggregation_layer import aggregate


def generate_two_random_dimensions(*, number_of_pairs: int = 10):
    """
    Generates two integers representing
    input and output size of aggregation layer.
    """
    return (torch.randint(0, 1000, size=(2,)).tolist() for _ in range(number_of_pairs))


def generate_dimension_for_identity(*, number_of_pairs: int = 10):
    """
    Generates dimension of square aggregation matrix.
    """
    return torch.randint(0, 1000, size=(number_of_pairs,)).tolist()


@ddt.ddt
class AggregationTest(unittest.TestCase):
    """
    Tests which checks if aggregation layer is
    implemented correctly.
    """

    def setUp(self) -> None:
        self.batch_size: int = 128

    @ddt.idata(generate_two_random_dimensions())
    @ddt.unpack
    def test_output_size(self, size_in: int, size_out: int) -> None:
        """
        Tests if calculated output size is properly calculated.
        """
        W = torch.rand(size_in, size_out, requires_grad=False)
        x = torch.rand(self.batch_size, size_in, requires_grad=False)

        out = aggregate(W, x)
        self.assertEqual(out.size(), torch.Size([self.batch_size, size_out]))

    @ddt.idata(generate_dimension_for_identity())
    def test_aggregation_for_identity(self, dim: int) -> None:
        """
        Tests aggregation for W matrix equal to identity.
        Output should be equal input tensor.
        """
        W = torch.eye(dim, requires_grad=False)
        x = torch.rand(self.batch_size, dim, requires_grad=False)
        out = aggregate(W, x)
        self.assertTrue(torch.allclose(x, out))


if __name__ == '__main__':
    unittest.main()
