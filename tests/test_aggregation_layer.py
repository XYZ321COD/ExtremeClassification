import unittest
import torch

from utils.aggregation_layer import aggregate, Reduction_Layer


class AggregationTest(unittest.TestCase):
    """
    Tests which checks if aggregation layer is
    implemented correctly.
    """

    def setUp(self) -> None:
        self.size_in: int = 5
        self.size_out: int = 10
        self.batch_size: int = 128

        self.reduction_layer = Reduction_Layer(self.size_in, self.size_out)
        self.x = torch.rand(self.batch_size, self.size_in, requires_grad=False)
        self.random_W = torch.rand(self.size_in, self.size_out, requires_grad=False)

    def test_output_size(self) -> None:
        """
        Tests if calculated output size is properly calculated.
        """
        out = aggregate(self.random_W, self.x)
        self.assertEqual(out.size(), torch.Size([self.batch_size, self.size_out]))

    def test_aggregation_for_identity(self) -> None:
        """
        Tests aggregation for W matrix equal to identity.
        Output should be equal input tensor.
        """
        out = aggregate(torch.eye(self.size_in), self.x)
        self.assertTrue(torch.allclose(self.x, out))


if __name__ == '__main__':
    unittest.main()
