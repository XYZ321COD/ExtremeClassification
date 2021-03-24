import unittest

import ddt
import torch

from utils.aggregation_layer import aggregate


@ddt.ddt
class AggregationTest(unittest.TestCase):
    """
    Tests which checks if aggregation layer is
    implemented correctly.
    """

    def setUp(self) -> None:
        self.batch_size: int = 128

    @ddt.file_data('test_config.yaml')
    def test_output_size(self, size_in: int, size_out: int) -> None:
        """
        Tests if calculated output size is properly calculated.
        """
        W = torch.rand(size_in, size_out, requires_grad=False)
        x = torch.rand(self.batch_size, size_in, requires_grad=False)

        out = aggregate(W, x)
        self.assertEqual(out.size(), torch.Size([self.batch_size, size_out]))

    @ddt.file_data('test_config.yaml')
    def test_aggregation_for_identity(self, size_in: int, size_out: int) -> None:
        """
        Tests aggregation for W matrix equal to identity.
        Output should be equal input tensor.
        """
        if size_in != size_out:
            self.skipTest("Test only if W is identity matrix")

        W = torch.eye(size_in, requires_grad=False)
        x = torch.rand(self.batch_size, size_out, requires_grad=False)
        out = aggregate(W, x)
        self.assertTrue(torch.allclose(x, out))


if __name__ == '__main__':
    unittest.main()
