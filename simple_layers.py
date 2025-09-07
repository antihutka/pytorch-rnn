import torch

from torch.nn import functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter


class RNNLinear(Module):
  def __init__(self, input_dim = None, output_dim = None, weight = None, bias = None):
    super(RNNLinear, self).__init__()
    assert weight is not None
    if weight is not None:
      assert input_dim is None and output_dim is None
      assert weight.dim() == 2
      self.input_dim = weight.size(1)
      self.output_dim = weight.size(0)
      self.weight = Parameter(weight)
      assert bias is not None
      assert bias.dim() == 1
      assert bias.size(0) == self.output_dim
      self.bias = Parameter(bias)

  def forward(self, x):
    return F.linear(x, self.weight, self.bias)
