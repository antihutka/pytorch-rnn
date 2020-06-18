import torch
import math
import torch.nn.init as init
from torch import _VF
from torch.nn.parameter import Parameter

class PTLSTM(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, w_ih = None, w_hh = None, b_ih = None, b_hh = None):
    super(PTLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    if w_ih is None and w_hh is None and b_ih is None and b_hh is None:
      self.w_ih = Parameter(torch.Tensor(4*hidden_dim, input_dim))
      self.w_hh = Parameter(torch.Tensor(4*hidden_dim, hidden_dim))
      self.b_ih = Parameter(torch.Tensor(4*hidden_dim))
      self.b_hh = Parameter(torch.Tensor(4*hidden_dim))
      self.reset()
    else:
      assert w_ih.dim() == 2 and w_hh.dim() == 2 and b_ih.dim() == 1 and b_hh.dim() == 1
      assert w_ih.size(0) == 4*hidden_dim and w_ih.size(1) == input_dim
      assert w_hh.size(0) == 4*hidden_dim and w_hh.size(1) == hidden_dim
      assert b_ih.size(0) == 4*hidden_dim
      assert b_hh.size(0) == 4*hidden_dim
      self.w_ih = Parameter(w_ih)
      self.w_hh = Parameter(w_hh)
      self.b_ih = Parameter(b_ih)
      self.b_hh = Parameter(b_hh)

  def reset(self, std = None):
    if not std:
      std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
    init.normal_(self.b_ih, std = std)
    init.normal_(self.b_hh, std = std)
    init.normal_(self.w_ih, std = std)
    init.normal_(self.w_hh, std = std)

  def new_state(self, x):
    N = x.size(0)
    H = self.hidden_dim
    z = x.new_zeros(N, 2*H)
    return z

  def forward(self, input, state = None):
    H = self.hidden_dim
    if state is None:
      state = self.new_state(input)
    state = (state[:, :H].unsqueeze(0), state[:, H:].unsqueeze(1))
    res = _VF.lstm(input, state, (self.w_ih, self.w_hh, self.b_ih, self.b_hh), True, 1, 0, self.training, False, True)
    output, hidden1, hidden2 = res
    outstate = torch.cat([hidden1[0], hidden2[0]], 1)
    return output, outstate
