import torch
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GRIDGRU(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, zoneout = 0, zoneoutd = 0, weight = None, bias = None):
    super(GRIDGRU, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.zoneout = zoneout
    self.zoneoutd = zoneoutd
    if weight is not None and bias is not None:
      assert weight.dim() == 2
      assert weight.size(0) == input_dim + hidden_dim
      assert weight.size(1) == 3 * (input_dim + hidden_dim)
      self.weight = Parameter(weight)
      assert bias.dim() == 1
      assert bias.size(0) == (3 * (input_dim + hidden_dim))
      self.bias = Parameter(bias)
    else:
      assert weight is None and bias is None
      self.weight = Parameter(torch.Tensor(input_dim + hidden_dim, 3 * (input_dim + hidden_dim)))
      self.bias = Parameter(torch.Tensor(3 * (input_dim + hidden_dim)))
      self.reset()

  def reset(self, std = None):
    if not std:
      std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
    self.bias:normal_(std = std)
    self.weight:normal_(std = std)
    return self

  def get_weights(self):
    D = self.input_dim
    H = self.hidden_dim
    W = self.weight
    Wxt  = W[:D, :3*H]
    Wxd  = W[:D, 3*H:]
    Whd  = W[D:, 3*H:]
    Whtg = W[D:, :2*H]
    Whtc = W[D:, 2*H:3*H]
    return Wxt, Wxd, Whd, Whtg, Whtc

  def forward(self, x, state = None):
    Wxt, Wxd, Whd, Whtg, Whtc = self.get_weights()
    N = x.size(0)
    T = x.size(1)
    H = self.hidden_dim
    D = self.input_dim
    assert x.size(2) == D
    prev_ht = None
    if state is None:
      prev_ht = x.new(N, H).zero_()
    else:
      assert state.dim() == 2
      assert state.size(0) == N and state.size(1) == H
      prev_ht = state
    x_nt = x.view(N * T, -1)
    bias_nt = self.bias.expand(N * T, -1)
    gates_nt = torch.addmm(bias_nt[:,:3*H], x_nt, Wxt)
    gates = gates_nt.view(N, T, -1)
    gatesd_nt = x.new(N * T, 3 * D).copy_(bias_nt[:, 3*H:])
    gatesd_nt[:, :2*D].addmm_(x_nt, Wxd[:, :2*D])
    #gatesd = gatesd_nt.view(N, T, -1)
    ht = x.new(N, T, H).zero_()
    for t in range(0, T):
      next_ht = ht[:, t]
      cur_gates = gates[:, t]
      cur_gates_g = cur_gates[:, :2 * H]
      cur_gates_g.addmm_(prev_ht, Whtg).sigmoid_()
      u = cur_gates[:, :H]
      r = cur_gates[:, H:2*H]
      bfr1 = torch.mul(r, prev_ht)
      hc = cur_gates[:, 2*H:3*H]
      hc.addmm_(bfr1, Whtc)
      hc.tanh_()
      # make sure we're handling direct/inveretd dropout correctly
      if self.zoneout > 0:
        if self.training:
          F.dropout(u, p=self.zoneout, training=True, inplace=True)
        u.mul_(1-self.zoneout)
      next_ht.copy_(prev_ht).addcmul_(-1, u, prev_ht).addcmul_(u, hc)
      prev_ht = next_ht
    gatesd_nt.addmm_(ht.view(N*T, -1), Whd)
    gatesd_nt[:, :2*D].sigmoid_()
    ud_b = gatesd_nt[:, :D]
    rd_b = gatesd_nt[:, D:2*D]
    hcd_b = gatesd_nt[:, 2*D:3*D]
    if self.zoneoutd > 0:
      if self.training:
        F.dropout(ud_b, p=self.zoneoutd, training=self.training, inplace=True)
      ud_b.mul_(1-self.zoneoutd)

    bfr2 = torch.mul(rd_b, x).view(N*T, -1)
    hcd_b.addmm_(bfr2, Wxd[:, 2*D:3*D])

    hcd_b.tanh_()
    h=torch.addcmul(x, -1, ud_b, x)
    h.addcmul_(ud_b, hcd_b)
    return (h, next_ht)



