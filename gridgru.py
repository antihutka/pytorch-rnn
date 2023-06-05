import torch
import math
from torch.nn.parameter import Parameter
from extensions import sigmoid_gradient, tanh_gradient, tanh_gradient_mul, sigmoid_gradient_mul
import torch.nn.functional as F
import torch.nn.init as init

def get_weights(D, H, W):
  Wxt  = W[:D, :3*H]
  Wxd  = W[:D, 3*H:]
  Whd  = W[D:, 3*H:]
  Whtg = W[D:, :2*H]
  Whtc = W[D:, 2*H:3*H]
  return Wxt, Wxd, Whd, Whtg, Whtc

class GRIDGRU(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, zoneout = 0, zoneoutd = 0, weight = None, bias = None):
    super(GRIDGRU, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.zoneout = zoneout
    self.zoneoutd = zoneoutd
    self.swapout = False
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

  def set_swapout(self, s):
    self.swapout = s

  def reset(self, std = None):
    if not std:
      std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
    init.normal_(self.bias, std = std)
    init.normal_(self.weight, std = std)
    return self

  def new_state(self, x):
    N = x.size(0)
    H = self.hidden_dim
    return x.new_zeros(N,H)

  def forward(self, x, state = None):
    N = x.size(0)
    H = self.hidden_dim
    D = self.input_dim
    if state is None:
      prev_ht = self.new_state(x)
    else:
      assert state.dim() == 2
      assert state.size(0) == N and state.size(1) == H
      prev_ht = state
    use_swapout = self.swapout and torch.is_grad_enabled()
    return GRIDGRUFunction.apply(x, prev_ht, self.weight, self.bias, H, D, self.zoneout, self.zoneoutd, self.training, use_swapout)

def swapout_tensor(t):
  return torch.empty_like(t, device='cpu', pin_memory=True).copy_(t, non_blocking=True)

class GRIDGRUFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, prev_ht, weight, bias, H, D, zoneout, zoneoutd, training, swapout):
    ctx.first_ht = prev_ht
    N = x.size(0)
    T = x.size(1)
    Wxt, Wxd, Whd, Whtg, Whtc = get_weights(D, H, weight)
    assert x.size(2) == D
    x_nt = x.view(N * T, -1)
    bias_nt = bias.expand(N * T, -1)
    gates_nt = torch.addmm(bias_nt[:,:3*H], x_nt, Wxt)
    gates = gates_nt.view(N, T, -1)
    gatesd_nt = bias_nt[:, 3*H:].clone()
    gatesd_nt[:, :2*D].addmm_(x_nt, Wxd[:, :2*D])
    #gatesd = gatesd_nt.view(N, T, -1)
    ht = x.new_zeros(N, T, H)
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
      if zoneout > 0:
        if training:
          F.dropout(u, p=zoneout, training=True, inplace=True)
        u.mul_(1-zoneout)
      next_ht.copy_(prev_ht).addcmul_(u, prev_ht, value=-1).addcmul_(u, hc)
      prev_ht = next_ht
    gatesd_nt.addmm_(ht.view(N*T, -1), Whd)
    gatesd_nt[:, :2*D].sigmoid_()
    ud_b = gatesd_nt[:, :D]
    rd_b = gatesd_nt[:, D:2*D]
    hcd_b = gatesd_nt[:, 2*D:3*D]
    if zoneoutd > 0:
      if training:
        F.dropout(ud_b, p=zoneoutd, training=training, inplace=True)
      ud_b.mul_(1-zoneoutd)
    bfr2 = torch.mul(rd_b.view(N, T, -1), x).view(N*T, -1)
    hcd_b.addmm_(bfr2, Wxd[:, 2*D:3*D])

    hcd_b.tanh_()
    h=torch.addcmul(x, ud_b.view(N, T, -1), x, value=-1)
    h.addcmul_(ud_b.view(N, T, -1), hcd_b.view(N, T, -1))
    ctx.H = H
    ctx.swapout = swapout
    if swapout:
      gates = swapout_tensor(gates)
      gatesd_nt = swapout_tensor(gatesd_nt)
    ctx.save_for_backward(weight, bias, ht, gatesd_nt, x, gates)
    return (h, next_ht.clone())

  @staticmethod
  def backward(ctx, grad_output, grad_lastht):
    (weight, bias, ht, gatesd_nt, x, gates) = ctx.saved_tensors
    if ctx.swapout:
      gates = gates.to(weight.device, non_blocking=True)
      gatesd_nt = gatesd_nt.to(weight.device, non_blocking=True)
    TB = 16
    N = grad_output.size(0)
    T = grad_output.size(1)
    D = grad_output.size(2)
    H = ctx.H

    Wxt, Wxd, Whd, Whtg, Whtc = get_weights(D, H, weight)
    gatesd = gatesd_nt.view(N, T, -1)

    grad_x = grad_first_ht = grad_weight = grad_bias = None
    grad_x = grad_output.new_zeros(T, N, D)
    grad_weight = weight.new_zeros(weight.size())
    grad_h0_tb = ht.new_zeros(TB, N, H)
    grad_a_tb = ht.new_zeros(TB, N, 3*H)
    grad_ad_tb = ht.new_zeros(TB, N, 3*D)
    temp_bufferd_tb = ht.new(TB, N, D)
    #grad_a_sumd = ht.new(1, 3*D)
    grad_bias = bias.new_zeros(bias.size())
    grad_bt = grad_bias[:3*H]
    grad_bd = grad_bias[3*H:]
    grad_next_h = ht.new_zeros(ctx.first_ht.size())
    temp_buffer_tb = ht.new_zeros(TB, N, H)
    temp_buffer = temp_buffer_tb[0]
    grad_next_hd_tb = ht.new_zeros(TB, N, D)

    grad_Wxt, grad_Wxd, grad_Whd, grad_Whtg, grad_Whtc = get_weights(D, H, grad_weight)

    for t in range(T-1, -1, -1):
      if t == 0:
        prev_h = ctx.first_ht
      else:
        prev_h = ht[:, t-1]
      TBi = t % TB
      grad_h0 = grad_h0_tb[TBi]
      grad_a = grad_a_tb[TBi]
      grad_au = grad_a[:, :H]
      grad_ar = grad_a[:, H:2*H]
      grad_ahc = grad_a[:, 2*H:3*H]

      if TBi == (TB - 1) or t == (T - 1):
        TBl = TBi + 1
        tfirst = t - TBi
        
        grad_ad_t = grad_ad_tb[:TBl].transpose(0,1)
        grad_aud_t = grad_ad_t[:, :, :D]
        grad_ard_t = grad_ad_t[:, :, D:2*D]
        grad_ahcd_t = grad_ad_t[:, :, 2*D:3*D]
        
        grad_ad_tn = grad_ad_tb[:TBl].view(TBl*N, 3*D)
        grad_aud_tn = grad_ad_tn[:, :D]
        grad_ahcd_tn = grad_ad_tn[:, 2*D:3*D]
        grad_h0_tn = grad_h0_tb[:TBl].view(TBl*N, H)
        
        temp_bufferd_t = temp_bufferd_tb[:TBl].transpose(0,1)
        temp_bufferd_tn = temp_bufferd_tb[:TBl].view(TBl * N, D)
        
        ud_t = gatesd[:, tfirst:t+1, :D]
        rd_t = gatesd[:, tfirst:t+1, D:2*D]
        hcd_t = gatesd[:, tfirst:t+1, 2*D:3*D]
        x_t = x[:, tfirst:t+1]
        grad_h_t = grad_output[:, tfirst:t+1]
        
        tanh_gradient_mul(grad_ahcd_t, hcd_t, grad_h_t, ud_t)
        torch.mm(grad_ahcd_tn, Wxd[:, 2*D:3*D].t(), out=grad_aud_tn)
        sigmoid_gradient_mul(grad_ard_t, rd_t, grad_aud_t, x_t)
        torch.add(hcd_t, x_t, out=temp_bufferd_t, alpha=-1)
        sigmoid_gradient_mul(grad_aud_t, ud_t, grad_h_t, temp_bufferd_t)
        torch.mm(grad_ad_tn, Whd.t(), out=grad_h0_tn)
        grad_Whd.addbmm_(ht[:, tfirst:t+1].transpose(0,1).transpose(1,2), grad_ad_tb[:TBl])
        grad_Wxd[:, :2*D].addbmm_(x_t.transpose(0,1).transpose(1,2), grad_ad_tb[:TBl, :, :2*D])
        grad_a_sumd = grad_ad_tn.sum(0)
        grad_bd.add_(grad_a_sumd)
        torch.mul(x_t, rd_t, out=temp_bufferd_t)
        grad_Wxd[:, 2*D:3*D].addbmm_(temp_bufferd_t.transpose(0,1).transpose(1,2), grad_ad_tb[:TBl, :, 2*D:3*D])
        torch.mm(grad_ahcd_tn, Wxd[:, 2*D:3*D].t(), out=temp_bufferd_tn)
        temp_bufferd_t.mul_(rd_t)
      
      u = gates[:, t, :H]
      r = gates[:, t, H:2*H]
      hc = gates[:, t, 2*H:3*H]
      
      grad_next_h.add_(grad_h0)
      tanh_gradient_mul(grad_ahc, hc, grad_next_h, u)
      torch.mm(grad_ahc, Whtc.t(), out=grad_au)
      grad_r = grad_au
      sigmoid_gradient_mul(grad_ar, r, grad_r, prev_h)
      
      torch.add(hc, prev_h, out=temp_buffer, alpha=-1)
      sigmoid_gradient_mul(grad_au, u, grad_next_h, temp_buffer)
      
      grad_next_h.addcmul_(u, grad_next_h, value=-1)
      grad_next_h.addmm_(grad_a[:, :2*H], Whtg.t())
      torch.mm(grad_a[:, 2*H:3*H], Whtc.t(), out=temp_buffer)
      temp_buffer.mul_(r)
      grad_next_h.add_(temp_buffer)
      
      if TBi == 0:
        tlast = t + TB
        if tlast > T:
          tlast = T
        TBl = tlast - t
        
        grad_h_tb = grad_output[:, t:tlast]
        grad_next_hd_t = grad_next_hd_tb[:TBl]
        grad_a_t = grad_a_tb[:TBl]
        grad_a_tn = grad_a_t.view(TBl*N, 3*H)
        temp_buffer_t = temp_buffer_tb[:TBl]
        r_t = gates[:, t:tlast, H:2*H]
        
        torch.addcmul(grad_h_tb, gatesd[:, t:tlast, :D], grad_h_tb, out=grad_next_hd_t.transpose(0,1), value=-1)
        grad_next_hd_t.view(TBl*N, D).addmm_(grad_ad_tb[:TBl].view(TBl*N, 3*D)[:, :2*D], Wxd[:, :2*D].t())
        grad_next_hd_t.add_(temp_bufferd_tb[:TBl])
        o = grad_x[t:tlast].view(TBl*N, D)
        torch.addmm(grad_next_hd_t.view(TBl*N, D), grad_a_tn, Wxt.t(), out=o) #nan started
        grad_Wxt.addbmm_(x[:, t:tlast].transpose(0,1).transpose(1,2), grad_a_t)
        grad_a_sum = torch.sum(grad_a_tn, 0)
        grad_bt.add_(grad_a_sum)
        if t > 0:
          grad_Whtg.addbmm_(ht[:, t-1:tlast-1].transpose(0,1).transpose(1,2), grad_a_t[:, :, :2*H])
          torch.mul(ht[:, t-1:tlast-1], r_t, out=temp_buffer_t.transpose(0,1))
        else:
          grad_Whtg.addbmm_(ht[:, t:tlast-1].transpose(0,1).transpose(1,2), grad_a_t[1:TBl, :, :2*H])
          grad_Whtg.addmm_(ctx.first_ht.t(), grad_a[:, :2*H])
          torch.mul(ht[:, t:tlast-1], r_t[:, 1:TBl], out=temp_buffer_t[1:TBl].transpose(0,1))
          torch.mul(ctx.first_ht, r_t[:, 0], out=temp_buffer_t[0])
        grad_Whtc.addbmm_(temp_buffer_t.transpose(1,2), grad_a_tb[:TBl, :, 2*H:3*H])
    grad_first_ht = grad_next_h.clone()
    return (grad_x.transpose(0,1), grad_first_ht, grad_weight, grad_bias, None, None, None, None, None, None)







