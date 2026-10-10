import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter, init

@torch.compile
class AttentionLayer(torch.nn.Module):
  def __init__(self, input_dim, kq_size = 32, v_size = 32, kv_num = 2, q_num = 4, ctx_len = 256, weight_in = None, bias_in = None, weight_out = None, bias_out = None, sinks = None, tmix = None):
    super(AttentionLayer, self).__init__()
    assert(q_num % kv_num == 0)
    self.input_dim = input_dim
    self.kq_size = kq_size
    self.v_size = v_size
    self.kv_num = kv_num
    self.q_num = q_num
    self.ctx_len = ctx_len
    self.internal_size = q_num*kq_size + kv_num*kq_size + kv_num*v_size
    self.weight_in = Parameter(weight_in if weight_in is not None else torch.Tensor(input_dim, self.internal_size))
    self.bias_in = Parameter(bias_in if bias_in is not None else torch.Tensor(1, self.internal_size))
    self.weight_out = Parameter(weight_out if weight_out is not None else torch.Tensor(v_size*q_num + input_dim, 2*input_dim))
    self.bias_out = Parameter(bias_out if bias_out is not None else torch.Tensor(1, 2*input_dim))
    self.sinks = Parameter(sinks if sinks is not None else torch.Tensor(self.kv_num, self.kq_size+self.v_size))
    self.tmix = Parameter(tmix if tmix is not None else torch.Tensor(self.kv_num, self.kq_size + self.v_size))
    if tmix is None:
      init.zeros_(self.tmix)
    if all(t is None for t in (weight_in, bias_in, weight_out, bias_out)):
      self.reset()

  def reset(self, std_in = None, std_out = None):
    if not std_in:
      std_in = 1.0 / math.sqrt(self.input_dim + self.internal_size)
    init.normal_(self.weight_in, std = std_in)
    init.normal_(self.bias_in, std = std_in)
    if not std_out:
      std_out = 0.1 / math.sqrt(self.v_size * self.q_num + 2 * self.input_dim)
    init.normal_(self.weight_out, std = std_out)
    init.constant_(self.bias_out[:, :self.input_dim], -2)
    init.zeros_(self.bias_out[:, self.input_dim:])
    init.zeros_(self.sinks)
    return self

  def new_state(self, x):
    N = x.size(0)
    return torch.zeros(N, 0, self.kv_num, self.kq_size + self.v_size)

  def merge_states(self, x, states):
    new = self.new_state(x[0:1])[0]
    by_t = {}
    for state_idx, state in enumerate(states):
      if state is None:
        state = new
      t = state.size(0)
      if t not in by_t:
        by_t[t] = ([], [], [])
      by_t[t][0].append(x[state_idx])
      by_t[t][1].append(state)
      by_t[t][2].append(state_idx)
    return [(torch.stack(bx), torch.stack(bs), bi) for (bx, bs, bi) in by_t.values()]

  def forward(self, x, state = None):
    N = x.size(0)
    T = x.size(1)
    x_nt = x.view(N*T, -1)
    key, value, query, new_state = BuildKVQFunction.apply(x, state, self.weight_in, self.sinks, self.bias_in, self.tmix, self.q_num, self.kv_num, self.kq_size, self.v_size, self.ctx_len)

    QT = query.size(-2)
    KT = key.size(-2)
    mask = build_mask(QT, KT, self.ctx_len, query.device)

    attn_out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, enable_gqa = True)
    aout_nt = attn_out.transpose(1,2).reshape(N*T, self.q_num * self.v_size)

    gates = torch.addmm(self.bias_out.expand(N*T, -1), aout_nt, self.weight_out[:self.v_size*self.q_num])
    gates = torch.addmm(gates, x_nt, self.weight_out[self.v_size*self.q_num:])
    gates_u = gates[:, :self.input_dim].sigmoid()
    gates_h = gates[:, self.input_dim:]
    out = torch.addcmul(x_nt, gates_u, gates_h)
    
    out = out.view(N, T, -1)
    return (out, new_state)

def build_mask(QT, KT, cmax, device):
  full_k_len = QT + cmax
  missing = full_k_len - KT
  mask = torch.ones(1, 1, QT, KT, dtype=torch.bool, device=device)
  mask[0, 0,          :, -QT:].tril_(diagonal=0)
  mask[0, 0, 1+missing:,   1:].triu_(diagonal=1)
  return mask

def build_kvq(x, state, weight_in, sinks, bias_in, tmix, q_num, kv_num, kq_size, v_size, ctx_len):
  N = x.size(0)
  T = x.size(1)
  x_nt = x.view(N*T, -1)
  xp = torch.addmm(bias_in.expand(N*T, -1), x_nt, weight_in)
  Q = q_num * kq_size
  query = xp[:, :Q].view(N, T, q_num, kq_size).transpose(1,2)
  key_value = xp[:, Q:].view(N, T, kv_num, kq_size + v_size)
  sinks_exp = sinks.expand(N, 1, -1, -1)

  if state is not None:
    key_value = torch.cat([state, key_value], 1)
  if key_value.size(1) > ctx_len:
    new_state = key_value[:, 1-ctx_len:]
  else:
    new_state = key_value

  tmix_exp = tmix.expand(1, 1, -1, -1)
  key_value = key_value[:, 1:] * (1-tmix_exp) + key_value[:, :-1] * tmix_exp

  key_value = torch.cat([sinks_exp, key_value], 1)
  key_value = key_value.transpose(1,2)
  key = key_value[:, :, :, :kq_size]
  value = key_value[:, :, :, kq_size:]
  return key, value, query.clone(), new_state.clone()

class BuildKVQFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, state, weight_in, sinks, bias_in, tmix, q_num, kv_num, kq_size, v_size, ctx_len):
    ctx.sizes = (q_num, kv_num, kq_size, v_size, ctx_len)
    ctx.save_for_backward(x, state, weight_in, sinks, bias_in, tmix)
    return build_kvq(x, state, weight_in, sinks, bias_in, tmix, q_num, kv_num, kq_size, v_size, ctx_len)

  @staticmethod
  def backward(ctx, grad_key, grad_value, grad_query, grad_new_state):
    (q_num, kv_num, kq_size, v_size, ctx_len) = ctx.sizes
    (x, state, weight_in, sinks, bias_in, tmix) = ctx.saved_tensors
    N = x.size(0)
    T = x.size(1)
    D = kq_size + v_size
    Q = q_num*kq_size
    S = 0 if state is None else state.size(1)

    # grads of key/value share one (N, L, kv_num, D) buffer, position 0 = sinks
    grad_kv = torch.cat([grad_key, grad_value], -1).transpose(1, 2)
    grad_sinks = grad_kv[:, 0].sum(0)
    gsm = grad_kv[:, 1:]
    # backprop key_value = kv[:, 1:]*(1-tmix) + kv[:, :-1]*tmix, reusing grad_kv
    grad_kv = torch.cat([gsm.new_zeros(N, 1, kv_num, D), gsm*(1-tmix)], 1)
    grad_kv[:, :-1] += gsm*tmix

    grad_tmix = None
    if ctx.needs_input_grad[5]:
      x_nt = x.view(N*T, -1)
      kv = torch.addmm(bias_in[:, Q:].expand(N*T, -1), x_nt, weight_in[:, Q:]).view(N, T, kv_num, D)
      if state is not None:
        kv = torch.cat([state, kv], 1)
      gsm *= torch.sub(kv[:, :-1], kv[:, 1:])
      grad_tmix = gsm.sum((0, 1))

    if grad_new_state is not None:
      grad_kv[:, (1-ctx_len) if grad_kv.size(1) > ctx_len else 0:] += grad_new_state

    grad_state = grad_kv[:, :S] if state is not None else None
    
    grad_xp = grad_query.new_empty(N*T, weight_in.size(1))
    grad_xp[:, :Q].view(N, T, q_num, kq_size).copy_(grad_query.transpose(1,2))
    grad_xp[:, Q:].view(N, T, -1).copy_(grad_kv[:, S:].view(N, T, -1))
    
    grad_x = torch.mm(grad_xp, weight_in.t()).view(N, T, -1)
    grad_weight_in = torch.mm(x.view(N*T, -1).t(), grad_xp)
    grad_bias_in = grad_xp.sum(0, keepdim=True)
    return grad_x, grad_state, grad_weight_in, grad_sinks, grad_bias_in, grad_tmix, None, None, None, None, None