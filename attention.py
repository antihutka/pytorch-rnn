import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter, init

@torch.compile
class AttentionLayer(torch.nn.Module):
  def __init__(self, input_dim, kq_size = 32, v_size = 32, kv_num = 2, q_num = 4, ctx_len = 256, weight_in = None, bias_in = None, weight_out = None, bias_out = None, sinks = None):
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
    xp = torch.addmm(self.bias_in.expand(N*T, -1), x_nt, self.weight_in)
    query = xp[:, :self.q_num * self.kq_size].view(N, T, self.q_num, self.kq_size).transpose(1,2)

    key_value = xp[:, self.q_num*self.kq_size:].view(N, T, self.kv_num, self.kq_size + self.v_size)
    sinks_exp = self.sinks.expand(N, 1, -1, -1)
    if state is not None:
      key_value = torch.cat([sinks_exp, state, key_value], 1)
    else:
      key_value = torch.cat([sinks_exp, key_value], 1)
    new_state = key_value[:, -(self.ctx_len-1):] # TODO ensure that sinks are stripped
    key_value = key_value.transpose(1,2)
    key = key_value[:, :, :, :self.kq_size]
    value = key_value[:, :, :, self.kq_size:]

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
