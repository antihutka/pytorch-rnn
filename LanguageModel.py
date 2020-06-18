import torch
import json
import os.path
from gridgru import GRIDGRU
from lstm import PTLSTM
from simple_layers import RNNLinear

def find_data_file(jpath):
  (r, e) = os.path.splitext(jpath)
  return r + '.0'

def tensor_from_tensordef(td, storage, clone_tensors):
  t = torch.FloatTensor()
  offset = td['offset']
  tsize = torch.Size(td['size'])
  tstride = torch.Size(td['stride'])
  t.set_(storage, td['offset'], tsize, tstride)
  if clone_tensors:
    return t.clone().detach()
  else:
    return t

def layer_from_layerdef(layerdef, storage, clone_tensors):
  ltype = layerdef['type']
  if ltype == 'LookupTable' or ltype == 'Embedding':
    w = tensor_from_tensordef(layerdef['weight'], storage, clone_tensors)
    return torch.nn.Embedding.from_pretrained(embeddings=w, freeze=False), False
  elif ltype == 'GRIDGRU':
    w = tensor_from_tensordef(layerdef['weight'], storage, clone_tensors)
    b = tensor_from_tensordef(layerdef['bias'], storage, clone_tensors)
    return GRIDGRU(layerdef['input_dim'], layerdef['hidden_dim'], zoneout = layerdef['zoneout_p'], zoneoutd = layerdef['zoneout_pd'], weight = w, bias = b), True
  elif ltype == 'PTLSTM':
    w_ih = tensor_from_tensordef(layerdef['w_ih'], storage, clone_tensors)
    w_hh = tensor_from_tensordef(layerdef['w_hh'], storage, clone_tensors)
    b_ih = tensor_from_tensordef(layerdef['b_ih'], storage, clone_tensors)
    b_hh = tensor_from_tensordef(layerdef['b_hh'], storage, clone_tensors)
    return PTLSTM(layerdef['input_dim'], layerdef['hidden_dim'], w_ih=w_ih, w_hh=w_hh, b_ih=b_ih, b_hh=b_hh), True
  elif ltype == 'Dropout':
    return torch.nn.Dropout(p = layerdef['p'], inplace = True), False
  elif ltype == 'Linear':
    w = tensor_from_tensordef(layerdef['weight'], storage, clone_tensors)
    b = tensor_from_tensordef(layerdef['bias'], storage, clone_tensors)
    return RNNLinear(weight = w, bias = b), False
  else:
    raise(Exception("unknown layer %s" % ltype))

def save_layer(layer, params):
  ltype = layer.__class__.__name__
  if ltype == 'Embedding':
    ld = {'weight' : params[layer.weight]}
  elif ltype == 'GRIDGRU':
    ld = {'weight' : params[layer.weight], 'bias' : params[layer.bias],
          'hidden_dim' : layer.hidden_dim, 'input_dim' : layer.input_dim,
          'zoneout_p' : layer.zoneout, 'zoneout_pd' : layer.zoneoutd }
  elif ltype == 'PTLSTM':
    ld = {'w_ih' : params[layer.w_ih], 'w_hh' : params[layer.w_hh],
          'b_ih' : params[layer.b_ih], 'b_hh' : params[layer.b_hh],
          'hidden_dim' : layer.hidden_dim, 'input_dim' : layer.input_dim}
  elif ltype == 'Dropout':
    ld = {'p' : layer.p}
  elif ltype == 'Linear' or ltype == 'RNNLinear':
    ltype = 'Linear'
    ld = {'weight' : params[layer.weight], 'bias' : params[layer.bias]}
  else:
    raise Exception('Unknown layer type ' + ltype)
  ld['type'] = ltype
  return ld

class LanguageModel(torch.nn.Module):
  def __init__(self):
    super(LanguageModel, self).__init__()
    self.idx_to_token = {}
    self.token_to_idx = {}
    self.layers = []
    self.stateful_layers = set()
    self.layer_states = {}

  def parse_tokendata(self, j):
    self.longest_token = 0
    for idx, token in enumerate(j['idx_to_token']):
      if isinstance(token, list):
        token_e = bytes(token)
      # transform [n] tokens to byte value
      elif (isinstance(token, str) and
            len(token) > 2
            and token.startswith("[")
            and token.endswith("]")):
        token_e = bytes([int(token[1:-1])])
      else:
        token_e = token.encode()
      self.idx_to_token[idx] = token_e
      self.token_to_idx[token_e] = idx
      if self.longest_token < len(token_e):
        self.longest_token = len(token_e)
    print('longest token %d' % self.longest_token)

  def load_tokendata(self, filename):
    with open(filename, "r") as f:
      j = json.load(f)
    if isinstance(j['idx_to_token'], dict):
      itt = j['idx_to_token']
      itt = {int(k)-1:v for (k,v) in itt.items()}
      itt = [itt[n] for n in range(0, len(itt))]
      j['idx_to_token'] = itt
    self.parse_tokendata(j)

  def load_json(self, filename, clone_tensors=False):
    with open(filename, "r") as f:
      j = json.load(f)
    self.parse_tokendata(j)
    # let's only support one data file for checkpoint
    datafile = find_data_file(filename)
    filesize = os.path.getsize(datafile) // 4
    storage = torch.FloatStorage.from_file(datafile, False, filesize)
    #print("Loaded storage from %s with %d elements" % (datafile, storage.size()))
    for idx, layerdef in enumerate(j['layers']):
      #print("layer %d -> %s" % (idx, layerdef))
      layer, has_state = layer_from_layerdef(layerdef, storage, clone_tensors)
      self.layers += [layer]
      self.add_module("%d-%s" % (idx, layerdef['type']), layer)
      if has_state:
        self.stateful_layers.add(layer)

  def save_model(self, filename):
    params = {}
    storages = {}
    filesize = 0
    for param in self.parameters():
      stor = param.storage()
      if stor not in storages:
        storages[stor] = filesize
        filesize += stor.size()
      params[param] = { "stride" : param.stride(), "size" : param.size(), "storage" : 0, "offset" : storages[stor] + param.storage_offset() }
    layers = [save_layer(l, params) for l in self.layers]
    with open(filename + '.json', 'w') as f:
      json.dump({'layers':layers, 'idx_to_token' : self.jsonify_tokens()}, f)
    filestorage = torch.FloatStorage.from_file(filename + '.0', shared=True, size=filesize)
    for (stor,off) in storages.items():
      filestorage[off : off+stor.size()].copy_(stor)

  def jsonify_tokens(self):
    def t(token):
      if token == token.decode('utf8', errors='ignore').encode('utf8'):
        return token.decode('utf8')
      else:
        return list(token)
    return [t(self.idx_to_token[i]) for i in range(0, len(self.idx_to_token))]

  def build_model(self, layertype = 'GRIDGRU', dropout = 0, num_layers = 2, **kwargs):
    current_size = len(self.idx_to_token)
    self.layers.append(torch.nn.Embedding(current_size, kwargs['D']))
    current_size = kwargs['D']
    for i in range(0, num_layers):
      if layertype == 'GRIDGRU':
        lay = GRIDGRU(current_size, kwargs['H'], kwargs['zoneout'])
        self.layers.append(lay)
        self.stateful_layers.add(lay)
      elif layertype == 'LSTM':
        lay = PTLSTM(current_size, kwargs['H'])
        self.layers.append(lay)
        self.stateful_layers.add(lay)
        current_size = kwargs['H']
      else:
        raise Exception("Unknown layer type")
      if dropout > 0:
        self.layers.append(torch.nn.Dropout(p=dropout, inplace=True))
    self.layers.append(torch.nn.Linear(current_size, len(self.idx_to_token)))
    for li, lay in enumerate(self.layers):
      ln = "%d-%s" % (li, type(lay).__name__)
      self.add_module(ln, lay)
      print(ln)

  def longest_prefix(self, input_string):
    for toklen in range(self.longest_token, 0, -1):
      t = input_string[0:toklen]
      if t in self.token_to_idx:
        return t
    return None

  def encode_string(self, input_string):
    if isinstance(input_string, str):
      input_string = input_string.encode()
    tokens = []
    while len(input_string) > 0:
      tok = self.longest_prefix(input_string)
      if tok is None:
        print('warning: token not found {0}'.format(input_string[0:1]))
        input_string = input_string[1:]
      else:
        tokens.append(self.token_to_idx[tok])
        input_string = input_string[len(tok):]
    return torch.LongTensor(tokens).unsqueeze(0)

  def decode_string(self, input_sequence):
    return b''.join([self.idx_to_token[int(x)] for x in input_sequence])

  def forward(self, x):
    for (idx, layer) in enumerate(self.layers):
#      print("running layer %s" % layer)
      if layer in self.stateful_layers:
        x, new_state = layer.forward(x, self.layer_states.get(idx))
        self.layer_states[idx] = new_state.detach()
      else:
        x = layer.forward(x)
    return x

  def forward_with_states(self, x, h0_split):
    batchsize = x.size(0)
    hn = [{} for i in range(batchsize)]
    for (layeridx, layer) in enumerate(self.layers):
      if layer in self.stateful_layers:
        h0 = layer.new_state(x)
        for batchidx in h0_split:
          if h0_split[batchidx] is not None:
            h0[batchidx].copy_(h0_split[batchidx][layeridx])
        x, new_state = layer.forward(x, h0)
        for batchidx in range(batchsize):
          hn[batchidx][layeridx] = new_state[batchidx]
      else:
        x = layer.forward(x)
    return (x, hn)

  def get_state(self, batch_idx = None):
    return { layerid:(tensor.cpu()) if (tensor.is_cuda) else (tensor.clone()) for (layerid, tensor) in self.layer_states }

  def clear_states(self):
    self.layer_states.clear()
