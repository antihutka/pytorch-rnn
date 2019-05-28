import torch
import json
import os.path
from gridgru import GRIDGRU
from simple_layers import RNNLinear

def find_data_file(jpath):
  (r, e) = os.path.splitext(jpath)
  return r + '.0'

def tensor_from_tensordef(td, storage):
  t = torch.FloatTensor()
  offset = td['offset']
  tsize = torch.Size(td['size'])
  tstride = torch.Size(td['stride'])
  t.set_(storage, td['offset'], tsize, tstride)
  return t

def layer_from_layerdef(layerdef, storage):
  ltype = layerdef['type']
  if ltype == 'LookupTable':
    w = tensor_from_tensordef(layerdef['weight'], storage)
    return torch.nn.Embedding.from_pretrained(embeddings=w), False
  elif ltype == 'GRIDGRU':
    w = tensor_from_tensordef(layerdef['weight'], storage)
    b = tensor_from_tensordef(layerdef['bias'], storage)
    return GRIDGRU(layerdef['input_dim'], layerdef['hidden_dim'], zoneout = layerdef['zoneout_p'], zoneoutd = layerdef['zoneout_pd'], weight = w, bias = b), True
  elif ltype == 'Dropout':
    return torch.nn.Dropout(p = layerdef['p'], inplace = True), False
  elif ltype == 'Linear':
    w = tensor_from_tensordef(layerdef['weight'], storage)
    b = tensor_from_tensordef(layerdef['bias'], storage)
    return RNNLinear(weight = w, bias = b), False
  else:
    raise(Exception("unknown layer %s" % ltype))

class LanguageModel(torch.nn.Module):
  def __init__(self):
    super(LanguageModel, self).__init__()
    self.idx_to_token = {}
    self.token_to_idx = {}
    self.layers = []
    self.stateful_layers = set()
    self.layer_states = {}

  def load_json(self, filename):
    with open(filename, "r") as f:
      j = json.load(f)
    for idx, token in enumerate(j['idx_to_token']):
      token_e = token.encode()
      # assume we used bytes encoding
      if (len(token) > 1):
        token_e = bytes([int(token[1:-1])])
      self.idx_to_token[idx] = token_e
      self.token_to_idx[token_e] = idx
    #print(self.idx_to_token)
    #print(self.token_to_idx)
    # let's only support one data file for checkpoint
    datafile = find_data_file(filename)
    filesize = os.path.getsize(datafile) // 4
    storage = torch.FloatStorage.from_file(datafile, False, filesize)
    #print("Loaded storage from %s with %d elements" % (datafile, storage.size()))
    for idx, layerdef in enumerate(j['layers']):
      #print("layer %d -> %s" % (idx, layerdef))
      layer, has_state = layer_from_layerdef(layerdef, storage)
      self.layers += [layer]
      self.add_module("%d-%s" % (idx, layerdef['type']), layer)
      if has_state:
        self.stateful_layers.add(layer)

  def encode_string(self, input_string):
    if isinstance(input_string, str):
      input_string = input_string.encode()
    tokens = []
    # assume all tokens are 1-byte for now
    while len(input_string) > 0:
      if input_string[0:1] in self.token_to_idx:
        tokens += [self.token_to_idx[input_string[0:1]]]
      else:
        print('warning: token not found {0}'.format(input_string[0:1]))
      input_string = input_string[1:]
    #print(tokens)
    return torch.LongTensor(tokens).unsqueeze(0)

  def forward(self, x):
    for (idx, layer) in enumerate(self.layers):
#      print("running layer %s" % layer)
      if layer in self.stateful_layers:
        x, new_state = layer.forward(x, self.layer_states.get(idx))
        self.layer_states[idx] = new_state
      else:
        x = layer.forward(x)
    return x

  def get_state(self, batch_idx = None):
    return { key:(value.cpu()) if (value.is_cuda) else (value.clone()) for (key, value) in self.layer_states }
