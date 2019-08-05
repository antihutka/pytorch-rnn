import argparse
import json
import numpy as np
import h5py

def jsonify_token(token):
  if token == token.decode('utf8', errors='ignore').encode('utf8'):
    return token.decode('utf8')
  else:
    return list(token)

def dejsonify_token(token):
  if isinstance(token, str):
    return token.encode()
  else:
    return bytes(token)

class Vocabulary():
  def __init__(self):
    self.idx_to_token = []
    self.token_to_idx = {}
    self.max_token_length = 0

  def get_id(self, tok, allow_add = False):
    assert tok
    if tok in self.token_to_idx:
      return self.token_to_idx[tok]
    elif allow_add:
      i = len(self.idx_to_token)
      self.idx_to_token.append(tok)
      self.token_to_idx[tok] = i
      if self.max_token_length < len(tok):
        self.max_token_length = len(tok)
      return i
    else:
      raise KeyError

  def load(self, filename):
    with open(filename, "r") as f:
      itt = json.load(f)['idx_to_token']
    itt = [dejsonify_token(t) for t in itt]
    self.idx_to_token = itt
    self.token_to_idx = {k:v for (v,k) in enumerate(itt)}
    self.max_token_length = max((len(t) for t in itt))

  def save(self, filename):
    with open(filename, "w") as f:
      json.dump({ 'idx_to_token': [jsonify_token(t) for t in self.idx_to_token] }, f) #we can't handle non-UTF8 stuff yet

def read_file(filename):
  with open(filename, "rb") as f:
    while True:
      chunk = f.read(65536)
      if chunk:
        yield chunk
      else:
        break

def get_token_at(vocabulary, chunk, start, tokens):
  for tlen in range(vocabulary.max_token_length, 0, -1):
    if chunk[start:start+tlen] in vocabulary.token_to_idx:
      tokens.append(vocabulary.get_id(chunk[start:start+tlen]))
      return start + tlen
  tokens.append(vocabulary.get_id(chunk[start:start+1], allow_add = True))
  return start + 1

def tokenize_chunk(vocabulary, chunk, lookahead = 0):
  start = 0
  tokens = []
  while True:
    if start + lookahead >= len(chunk):
      return tokens, chunk[start:]
    start = get_token_at(vocabulary, chunk, start, tokens)

def tokenize_chunks(vocab, chunks, lookahead = 256):
  extra = b""
  for chnk in reader:
    (toks, extra) = tokenize_chunk(vocab, extra + chnk, lookahead=lookahead)
    for tok in toks:
      yield tok
  (toks, extra) = tokenize_chunk(vocab, extra, lookahead=0)
  for tok in toks:
    yield tok

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', default='data/tiny-shakespeare.txt')
parser.add_argument('--input-json', default='')
parser.add_argument('--output-json', default='data/tiny-shakespeare.json')
parser.add_argument('--output-h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--freeze-vocab', action='store_true')
args = parser.parse_args()

vocab = Vocabulary()

if args.input_json:
  vocab.load(args.input_json)

reader = read_file(args.input_file)
numtok = 0
outarr = np.zeros(128*1024, dtype=np.int16)

for tok in tokenize_chunks(vocab, reader, 256):
  if numtok >= outarr.size:
    outarr.resize(outarr.size * 2)
  outarr[numtok] = tok
  numtok += 1
  if numtok % 5000000 == 0:
    print("%dM tokens processed" % (numtok/1000000,))
outarr.resize(numtok)
print("Number of tokens: %d" % len(vocab.idx_to_token))
print("Tokenized data length: %d" % numtok)

val_size = int(args.val_frac * numtok)
test_size = int(args.test_frac * numtok)
train_size = numtok - val_size - test_size

print("Train / val / test sizes: %d %d %d" % (train_size, val_size, test_size))

train = outarr[:train_size]
val = outarr[train_size:train_size+val_size]
test = outarr[train_size+val_size:]

with h5py.File(args.output_h5, 'w') as f:
  f.create_dataset('train', data=train)
  f.create_dataset('val', data=val)
  f.create_dataset('test', data=test)

vocab.save(args.output_json)
