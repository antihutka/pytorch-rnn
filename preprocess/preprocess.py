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

  def __len__(self):
    return len(self.idx_to_token)

  def __getitem__(self, k):
    return self.idx_to_token[k]

  def remove(self, tok):
    itt = self.idx_to_token
    itt.remove(tok)
    self.token_to_idx = {k:v for (v,k) in enumerate(itt)}
    self.max_token_length = max((len(t) for t in itt))

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
parser.add_argument('--max-tokens', type=int, default=0)
parser.add_argument('--min-merge-count', type=int, default=1000000)
args = parser.parse_args()

vocab = Vocabulary()

if args.input_json:
  vocab.load(args.input_json)

def tts(*args):
  return b''.join(args).decode(errors='backslashreplace')

def find_mergeable(vocab, reader, lookahead, stop_on_count = None):
  last = -1
  counts = {}
  toks = 0
  tbytes = 0
  for tok in tokenize_chunks(vocab, reader, 256):
    toks += 1
    tbytes += len(vocab[tok])
    if last >= 0:
      pair = (last, tok)
      if pair in counts:
        counts[pair] += 1
      else:
        counts[pair] = 1
    if toks % 100000 == 0:
      toppairs = list(sorted(counts, key=counts.get, reverse=True))
      print("%3dM tokens %3dM bytes  %.3f bpt | top: %s" % (toks/1000000, tbytes/1000000, (tbytes/toks),
            ", ".join(["%6s:%8d" % (repr(tts(vocab[x], vocab[y])), counts[x,y]) for (x,y) in toppairs[:10]])),
            end = '    \r')
      if stop_on_count and stop_on_count < counts[toppairs[0]]:
        break
    last = tok
  toppairs = list(sorted(counts, key=counts.get, reverse=True))
  print("")
  return counts, toppairs

merged_tokens = set()
removed_tokens = set()
if args.max_tokens > 0:
  while len(vocab) < args.max_tokens:
    reader = read_file(args.input_file)
    counts, toppairs = find_mergeable(vocab, reader, 256, args.min_merge_count)
    pair = toppairs[0]
    if counts[pair] > args.min_merge_count:
      t1 = vocab[pair[0]]
      t2 = vocab[pair[1]]
      print("Merging tokens %d/%s + %d/%s => %s" % (pair[0], repr(tts(t1)), pair[1], repr(tts(t2)), repr(tts(t1,t2))))
      merged_tokens.add(t1+t2)
      removed_tokens.discard(t1+t2)
      vocab.get_id(t1 + t2, allow_add = True)
      if t1 in merged_tokens:
        print("Deleting token %d/%s" % (pair[0], repr(tts(t1))))
        vocab.remove(t1)
        merged_tokens.remove(t1)
        removed_tokens.add(t1)
      if t2 in merged_tokens:
        print("Deleting token %d/%s" % (pair[1], repr(tts(t2))))
        vocab.remove(t2)
        merged_tokens.remove(t2)
        removed_tokens.add(t2)
      print("New extra vocabulary is: %s, total size is %d" % ([tts(x) for x in sorted(merged_tokens, key=len)][-15:], len(vocab)))
      print("Removed vocabulary: %s" % ([tts(x) for x in sorted(removed_tokens, key=len)][-15:]))
      #print("Full vocabulary is: %s" % [tts(x) for x in vocab.idx_to_token])
    else:
      break

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
