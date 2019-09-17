from LanguageModel import LanguageModel
from trainutils import Timer
import argparse
import time
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--layer-type', default='GRIDGRU')
parser.add_argument('--num-layers', default=2, type=int)
parser.add_argument('--embedding-dim', default=128, type=int)
parser.add_argument('--hidden-dim', default=128, type=int)
parser.add_argument('--zoneout', default=0, type=float)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--vocab-size', default=200, type=int)

parser.add_argument('--min-batch', default=1, type=int)
parser.add_argument('--max-batch', default=32, type=int)
parser.add_argument('--min-iter', default=10)
args = parser.parse_args()

model = LanguageModel()
for i in range(0, args.vocab_size):
  ib = bytes([i])
  model.idx_to_token[i] = ib
  model.token_to_idx[ib] = i
model.longest_token = 1
model.build_model(
  layertype = args.layer_type,
  dropout = args.dropout,
  num_layers = args.num_layers,
  D = args.embedding_dim,
  H = args.hidden_dim,
  zoneout = args.zoneout
  )
print('Created model with %d parameters' % sum((p.numel() for p in model.parameters())))

def do_benchmark_for(bsize):
  tmr = Timer()
  inp = torch.LongTensor(bsize, 1).random_(0, args.vocab_size)
  with torch.no_grad():
    model.clear_states()
    with tmr:
      for i in range(0, args.min_iter):
        model.forward(inp)
  return tmr.last / args.min_iter

bsize = 1
for bsize in range(args.min_batch, args.max_batch + 1):
  print("%5d " % bsize, end='')
print('')
for bsize in range(args.min_batch, args.max_batch + 1):
  itime = do_benchmark_for(bsize)
  print("%5.2f " % (1/itime), end='')
  sys.stdout.flush()
print('')

