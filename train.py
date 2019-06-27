import argparse
import logging
from dataloader import DataLoader
from LanguageModel import LanguageModel
import torch.optim as optim
import torch.nn as nn
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input_h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--input_json', default='data/tiny-shakespeare.json')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seq_length', default=64, type=int)

parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--zoneout', default=0, type=float)
parser.add_argument('--dropout', default=0, type=float)

parser.add_argument('--checkpoint', default='models/output')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('train')

logger.info('Creating model')
model = LanguageModel()
model.load_tokendata(args.input_json)
model.build_model(
  layertype = 'GRIDGRU',
  dropout = args.dropout,
  num_layers = args.num_layers,
  D = args.embedding_dim,
  H = args.hidden_dim,
  zoneout = args.zoneout
  )
print(model.layers)
logger.info('Created model with %d parameters' % sum((p.numel() for p in model.parameters())))
optimizer = optim.Adam(model.parameters())
crit = nn.CrossEntropyLoss()

logger.info('Loading data')

loader = DataLoader(
  filename = args.input_h5,
  batch_size = args.batch_size,
  seq_length = args.seq_length
  )

totalfwd = 0
totalbck = 0

for epoch in range(0, 1):
  traindata = loader.make_batches('train', 0)
  for iter_data in traindata.data:
    tstart = time.clock()
    N = iter_data.inputs.size(0)
    T = iter_data.inputs.size(1)
    optimizer.zero_grad()
    # handle pre-input
    tfwd_start = time.clock()
    outputs = model(iter_data.inputs.long())
    loss = crit(outputs.view(N*T, -1), iter_data.outputs.long().view(N*T))
    tfwd_end = time.clock()
    loss.backward()
    tbck_end = time.clock()
    optimizer.step()
    tend = time.clock()
    print('iteration %d/%d loss %.2f time %.2f fwd %.2f bck %.2f' % (iter_data.i, traindata.batch_count, loss, tend - tstart, tfwd_end - tfwd_start, tbck_end - tfwd_end))
    totalfwd += tfwd_end-tfwd_start
    totalbck += tbck_end-tfwd_end
    if iter_data.i > 10:
      break

print("total fwd/bck: %.2f/%.2f" % (totalfwd, totalbck))