import argparse
import logging
from dataloader import DataLoader
from LanguageModel import LanguageModel
import torch.optim as optim
import torch.nn as nn
import torch
import time
import json
from trainutils import Timer, Average, ValueHistory

parser = argparse.ArgumentParser()
parser.add_argument('--input-h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--input-json', default='data/tiny-shakespeare.json')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--seq-length', default=64, type=int)
parser.add_argument('--no-offset', default=False, action='store_true')
parser.add_argument('--double-seq-on', default='')

parser.add_argument('--num-epochs', default=50, type=int)

parser.add_argument('--layer-type', default='GRIDGRU')
parser.add_argument('--num-layers', default=2, type=int)
parser.add_argument('--embedding-dim', default=128, type=int)
parser.add_argument('--hidden-dim', default=128, type=int)
parser.add_argument('--zoneout', default=0, type=float)
parser.add_argument('--dropout', default=0, type=float)

parser.add_argument('--learning-rate', default=0.002, type=float)
parser.add_argument('--lrdecay-every', default=5, type=int)
parser.add_argument('--lrdecay-factor', default=0.5, type=float)
parser.add_argument('--grad-clip', default=5, type=float)

parser.add_argument('--checkpoint-name', default='models/output')
parser.add_argument('--device', default='cpu')
parser.add_argument('--print-every', default=1, type=float)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('train')

logger.info('Creating model')
model = LanguageModel()
model.load_tokendata(args.input_json)
model.build_model(
  layertype = args.layer_type,
  dropout = args.dropout,
  num_layers = args.num_layers,
  D = args.embedding_dim,
  H = args.hidden_dim,
  zoneout = args.zoneout
  )
print(model.layers)
logger.info('Created model with %d parameters' % sum((p.numel() for p in model.parameters())))
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.lrdecay_every, args.lrdecay_factor)
crit = nn.CrossEntropyLoss()

logger.info('Loading data')

loader = DataLoader(
  filename = args.input_h5,
  batch_size = args.batch_size,
  seq_length = args.seq_length
  )

device = torch.device(args.device)
model.to(device)

double_seq_on = [int(x) for x in args.double_seq_on.split(',')]

totalfwd = 0
totalbck = 0
timer_pre = Timer()
timer_fwd = Timer()
timer_bck = Timer()
timer_tot = Timer()
avg_tloss = Average(100)
vloss_history = ValueHistory('val loss')
tloss_history = ValueHistory('train loss')

for epoch in range(0, args.num_epochs):
  if epoch in double_seq_on:
    args.seq_length *= 2
    args.batch_size //= 2
    loader.set_seq_batch(args.seq_length, args.batch_size)
    logger.info('Doubling sequence length to seq_length=%d batch_size=%d' % (args.seq_length, args.batch_size))
  traindata = loader.make_batches('train', 0 if (args.no_offset or epoch % 2 == 0) else (args.seq_length // 2))
  timer_pre.reset()
  timer_fwd.reset()
  timer_bck.reset()
  timer_tot.reset()
  totalloss = 0
  model.train()
  for iter_data in traindata.data:
    timer_tot.start()
    N = iter_data.inputs.size(0)
    T = iter_data.inputs.size(1)
    optimizer.zero_grad()
    model.clear_states()
    with torch.no_grad(), timer_pre:
      model(iter_data.preinputs.to(device).long())
    with timer_fwd:
      outputs = model(iter_data.inputs.to(device).long())
      loss = crit(outputs.view(N*T, -1), iter_data.outputs.to(device).long().view(N*T))
    with timer_bck:
      loss.backward()
    if args.grad_clip > 0:
      for par in model.parameters():
        par.grad.clamp_(-args.grad_clip, args.grad_clip)
    optimizer.step()
    timer_tot.stop()
    totalloss += loss.detach()
    avg_tloss.add_value(loss.detach())
    tloss_history.add_value(epoch + iter_data.i / traindata.batch_count, float(loss))
    if iter_data.i % args.print_every == 0:
      assert not torch.isnan(loss)
      print('ep %d/%d iter %d/%d loss=%.4f, %.4f lr=%.2e Times: %.2f %.2f %.2f %.2f (%4.1f tps)' %
        (epoch, args.num_epochs, iter_data.i, traindata.batch_count, loss, avg_tloss.avg(), optimizer.param_groups[0]['lr'], timer_pre.last, timer_fwd.last, timer_bck.last, timer_tot.last, N*T/timer_tot.average()))
  print('average loss: %.4f' % (totalloss.item()/traindata.batch_count))

  model.clear_states()
  model.eval()
  valdata = loader.make_batches('val', shuffle=False)
  timer_tot.reset()
  timer_fwd.reset()
  with torch.no_grad():
    totalloss = torch.Tensor([0])
    for iter_data in valdata.data:
      timer_tot.start()
      if iter_data.preinputs is not None:
        model(iter_data.preinputs.to(device).long())
      with timer_fwd:
        outputs = model(iter_data.inputs.to(device).long())
      loss = crit(outputs.view(N*T, -1), iter_data.outputs.to(device).long().view(N*T))
      totalloss += loss
      timer_tot.stop()
      if iter_data.i % args.print_every == 0:
        print('ep %d/%d iter %d/%d loss: %.4f Time: %.2f %.2f (%4.1f tps)' % (epoch, args.num_epochs, iter_data.i, valdata.batch_count, loss, timer_fwd.last, timer_tot.last, (iter_data.inputs.size(0)*iter_data.inputs.size(1))/timer_tot.last))
    vloss_history.add_value(epoch, totalloss.item()/valdata.batch_count)
    print(vloss_history.format())
  scheduler.step()

  model.save_model("%s_%d" % (args.checkpoint_name, epoch))
  with open("%s_%d_stats.json" % (args.checkpoint_name, epoch), 'w') as f:
    json.dump(
      {"train_loss" : tloss_history.values, "train_loss_i" : tloss_history.steps,
       "val_loss" : vloss_history.values, "val_loss_i" : vloss_history.steps}, f)
