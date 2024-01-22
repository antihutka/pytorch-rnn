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
parser.add_argument('--use-masks', default=False, action='store_true')
parser.add_argument('--double-seq-on', default='')
parser.add_argument('--max-batches', default=None, type=int)
parser.add_argument('--max-batches-val', default=None, type=int)

parser.add_argument('--num-epochs', default=50, type=int)

parser.add_argument('--load-model')
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
parser.add_argument('--warmup-iters', default=50, type=int)

parser.add_argument('--checkpoint-name', default='models/output')
parser.add_argument('--device', default='cpu')
parser.add_argument('--layerdevices', default=[], nargs='+')
parser.add_argument('--swapoutlayers', default=[], type=int, nargs='+')
parser.add_argument('--print-every', default=1, type=float)
parser.add_argument('--bf16', default=False, action='store_true')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('train')

logger.info('Creating model')
def get_model():
  m = LanguageModel()
  if args.load_model is None:
    m.load_tokendata(args.input_json)
    m.build_model(
      layertype = args.layer_type,
      dropout = args.dropout,
      num_layers = args.num_layers,
      D = args.embedding_dim,
      H = args.hidden_dim,
      zoneout = args.zoneout
      )
  else:
    m.load_json(args.load_model, clone_tensors=True)
    m.replace_tokendata(args.input_json)
  return m
model = get_model()
print(model.layers)

logger.info('%s model with %d parameters' % ('Created' if args.load_model is None else 'Loaded', sum((p.numel() for p in model.parameters()))))
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.lrdecay_every, args.lrdecay_factor)
scheduler_start = optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, total_iters=args.warmup_iters)
crit = nn.CrossEntropyLoss(reduction = 'none' if args.use_masks else 'mean')

logger.info('Loading data')

loader = DataLoader(
  filename = args.input_h5,
  batch_size = args.batch_size,
  seq_length = args.seq_length
  )

if args.bf16:
  model.to(dtype=torch.bfloat16)

device = torch.device(args.device)
if args.layerdevices:
  for ld in args.layerdevices:
    start, end, device = ld.split(',')
    for layerid in range(int(start), int(end)+1):
      print("Moving layer %d-%s to device %s" % (layerid, model.layers[layerid], device))
      model.layers[layerid].to(device)
else:
  model.to(device)

for swl in args.swapoutlayers:
  print("Enabling swapout for layer %d" % swl)
  model.layers[swl].set_swapout(True)

double_seq_on = [int(x) for x in args.double_seq_on.split(',')] if args.double_seq_on else []

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
  traindata = loader.make_batches('train', 0 if (args.no_offset or epoch % 2 == 0) else (args.seq_length // 2), use_masks = args.use_masks, max_batches = args.max_batches)
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
      model(iter_data.preinputs.long())
    with timer_fwd:
      outputs = model(iter_data.inputs.long())
      loss = crit(outputs.contiguous().view(N*T, -1), iter_data.outputs.to(device).long().view(N*T))
      if args.use_masks:
        masks = iter_data.masks.float().to(device).view(N*T)
        masksum = iter_data.masks.sum()
        loss_unmasked = loss.sum() / loss.numel()
        loss = (loss * masks).sum() / masksum
      outputs = None
    with timer_bck:
      loss.backward()
    if args.grad_clip > 0:
      for par in model.parameters():
        par.grad.clamp_(-args.grad_clip, args.grad_clip)
    optimizer.step()
    timer_tot.stop()
    assert not torch.isnan(loss)
    loss = loss.item()
    totalloss += loss
    avg_tloss.add_value(loss)
    tloss_history.add_value(epoch + iter_data.i / traindata.batch_count, loss)
    if iter_data.i % args.print_every == 0:
      s = 'ep %d/%d iter %d/%d ' % (epoch, args.num_epochs, iter_data.i, traindata.batch_count)
      s += 'loss=%.4f, %.4f lr=%.2e ' % (loss, avg_tloss.avg(), optimizer.param_groups[0]['lr'])
      if args.use_masks:
        s += 'uloss %.4f masked %d/%d ' % (loss_unmasked, iter_data.masks.sum(), iter_data.masks.numel())
      s +='Times: %.2f %.2f %.2f %.2f (%4.1f tps) ' % (timer_pre.last, timer_fwd.last, timer_bck.last, timer_tot.last, N*T/timer_tot.average())
      s += "%.2fh remaining" % (timer_tot.average() * (traindata.batch_count - iter_data.i) / 3600)
      print(s)
    scheduler_start.step()
  print('average loss: %.4f' % (totalloss/traindata.batch_count))

  model.clear_states()
  model.eval()
  valdata = loader.make_batches('val', shuffle=False, use_masks = args.use_masks, max_batches = args.max_batches_val)
  timer_tot.reset()
  timer_fwd.reset()
  with torch.no_grad():
    totalloss = 0
    for iter_data in valdata.data:
      timer_tot.start()
      if iter_data.preinputs is not None:
        model(iter_data.preinputs.long())
      with timer_fwd:
        outputs = model(iter_data.inputs.long())
      loss = crit(outputs.view(N*T, -1), iter_data.outputs.to(device).long().view(N*T))
      if args.use_masks:
        masks = iter_data.masks.float().to(device).view(N*T)
        masksum = iter_data.masks.sum()
        loss = (loss * masks).sum() / masksum
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
