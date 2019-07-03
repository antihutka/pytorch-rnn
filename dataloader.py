import h5py
import torch
import logging

logger = logging.getLogger('dataloader')

def split_tensor(tensor, cnt, length):
  tensor = tensor[0:cnt*length]
  tensor = tensor.view(cnt, length)
  return tensor

class IterationData:
  def __init__(self, i, preinputs, inputs, outputs):
    self.preinputs = preinputs
    self.inputs = inputs
    self.outputs = outputs
    self.i = i

class EpochData:
  def __init__(self, gen, bcnt):
    self.data = gen
    self.batch_count = bcnt

class DataLoader:
  def __init__(self, filename, batch_size, seq_length):
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.splits = {}
    with h5py.File(filename, 'r') as f:
      for spl in ['test', 'val', 'train']:
        self.splits[spl] = torch.from_numpy(f[spl][:])
        logger.info('Loaded %d items from %s' % (self.splits[spl].size(0), spl))
    if all((s.min()>0 for s in self.splits.values())):
      logger.info('No zeroes found in data, assuming one-based indexes')
      for t in self.splits.values():
        t.add_(-1)

  def make_batches(self, splitname = 'train', offset = 0, shuffle = True):
    data = self.splits[splitname]
    inputs = data[offset:-2]
    outputs = data[offset+1:-1]
    numseq = inputs.size(0) // self.seq_length
    numbat = (numseq-1) // self.batch_size
    logger.info('%d sequences, %d batches for split %s with offset %d' % (numseq, numbat, splitname, offset))
    if shuffle:
      permutation = split_tensor(torch.randperm(numseq - 1), numbat, self.batch_size)
    else:
      permutation = split_tensor(torch.arange(0, self.batch_size*numbat), self.batch_size, numbat).t()
    inputs_split = split_tensor(inputs, numseq, self.seq_length)
    outputs_split = split_tensor(outputs, numseq, self.seq_length)
    def gen():
      for i in range(0, numbat):
        bperm = permutation[i]
        bperm_next = bperm.add(1)
        if shuffle or i == 0:
          i_preinputs = torch.index_select(inputs_split, 0, bperm)
        else:
          i_preinputs = None
        i_inputs = torch.index_select(inputs_split, 0, bperm_next)
        i_outputs = torch.index_select(outputs_split, 0, bperm_next)
        yield IterationData(i, i_preinputs, i_inputs, i_outputs)
    return EpochData(gen(), numbat)
