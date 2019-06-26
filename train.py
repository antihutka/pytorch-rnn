import argparse
import logging
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--input_h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--input_json', default='data/tiny-shakespeare.json')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seq_length', default=64, type=int)

parser.add_argument('--checkpoint', default='models/output')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('train')

logger.info('Loading data')

loader = DataLoader(
  filename = args.input_h5,
  batch_size = args.batch_size,
  seq_length = args.seq_length
  )

for epoch in range(0, 1):
  traindata = loader.make_batches('train', 0)
  for iter_data in traindata.data:
    print('iteration %d/%d' % (iter_data.i, traindata.batch_count))
