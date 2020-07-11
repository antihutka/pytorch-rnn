from aiohttp import web
import sampling
import samplingthread
import LanguageModel
from configparser import ConfigParser
import sys
import modules as M
import asyncio
from asyncio import Event
import json
import statsrequest
import logging

config = ConfigParser()
config.read(sys.argv[1])

logfile = config.get('logging', 'filename', fallback=None)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=config.get('loglevel', 'default'), filename=logfile)
logger = logging.getLogger(__name__)
for (k,v) in config['loglevel'].items():
  if k != 'default':
    logging.getLogger(k).setLevel(v)

model = LanguageModel.LanguageModel()
model.load_json(config.get('model', 'checkpoint'))
model.eval()

sampler = samplingthread.SamplerServer(model)
default_ending_token = model.token_to_idx[b'\n']

stortype = config.get('store', 'type')

if stortype == 'memory':
  stor = M.DefaultStateStore(model, default_token = default_ending_token)
elif stortype == 'sqlite':
  import sqlitestore
  stor = sqlitestore.SQLiteStateStore(model, config.get('store', 'dbpath'), default_token = default_ending_token)
elif stortype == 'mysql':
  import mysqlstore
  stor = mysqlstore.MySQLStore(model,
      config.get('store', 'dbhost', fallback='localhost'),
      config.get('store', 'dbname'),
      config.get('store', 'dbuser', fallback=''),
      config.get('store', 'dbpass', fallback=''),
      default_token = default_ending_token,
      modelid = config.getint('store', 'modelid'))
else:
  raise Exception('Unknown store type', stortype)

device = 'cpu'
if len(sys.argv) > 2:
  device = sys.argv[2]

model.to(device)

put_chain = sampling.default_put_chains(stor)

def encode(data):
  return json.dumps(data, indent=4).encode('utf-8')

def get_option(args, option):
  return args.get(option, config.get('defaults', option))

locks = {}

async def run_request(rq):
  evt = Event()
  loop = asyncio.get_event_loop()
  if rq.key not in locks:
    locks[rq.key] = asyncio.Lock()
  async with locks[rq.key]:
    sampler.run_request(rq, lambda: loop.call_soon_threadsafe(evt.set))
    await evt.wait()
    assert rq.finished
  return rq

async def run_request_nl(rq):
  evt = Event()
  loop = asyncio.get_event_loop()
  sampler.run_request(rq, lambda: loop.call_soon_threadsafe(evt.set))
  await evt.wait()
  assert rq.finished
  return rq

routes = web.RouteTableDef()

@routes.get('/')
async def stats(request):
  text = 'pytorch-rnn server stats\n'
  text += 'Pending samples:\n'
  for (k,v) in locks.items():
    if (v._locked):
      text += "%s => %d\n" % (k, len(v._waiters))
  try:
    rq = statsrequest.StatsRequest(sampler)
    await run_request_nl(rq)
    text += 'elapsed: %.3f\n' % rq.elapsed
    text += 'Requests:\n%s\n' % rq.requestinfo
  except Exception:
    text += "Error getting stats"
  return web.Response(text=text)

@routes.post('/put')
async def put(request):
  args = await request.json()
  key = args.get('key', '')
  text = args['text']
  force_commit = args.get('force_commit', False)
  logger.info("put %s %d" % (key,len(text)))
  rq = sampler.sampler.make_put_request(put_chain, model.encode_string(text + '\n'), key=key)
  if force_commit:
    rq.force_commit = True
  await run_request(rq)
  if force_commit and stortype == 'mysql':
    stor.commit()
  resp = {'result': 'ok'}
  return web.Response(body=encode(resp), content_type='application/json')

@routes.post('/get')
async def get(request):
  args = await request.json()
  key = args.get('key', '')
  temperature = float(get_option(args, 'temperature'))
  maxlength = int(get_option(args, 'maxlength'))
  bw = args.get('bad_words', [])
  logger.info("get %s bw:%d" % (key, len(bw)))
  softlength_max = int(get_option(args, 'softlength_max'))
  softlength_mult = float(get_option(args, 'softlength_mult'))
  ending_tokens = [default_ending_token]
  if 'ending_tokens' in args:
    ending_tokens = [model.token_to_idx[tok.encode('utf8')] for tok in args['ending_tokens']]
  get_chain = sampling.default_get_chains(stor, maxlength=maxlength, temperature = temperature, endtoken = ending_tokens)
  if softlength_max > 0:
    get_chain.sample_post.insert(0, M.SoftLengthLimit(softlength_max, softlength_mult, ending_tokens))
  if bw:
    get_chain.sample_post.append(M.BlockBadWords(model, bw))
  rq = await run_request(sampler.sampler.make_get_request(get_chain, key=key))
  seq = rq.sampled_sequence
  if bool(args.get('strip_ending_token', True)) and seq[-1] in ending_tokens:
    seq.pop(-1)
  text = model.decode_string(seq).decode('utf8', 'ignore')
  resp = {'result': 'ok', 'text': text}
  return web.Response(body=encode(resp), content_type='application/json')

app = web.Application()
app.add_routes(routes)
web.run_app(
  app,
  port = config.get('http', 'port'),
)
