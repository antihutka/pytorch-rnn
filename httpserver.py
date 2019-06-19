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

config = ConfigParser()
config.read(sys.argv[1])

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
else:
  raise Exception('Unknown store type', stortype)

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
    rq = statsrequest.StatsRequest(sampler.sampler)
    await run_request_nl(rq)
    text += 'elapsed: %.3f' % rq.elapsed
  except Exception:
    text += "Error getting stats"
  return web.Response(text=text)

@routes.post('/put')
async def put(request):
  args = await request.json()
  key = args.get('key', '')
  text = args['text']
  await run_request(sampler.sampler.make_put_request(put_chain, model.encode_string(text + '\n'), key=key))
  resp = {'result': 'ok'}
  return web.Response(body=encode(resp), content_type='application/json')

@routes.post('/get')
async def get(request):
  args = await request.json()
  key = args.get('key', '')
  temperature = float(get_option(args, 'temperature'))
  maxlength = int(get_option(args, 'maxlength'))
  ending_tokens = [default_ending_token]
  if 'ending_tokens' in args:
    ending_tokens = [model.token_to_idx[tok.encode('utf8')] for tok in args['ending_tokens']]
  get_chain = sampling.default_get_chains(stor, maxlength=maxlength, temperature = temperature, endtoken = ending_tokens)
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
