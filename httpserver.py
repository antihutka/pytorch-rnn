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

config = ConfigParser()
config.read(sys.argv[1])

model = LanguageModel.LanguageModel()
model.load_json(config.get('model', 'checkpoint'))
model.eval()

sampler = samplingthread.SamplerServer(model)
default_ending_token = model.token_to_idx[b'\n']

stor = M.DefaultStateStore(model, default_token = default_ending_token)
put_chain = sampling.default_put_chains(stor)
#gc = sampling.default_get_chains(stor, endtoken = [default_ending_token], maxlength=args.maxlength, temperature = args.temperature)
#badword_mod = M.BlockBadWords(model, [])

def encode(data):
  return json.dumps(data, indent=4).encode('utf-8')

def get_option(args, option):
  return args.get(option, config.get('defaults', option))

async def run_request(rq):
  evt = Event()
  loop = asyncio.get_event_loop()
  sampler.run_request(rq, lambda: (print('done'), loop.call_soon_threadsafe(evt.set)))
  await evt.wait()
  return rq

routes = web.RouteTableDef()

@routes.get('/')
async def stats(request):
  return web.Response(text='pytorch-rnn server stats')

@routes.post('/put')
async def put(request):
  args = await request.json()
  key = args.get('key', '')
  text = args['text']
  await run_request(sampler.sampler.make_put_request(put_chain, model.encode_string(text + '\n')))
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
  rq = await run_request(sampler.sampler.make_get_request(get_chain))
  text = model.decode_string(rq.sampled_sequence).decode('utf8', 'ignore')
  resp = {'result': 'ok', 'text': text}
  return web.Response(body=encode(resp), content_type='application/json')

app = web.Application()
app.add_routes(routes)
web.run_app(
  app,
  port = config.get('http', 'port'),
)
