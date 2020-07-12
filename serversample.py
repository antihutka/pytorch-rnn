import argparse
import aiohttp
import json
import asyncio
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num-files', default=8, type=int)
parser.add_argument('--total-bytes', default=1*1024*1024, type=int)
parser.add_argument('--output-path', default='outputs/out-%03d')
parser.add_argument('--backend-url', default='http://localhost:7880/')
parser.add_argument('--key', default='sampling:%d')
parser.add_argument('--start-text')
parser.add_argument('--line-start', default='')
args = parser.parse_args()

loop = asyncio.get_event_loop()

current_size = 0
urlget = args.backend_url + 'get'
urlput = args.backend_url + 'put'

files = [open(args.output_path % f, 'a') for f in range(0, args.num_files)]

async def run_task(session, idx):
  global current_size
  key = args.key % idx
  rq = {'key' : args.key % idx }
  outfile = files[idx]
  print('Task started for %s' % key)

  if args.start_text:
    rqp = {'key' : args.key % idx, 'text' : args.start_text}
    async with session.post(urlput, json=rqp) as response:
      assert response.status == 200
      r = await response.json()

  while(current_size < args.total_bytes):
    if args.line_start:
      prq = {'key' : args.key % idx, 'text' : args.line_start, 'append_newline' : False}
      async with session.post(urlput, json=prq) as response:
        r = await response.json()
    async with session.post(urlget, json=rq) as response:
      assert response.status == 200
      r = await response.json()
    o = r['text'] + '\n'
    print(args.line_start + o, file=outfile, end='')
    current_size += len(o.encode())
  print('Task %s finished' % key)

async def report_status():
  while (current_size < args.total_bytes):
    await asyncio.sleep(10)
    print('%d/%d' % (current_size, args.total_bytes))

async def run_test():
  timeout = aiohttp.ClientTimeout(total=900)
  async with aiohttp.ClientSession(loop=loop, timeout=timeout) as session:
    tasks = [run_task(session, i) for i in range(0, args.num_files)]
    await asyncio.gather(*tasks, report_status())

loop.run_until_complete(run_test())

for f in files:
  f.close()
