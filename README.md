# pytorch-rnn
A rewrite of torch-rnn using PyTorch. Training is being worked on now, and torch-rnn checkpoints can be loaded and sampled from. An extensible and efficient HTTP sampling server has been implemented.

# Installation
Install PyTorch using the [official guide](https://pytorch.org/get-started/locally/)

# Data preprocessing
At the moment you'll have to use the preprocessing scripts from [torch-rnn](https://github.com/jcjohnson/torch-rnn). Only GridGRU models are supported at this time.

# Training
Train the network using `train.py`.
```bash
python3 train.py --input-h5 my_data.h5 --input-json my_data.json --checkpoint-name my_model
```
This will create two files `my_checkpoint_N.json` and `my_checkpoint_N.0` per epoch, where the JSON file contains architecture description and the .0 file contains raw model parameters. This allows faster, more flexible and more efficient model saving/loading.
You can use GPU using ``--device cuda``, but this is barely tested at this time.
When training on CPU, make sure to set the optimal number of threads using the OMP_NUM_THREADS environment variable - otherwise pytorch defaults to using all cores, which seems to cause a huge slowdown.
Also when running on a NUMA system, try binding the process to one node using numactl.
```bash
OMP_NUM_THREADS=3 numactl -m 0 -N 0 python3 train.py ...
```

# Using the model
`sampling.py` implements an extensible and efficient sampling module.
You can sample output from the model using `sample.py`:
```bash
python3 sample.py --checkpoint my_model.json
```
A simple chat application, `chatter.py` is also included. An efficient HTTP sampling server is also included. Edit the example config file and start the server:
```bash
python3 httpserver.py config.cfg
```
Then you can send text to the model and generate responses using a simple HTTP interface and specify different options for text generation:
```bash
curl -X POST -d '{"key":"test", "text":"User input"}' http://localhost:7880/put
curl -X POST -d '{"key":"test"}' http://localhost:7880/get
curl -X POST -d '{"key":"test", "bad_words":["stinky"], "temperature":0.7, "softlenght_max" : 50}' http://localhost:7880/get
```
The server can handle multiple parallel requests by packing them into one batch, which allows efficient generation of dozens of text streams at the same time.

# Differences from `torch-rnn`
- Only GridGRU layers are implemented at this time, based on [guillitte's implementation](https://github.com/guillitte/torch-rnn/tree/Dev). More layer types will be implemented later
- String decoder works on byte level and is fully encoding-agnostic. Any tokenization scheme (bytes, unicode, words...) should work, as long as it can be decoded by a greedy algorithm.
- Training now gives expected results. For some reason PyTorch 1.0 was causing gradient issues, but updating to 1.1 fixed it.
