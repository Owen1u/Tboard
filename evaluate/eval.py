import sys
sys.path.append('../train')
import torch
import numpy as np
from functools import partial
from contextlib import nullcontext
from model import Transformer, ModelArgs
from process import Task
from tokenizer import Tokenizer
from benchmark import Benchmark
from metric import Hint

tokenizer_model = "../tokenizer/tok32000-en.model"
checkpoint="../out/ckpt.pt"
core = "../tokenizer/core_en.txt"
with open(core) as f:
    corewords = f.readlines()
corewords = [i.strip().replace('▁','').lower() for i in corewords if '▁' in i]
test_dataset = Benchmark(path='benchmark_v1.txt')

device='cpu'
dtype = "bfloat16"
vocab_source="benchmark"
max_seq_len = 64
vocab_size = 32000
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line


checkpoint = torch.load(checkpoint,map_location=torch.device(device),weights_only=True)
checkpoint_model_args = checkpoint["model_args"]
for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
    model_args[k] = checkpoint_model_args[k]
gptconf = ModelArgs(**model_args)
dada = Transformer(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
dada.load_state_dict(state_dict)
dada.eval()

hint = Hint(4)
bad_cases=[]


enc = Tokenizer(tokenizer_model)
for data in test_dataset:
    tokens = enc.encode(data, bos=True, eos=False)
    tokens = np.array(tokens, dtype=np.uint16)
    chunk = torch.from_numpy(tokens.astype(np.int64))
    x = chunk[:-1].unsqueeze(0)
    y = chunk[-1:]
    _last_word = enc.decode(y.tolist())
    if _last_word not in corewords:
        hint.step()
        continue
    logits = dada(x).squeeze(0)
    logits = torch.softmax(logits,dim=-1)
    logit_v,logit_idx = logits.topk(4,dim=-1)
    for gt,logit,idx in zip(y,logit_v,logit_idx):
        correct = hint(gt,idx)
        if not correct:
            bad_cases.append(data)

print(hint)

