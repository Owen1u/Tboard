import sys
sys.path.append('../train')
import torch
from functools import partial
from contextlib import nullcontext
from model import Transformer, ModelArgs
from process import Task
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

gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
checkpoint="/Users/luminjun/Documents/mondo/out/dada-en-vocab32000-d288-len64-lr5e-05-ly6-h6.bin"

model.load_state_dict(torch.load("/Users/luminjun/Documents/mondo/out/dada-en-vocab32000-d288-len64-lr5e-05-ly6-h6.bin"))

# model.load_state_dict(torch.load(checkpoint),strict=False)
model.eval()
model.to(device)

# # compile the model
# if compile:
#     print("compiling the model... (takes a ~minute)")
#     unoptimized_model = model
#     model = torch.compile(model)  # requires PyTorch 2.0

iter_batches = partial(
    Task.iter_batches,
    batch_size=1,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
eval_iters = 10
batch_iter = iter_batches(split="val")
for k in range(eval_iters):
    X, Y = next(batch_iter)
    logits = model(X, Y)
    print(logits)

