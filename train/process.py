"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
import zipfile

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data/"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(hf_data,filename):
    """Downloads the dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = hf_data
    data_filename = os.path.join(DATA_CACHE_DIR, f"{filename}.zip")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, filename)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        with zipfile.ZipFile(data_filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example from a .jsonl file just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    with open(shard_filenames[0], "r") as f:
        # Read the first line of the .jsonl file
        first_line = f.readline()
        data = json.loads(first_line)

    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data}")


def train_vocab(vocab_size,file_path,defined_tokens):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join("tokenizer", f"tok{vocab_size}")

    
    print(f"Size is: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=file_path,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity",
                                   user_defined_symbols=defined_tokens
                                   )

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def extract_text_from_sections(sections):
    """
    Extracts all text from sections -> has_parts -> value.
    Skips sections that do not have the required structure.
    """
    all_text = []
    for section in sections:
        if 'has_parts' in section:
            for part in section['has_parts']:
                if 'value' in part:
                    all_text.append(part['value'])
                elif 'has_parts' in part:
                    # If there are nested has_parts, we recursively check them
                    all_text.extend(extract_text_from_sections([part]))
    return all_text


def process_shard(args, vocab_size,tokenizer_model):
    shard_id, shard = args
    tokenizer_model = tokenizer_model if vocab_size==0 else get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    all_tokens = []

    with open(shard, "r") as f:
        for line in f.readlines():
            try:
                # data = json.loads(line)

                # if 'sections' in data:
                #     # Extract text from sections
                #     extracted_texts = extract_text_from_sections(data['sections'])
                text = line.strip().lower()
                    # for text in extracted_texts:
                    #     text = text.strip()  # clean the text
                tokens = enc.encode(text, bos=True, eos=False)  # tokenize
                all_tokens.extend(tokens)

            except json.JSONDecodeError:
                print(f"Skipping malformed line in {shard}")

    # Convert to uint16 numpy array
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # Determine output file path
    if vocab_size == 0:
        tokenized_filename = shard.replace(".txt", ".bin")
    else:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".txt", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)

    # Write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    # Calculate average sequence length
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size,dirname,tokenizer_model):
    # Process all shards in the dataset
    data_dir = os.path.join(DATA_CACHE_DIR, dirname)
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    print(shard_filenames)

    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # Process shards in parallel
    fun = partial(process_shard, vocab_size=vocab_size,tokenizer_model=tokenizer_model)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))

    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        bin_dir = os.path.join(DATA_CACHE_DIR, self.vocab_source)
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

def split_txt(filename):
    count=0
    filepath = os.path.join(DATA_CACHE_DIR,filename)
    basename = filename.split('.')[0]
    dirname = os.path.join(DATA_CACHE_DIR,basename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    train = open(f"{dirname}/{basename}_train.txt","w", encoding="utf-8")
    test = open(f"{dirname}/{basename}_test.txt","w",encoding="utf-8")

    with open(filepath) as f:
        for line in tqdm(f.readlines()):
            count+=1
            line = line.strip()
            if count%10==0:
                test.write(line+'\n')
            else:
                train.write(line+'\n')


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab","split"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--hf_data", type=str, default="", help="dataset path from huggingface.")
    parser.add_argument("--filename", type=str, default="", help="filename of dataset.")
    parser.add_argument("--tokenizer_model", type=str, default="tokenizer/llama2.model", help="tokenizer_model.")
    parser.add_argument("--defined_tokens", type=str, default="", help="defined_tokens.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        assert args.hf_data and args.filename,"You must input huggingface url and filename"
        download(args.hf_data,args.filename)
    elif args.stage == "train_vocab":
        assert args.filename
        if args.defined_tokens:
            with open(args.defined_tokens) as defined_tokens_files:
                defined_tokens = defined_tokens_files.readlines()
                defined_tokens = [x.strip().lower() for x in defined_tokens]
                # 请在核心词文件预处理阶段完成该操作
                # defined_tokens = ['▁'+x for x in defined_tokens]
        else:
            defined_tokens=[""]
        train_vocab(vocab_size=args.vocab_size,file_path=args.filename,defined_tokens=defined_tokens)
    elif args.stage == "pretokenize":
        assert args.filename and args.tokenizer_model
        pretokenize(vocab_size=args.vocab_size,dirname=args.filename.split('.')[0],tokenizer_model=args.tokenizer_model)
    elif args.stage == "split":
        assert args.filename
        split_txt(args.filename)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
