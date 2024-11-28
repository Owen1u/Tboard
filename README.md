# Mondo LLM
Mondo Large Language Model

Luke, 2024-11-06, AI Search Department
Minjun Lu, 2024-11-28, AI Search Department

## Dirs
- `train`: training & data processing
- `parse`: parsing raw files
- `data`: dataset for training and test
- `tokenizer`: tokenizer model for LLM (e.g. Llama2 or custom)
- `script`: .sh files
- `out`: output model and log files
- `inference`: C code for inference
- `evaluate`: Python for eval

## Usages
### Download
Manually download your own TXT file and put it under `./data/`.

### Split data into Train and Test
```bash
sh script/dataset_split.sh
```

### Train custom vocabulary source

> [!CAUTION]
> 请确保在任何时候（特别是小语种的定制化开发），训练和推理过程的tokenizer编码过程是一致的。
> 
> 训练过程由Python sentencepiece实现；推理过程由C语言实现。
>

```bash
sh script/train_vocab.sh
```

### Pretokenize queries in dataset
```bash
sh script/pretokenize.sh
```

### Training LLM
> [!IMPORTANT]
> Mention on your device (CPU/GPU/MPS). Ensure that GPUs support `torch.compile` and PyTorch version > 2.0.
```bash
sh script/train.sh
```

### Evaluation
> [!IMPORTANT]
> 根据产品需求设定TopK数
```bash
cd evaluate
python eval.py
```

### Inference
```bash
cd inference
make run
cd ..
sh script/inference.sh
```
