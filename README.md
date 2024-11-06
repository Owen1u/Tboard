# Mondo LLM
Mondo Large Language Model

Luke, 2024-11-06, AI Search Department


## Dirs
- `train`: training & data processing
- `parse`: parsing raw files
- `data`: dataset for training and test
- `tokenizer`: tokenizer model for LLM (e.g. Llama2 or custom)
- `script`: .sh files
- `out`: output model and log files

## Usages
### Download
Manually download your own TXT file and put it under `./data/`.

### Split data into Train and Test
```bash
sh script/dataset_split.sh
```

### Train custom vocabulary source
> [!NOTE]
> This step is Optional.
```bash
sh script/train_vocab.sh
```

### Pretokenize queries in dataset
```bash
sh script/pretokenize.sh
```

### Training LLM
> [!IMPORTANT]
> Mention on your device (CPU/GPU/MPS).
```bash
sh script/train.sh
```
