# vocab_size一直设0，修改tokenizer_model即可。
python train/process.py pretokenize --filename reddit-amazon-5m --vocab_size 0 --tokenizer_model tokenizer/tok32000.model