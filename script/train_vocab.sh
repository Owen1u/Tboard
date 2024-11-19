vocab_size=32000
# lang="en"

python train/process.py train_vocab --vocab_size $vocab_size --filename data/reddit-amazon-5m.txt --defined_tokens tokenizer/core_en.txt
python train/tokenizer.py --tokenizer-model=tokenizer/tok$vocab_size.model