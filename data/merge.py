import os
import re
import random
import inflect
from textblob import TextBlob
from tqdm import tqdm

p = inflect.engine()

def replace_numbers_with_words(text):
    return re.sub(r'\b\d+\b', lambda x: p.number_to_words(int(x.group())), text)

files = ['data/reddit_clean','data/amazon']
new_file = open("data/reddit-amazon-5m.txt",'w',encoding='utf-8')

_count=5000000
queries = []

for file in files:
    with open(os.path.join(file,'train.txt')) as f:
        lines = f.readlines()
        queries.extend(lines[:_count])
        _count -= len(lines)
        
random.shuffle(queries)
for line in tqdm(queries):
    sentence = line.strip().lower()
    # try:
    #     sentence = replace_numbers_with_words(sentence)
    # except:
    #      continue
    
    # blob = TextBlob(sentence)
    # sentence = blob.correct()
    # sentence = str(sentence)
    
    new_file.write(sentence+'\n')

    