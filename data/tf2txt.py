from datasets import load_dataset
import os
import re
os.environ['NLTK_DATA']="tokenizer"
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab',download_dir="tokenizer")

ds_path = "data/amazon"
ds_wiki = load_dataset(ds_path)
transcripts = set()

f = open(os.path.join(ds_path,"train.txt"),'w',encoding='utf-8')

for ds in tqdm(ds_wiki['train'],total=len(ds_wiki['train'])):
    text = ds['review_text']
    text = re.sub(r'\(.*?\)|\[.*?\]|\<.*?\>|\{.*?\}', '', text)
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\"+','',text)
    sentences = sent_tokenize(text)
    for sent in sentences:
        sent = sent.strip().lower()
        sent = re.sub(r'\.+','.',sent)
        sent = re.sub(r'\!+','!',sent)
        sent = re.sub(r'\?+','?',sent)
        # sent = re.sub(r'\b\d+\b','[NUM]',sent)
        if len(sent.split())<=5 or '-' in sent or '/' in sent :
            continue
        if sent[-1] not in ['.','?','!',';']:
            sent+='.'
        f.write(sent+'\n')


