from concurrent.futures import ThreadPoolExecutor,as_completed
from googletrans import Translator
from tqdm import tqdm
import requests
translator = Translator()
# detected = translator.detect("Hello, world!")
# print(detected.lang)
# translated = translator.translate("i want to eat apples",src='en', dest='ha')
# print(translated.text)

# def translate2ha(q):
#     query = q.strip().lower()
#     translated = translator.translate(query,src='en', dest='ha')
#     return translated.text

def trans(q:str):
    q = q.strip().lower()
    resp = requests.post("http://10.220.1.57:8081/indexer/translate",headers={'Content-Type': 'application/json'},json={
        "fromLanguage": "en",
        "toLanguage": "ha",
        "text": q
    })
    if resp.status_code:
        info = resp.json()
        trans_t = info['data']['result']['texts'][0]
        return trans_t.strip().lower()

filename = "reddit-amazon-5m.txt"
hausa_file = open("reddit-amazon-5m-hausa.txt","w",encoding='utf-8')
n_threads = 32
with open(filename) as f:
    samples = f.readlines()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for n in tqdm(range(len(samples)//n_threads)):

            futures = {executor.submit(trans, line): line for line in samples[n*n_threads:(n+1)*n_threads]}
                
            # 等待每个任务完成，并将结果写入文件
            for future in as_completed(futures):
                result = future.result()  # 获取处理结果
                hausa_file.write(result+'\n')




    # for n in tqdm(range(len(samples)//10)):
    #     lines = samples[n*10:(n+1)*10]
    #     lines = [l.strip().lower() for l in lines]
    #     translations = translator.translate(lines,src='en', dest='ha')
    #     for trans in translations:
    #         hausa_file.write(trans.text+'\n')
