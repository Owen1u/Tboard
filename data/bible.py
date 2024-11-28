import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

def spider(chapter,num):
    resp = requests.get(f"https://www.bible.com/bible/71/{chapter}.{num}.HAU")
    results = []
    soup = BeautifulSoup(resp.text, 'lxml')
    spans = soup.find_all('span', class_='ChapterContent_content__RrUqA')
    for span in spans:
        text = span.get_text().strip().lower()
        text = re.sub(r'[“”]',"",text)
        text = text.replace('ɓ','b').replace('ƙ','k').replace('ɗ','d').replace('y','y').replace('ă','a')
        if text:
            if text[-1] in ['.','?','!']:
                results.append(text+'\n')
            else:
                results.append(text)
    return results

if __name__=='__main__':

    CHAPTER={
        "GEN":50,
        "EXO":40,
        "LEV":27,
        "NUM":36,
        "DEU":34,
        "JOS":24,
        "JDG":21,
        "RUT":4,
        "1SA":31,
        "2SA":24,
        "1KI":22,
        "2KI":25,
        "1CH":29,
        "2CH":36,
        "EZR":10,
        "NEH":13,
        "EST":10,
        "JOB":42,
        "PSA":150,

        "PRO":31,
        "ECC":12,
        "SNG":8,
        "ISA":66,
        "JER":52,
        "LAM":5,
        "EZK":48,
        "DAN":12,
        "HOS":14,
        "JOL":3,
        "AMO":9,
        "OBA":1,
        "JON":4,
        "MIC":7,
        "NAM":3,
        "HAB":3,
        "ZEP":3,
        "HAG":2,
        "ZEC":14,
        "MAL":4,
        "MAT":28,
        "MRK":16,
        "LUK":24,
        "JHN":21,
        "ACT":28,
        "ROM":16,
        "1CO":16,
        "2CO":13,
        "GAL":6,
        "EPH":6,
        "PHP":4,
        "COL":4,

        "1TH":5,
        "2TH":3,
        "1TI":6,
        "2TI":4,
        "TIT":3,
        "PHM":1,
        "HEB":13,
        "JAS":5,
        "1PE":5,
        "2PE":3,
        "1JN":5,
        "2JN":1,
        "3JN":1,
        "JUD":1,
        "REV":22
    }
    for chapter,num in tqdm(CHAPTER.items()):
        file = open(f"bible-{chapter}.txt","w",encoding="utf-8")
        for i in tqdm(range(1,num+1),desc=chapter):
            results = spider(chapter,i)
            for res in results:
                file.write(res)