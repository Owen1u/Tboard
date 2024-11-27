import re 
from torch.utils.data import Dataset

class Benchmark(Dataset):
    def __init__(self,path='benchmark.txt'):
        self.data=[]
        with open(path) as f:
            for line in f.readlines():
                line = line.strip()
                words = re.split(r'[^a-zA-Z\']+',line)
                for i in range(1,len(words)):
                    if bool(re.fullmatch(r"[a-zA-Z']+", words[i])):
                        _query = ' '.join(words[:i+1]).replace(" ,",",")
                        self.data.append(_query)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
if __name__=='__main__':
    bm = Benchmark()
    print(bm.data)