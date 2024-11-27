class Hint():
    def __init__(self,topk=3):
        self.topk = topk
        self.hint=0
        self.sample=0
        self.steps=0
    def __call__(self, gt, idx):
        self.steps+=1
        self.sample+=1
        if gt in idx[:self.topk]:
            self.hint+=1
            return True
        else:
            return False
    def step(self):
        self.steps+=1
    def __repr__(self):
        return f"in-core-words-hint{self.topk}: {self.hint/float(self.sample)*100:.2f}%\nall-hint{self.topk}: {self.hint/float(self.steps)*100:.2f}%"