import os

new_f = open("reddit-amazon-5m-.txt",'w',encoding='utf-8')
with open("reddit-amazon-5m.txt") as f:
    for line in f.readlines():
        if 'reddit' in line or 'amazon' in line:
            continue
        else:
            new_f.write(line)