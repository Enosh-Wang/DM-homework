import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import sys
import pypinyin
from collections import defaultdict

def check_chs(c):
    return '\u4e00' <= c <= '\u9fa5'

def to_pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s

##
whole_pub_info = pd.read_pickle('./pkl/whole_pub_info.pkl')
whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')
## 
train_pub_info = pd.read_pickle('./pkl/train_pub_info.pkl')
train_author_name_ids = pd.read_pickle('./pkl/train_author_name_ids.pkl')
train_author_paper_ids = pd.read_pickle('./pkl/train_author_paper_ids.pkl')
##
valid_pub_info = pd.read_pickle('./pkl/valid_pub_info.pkl')
cna_valid_unass = pd.read_pickle('./pkl/cna_valid_unass.pkl')

pub_info = pd.concat([whole_pub_info, train_pub_info, valid_pub_info])
pub_info = pub_info.drop_duplicates(subset='paper_id', keep='first')

# id 到 name 列表的映射
paper_authors = {}
for author_name, paper_ids in whole_author_name_paper_ids[['author_name', 'paper_ids']].values:
    for pid in paper_ids:
        if not pid in paper_authors:
            paper_authors[pid] = [author_name]
        else:
            paper_authors[pid].append(author_name)

paper_authors_df = pd.DataFrame([(k, v) for k,v in paper_authors.items()], columns=['paper_id', 'author_ids'])


pub_info['author_names'] = pub_info['authors'].apply(lambda x: [ao['name'] for ao in x])


pub_info = pub_info.merge(paper_authors_df, 'left', 'paper_id')




def score(n1, n2):
    n1 = ''.join(filter(str.isalpha, n1.lower()))
    if check_chs(n1):
        n1 = to_pinyin(n1)
    n2 = ''.join(filter(str.isalpha, n2.lower()))
    counter = defaultdict(int)
    score = 0
    for c in n1:
        counter[c] += 1
    for c in n2:
        if (c in counter) and (counter[c] > 0):
            counter[c] -= 1
        else:
            score += 1
    score += np.sum(list(counter.values()))
    return score
    

from tqdm import tqdm
author_name_map = {}
for author_names, author_ids in tqdm(pub_info[['author_names', 'author_ids']].values):
    if type(author_ids) == float:
        continue
    for aid in author_ids:
        dis = []
        for an in author_names:
            dis.append(score(an, aid))
        for i in range(len(dis)):
            if dis[i] == 0:
                cor = author_names[i]
                author_name_map[cor] = aid

import pickle
with open('./pkl/author_name_map.pkl', 'wb') as file:
    pickle.dump(author_name_map, file)

print(author_name_map)

print(len(author_name_map))