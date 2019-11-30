import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc
import time
import os

### 'authors', 'orgs'
data = pd.read_pickle('./pkl/data.pkl')
tmp = data[['author_org', 'authors_a', 'orgs_a', 'keywords_a', 'venue_a', 'venue_b', 'keywords_b', 'authors_b', 'orgs_b']]

def fun_org(a, b):
    cnt = 0
    for x in b:
        for y in x:
            if a == y:
                cnt += 1
    return cnt
tmp['author_org_in_orgs_b_times'] = tmp.apply(lambda row: fun_org(row['author_org'], row['orgs_b']), axis=1)

# 待匹配论文的机构在资料库中出现的次数，论文发表期刊在资料库中出现的次数

def fun_venue(a,b):
    cnt = 0
    for x in b:
        if a == x and a != ' ':
            cnt += 1
    return cnt
tmp['co_venue_times'] = tmp.apply(lambda row: fun_venue(row['venue_a'], row['venue_b']), axis=1)
tmp['co_venue_times/paper_ids_len'] = (tmp['co_venue_times'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)

def fun_keywords(a,b):
    a = set(a) 
    b = set([y for x in b for y in x])
    if len(b) >= 1:
        b.discard(" ")
    return len(a&b)

tmp['co_keywords_times'] = tmp.apply(lambda row: fun_keywords(row['keywords_a'], row['keywords_b']), axis=1)

def funct(a, b):
    b = set([y for x in b for y in x])
    a = set(a)
    return len(a & b)

# co-author 信息，共同作者重合数
tmp['author_interset_num'] = tmp.apply(lambda row: funct(row['authors_a'], row['authors_b']), axis=1)

tmp['author_interset_num/paper_ids_len'] = (tmp['author_interset_num'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)

tmp[['author_org_in_orgs_b_times', 'author_interset_num', 'author_interset_num/paper_ids_len','co_venue_times','co_venue_times/paper_ids_len','co_keywords_times']].to_pickle('./feat/tmp.pkl')