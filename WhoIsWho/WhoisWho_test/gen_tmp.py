import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc
import time
import os

start_time = time.time()
data = pd.read_pickle('./pkl/data.pkl')
tmp = data[['author_org', 'authors_a', 'orgs_a','lsi_a', 'keywords_a', 'venue_a', 'venue_b', 'keywords_b', 'authors_b', 'orgs_b', 'lsi_b']]
tmp = tmp.fillna(' ').replace('', ' ')

print("读取:%d" % (time.time() - start_time))

# 机构
def fun_org(a, b):
    cnt = 0
    if a != '' and a != ' ' and b != '':
        for x in b:
            for y in x:
                if a == y:
                    cnt += 1
    return cnt
tmp['author_org_in_orgs_b_times'] = tmp.apply(lambda row: fun_org(row['author_org'], row['orgs_b']), axis=1)
tmp['author_org_in_orgs_b_times/paper_ids_len'] = (tmp['author_org_in_orgs_b_times'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)

print("机构:%d" % (time.time() - start_time))

# 机构，重合个数
def fun_orgs_set(a,b):
    a = set(a) 
    b = set([y for x in b for y in x])
    if len(b) >= 1:
        b.discard(" ")
        b.discard("")
    return len(a&b)

tmp['co_orgs'] = tmp.apply(lambda row: fun_orgs_set(row['orgs_a'], row['orgs_b']), axis=1)
print("机构，重合个数:%d" % (time.time() - start_time))

# 论文发表期刊在资料库中出现的次数
def fun_venue(a,b):
    cnt = 0
    if a != '' and a != ' ':
        for x in b:
            if a == x :
                cnt += 1
    return cnt
tmp['co_venue_times'] = tmp.apply(lambda row: fun_venue(row['venue_a'], row['venue_b']), axis=1)
tmp['co_venue_times/paper_ids_len'] = (tmp['co_venue_times'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)
print("co_venue_times:%d" % (time.time() - start_time))

# keywords 重合个数
def fun_keywords(a,b):
    a = set(a) 
    b = set([y for x in b for y in x])
    if len(b) >= 1:
        b.discard(" ")
        b.discard("")
    return len(a&b)
tmp['co_keywords'] = tmp.apply(lambda row: fun_keywords(row['keywords_a'], row['keywords_b']), axis=1)
print("co_keywords:%d" % (time.time() - start_time))

# keywords 出现个数
def fun_keywords_cnt(a,b,c):
    cnt = 0
    if c != 0:
        for key in a:
            if key != ' ' and key != '':
                for paper in b:
                    for key_b in paper:
                        if key == key_b:
                            cnt += 1
    return cnt

tmp['co_keywords_times'] = tmp.apply(lambda row: fun_keywords_cnt(row['keywords_a'], row['keywords_b'],row['co_keywords']), axis=1)
tmp['co_keywords_times/paper_ids_len'] = (tmp['co_keywords_times'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)

print("co_keywords_times:%d" % (time.time() - start_time))

# co-author 信息，共同作者重合数
def funct(a, b):
    a = set(a) 
    b = set([y for x in b for y in x])
    if len(b) >= 1:
        b.discard(" ")
        b.discard("")
    return len(a & b)
tmp['author_interset_num'] = tmp.apply(lambda row: funct(row['authors_a'], row['authors_b']), axis=1)
tmp['author_interset_num/paper_ids_len'] = (tmp['author_interset_num'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)

print("author_interset_num:%d" % (time.time() - start_time))

# sim

def bit_product_sum(x, y):
    if len(x) > 1 and len(y) > 1 :
        return sum([item[0][1] * item[1][1] for item in zip(x, y)])
    else :
        return 0

def fun_sim(a,b):
    sim = 0
    for paper in b:
        sim += bit_product_sum(a, paper) / (np.sqrt(bit_product_sum(a, paper)) * np.sqrt(bit_product_sum(a, paper)))
    return sim

tmp['sim'] = tmp.apply(lambda row: fun_sim(row['lsi_a'], row['lsi_b']), axis=1)
tmp['sim/paper_ids_len'] = (tmp['sim'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)
print("sim:%d" % (time.time() - start_time))

tmp[['author_org_in_orgs_b_times', 'author_org_in_orgs_b_times/paper_ids_len',
'co_orgs','co_venue_times','co_venue_times/paper_ids_len','co_keywords',
'co_keywords_times','co_keywords_times/paper_ids_len',
'author_interset_num', 'author_interset_num/paper_ids_len',
'sim','sim/paper_ids_len']].to_pickle('./feat/tmp.pkl')

