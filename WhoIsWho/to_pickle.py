import json
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

pkl_path = './pkl'
if not os.path.exists(pkl_path):
    os.mkdir(pkl_path)

with open('data/cna_data/cna_valid_pub.json') as file:
    cna_valid_pub = json.load(file, object_pairs_hook=OrderedDict)
with open('data/cna_data/cna_valid_unass_competition.json') as file:
    cna_valid_unass = json.load(file, object_pairs_hook=OrderedDict)
with open('data/cna_data/whole_author_profile.json') as file:
    whole_author_profile = json.load(file, object_pairs_hook=OrderedDict)
with open('data/cna_data/whole_author_profile_pub.json') as file:
    whole_author_profile_pub = json.load(file, object_pairs_hook=OrderedDict)

with open('data/train/train_pub.json') as file:
    train_pub = json.load(file, object_pairs_hook=OrderedDict)
with open('data/train/train_author.json') as file:
    train_author = json.load(file, object_pairs_hook=OrderedDict)

### train_author

# 1. author_name, author_ids
train_author_names = list(train_author.keys())
train_author_ids = [list(v.keys()) for v in train_author.values()]
train_author_name_ids = pd.DataFrame(list(zip(train_author_names, train_author_ids)), columns=['author_name', 'author_ids'])

print(train_author_name_ids.head())

train_author_name_ids.to_pickle('./pkl/train_author_name_ids.pkl')

# 2. author_id, paper_ids
train_author_paper_ids = pd.DataFrame([(k2, v2) for v1 in train_author.values() for k2, v2 in v1.items()], columns=['author_id', 'paper_ids'])

print(train_author_paper_ids.head())

train_author_paper_ids.to_pickle('./pkl/train_author_paper_ids.pkl')

### train_pub

# paper_id, author_names&orgs, title, venue, year, keywords, abstract
train_pub_info = pd.DataFrame.from_dict(train_pub, orient='index').reset_index(drop=True).rename({'id':'paper_id'}, axis=1)

print(train_pub_info.head())

train_pub_info.to_pickle('./pkl/train_pub_info.pkl')

### whole_author_profile

whole_author_name_paper_ids = pd.DataFrame.from_dict(whole_author_profile, orient='index').reset_index()
whole_author_name_paper_ids.columns = ['author_id', 'author_name', 'paper_ids']

print(whole_author_name_paper_ids.head())

whole_author_name_paper_ids.to_pickle('./pkl/whole_author_name_paper_ids.pkl')

### whole_author_profile_pub

# paper_id, author_names&orgs, title, venue, year, keywords, abstract
whole_pub_info = pd.DataFrame.from_dict(whole_author_profile_pub, orient='index').reset_index(drop=True).rename({'id':'paper_id'}, axis=1)

print(whole_pub_info.head())

whole_pub_info.to_pickle('./pkl/whole_pub_info.pkl')

### cna_valid_unass

cna_valid_unass = pd.DataFrame(cna_valid_unass, columns=['cna_valid_unass'])

cna_valid_unass['cna_valid_unass'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x.split('-'))

cna_valid_unass['paper_id'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x[0])
cna_valid_unass['author_idx'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x[1])

del cna_valid_unass['cna_valid_unass']

print(cna_valid_unass.head())

cna_valid_unass.to_pickle('./pkl/cna_valid_unass.pkl')

### cna_valid_pub

# paper_id, author_names&orgs, title, venue, year, keywords, abstract
valid_pub_info = pd.DataFrame.from_dict(cna_valid_pub, orient='index').reset_index(drop=True).rename({'id':'paper_id'}, axis=1)

print(valid_pub_info.head())

valid_pub_info.to_pickle('./pkl/valid_pub_info.pkl')

### pub info 论文元信息 把三部分论文元信息拼接去重后规范了一下数据格式
# 删除重复的paper_id，保留第一次出现的行
pub_info = pd.concat([whole_pub_info, train_pub_info, valid_pub_info]).drop_duplicates(subset='paper_id', keep='first')

pub_info['orgs'] = pub_info['authors'].apply(lambda x: [ao['org'] for ao in x if 'org' in ao])
pub_info['authors'] = pub_info['authors'].apply(lambda x: [ao['name'] for ao in x if 'name' in ao])

pub_info['year'] = pub_info['year'].fillna(0).replace('', 0).astype(int)

pub_info['abstract'] = pub_info['abstract'].fillna(' ').replace('', ' ')
pub_info['keywords'] = pub_info['keywords'].fillna(' ').replace('', ' ')
pub_info['venue'] = pub_info['venue'].fillna(' ').replace('', ' ')
pub_info['title'] = pub_info['title'].fillna(' ').replace('', ' ')
print(pub_info.head())

pub_info.to_pickle('./pkl/pub_info.pkl')

### author_pub_detail
# 将whole_author和train_author两个数据集里相同作者的paper合并了
author_pub_ids = whole_author_name_paper_ids[['author_id','paper_ids']].merge(train_author_paper_ids, 'left', 'author_id')

author_pub_ids['paper_ids_x_len'] = author_pub_ids['paper_ids_x'].apply(len)
author_pub_ids['paper_ids_y_len'] = author_pub_ids['paper_ids_y'].apply(lambda x: 0 if type(x) == float else len(x))

author_pub_ids['paper_ids'] = author_pub_ids.apply(lambda row: list(set(row['paper_ids_x']) | (set() if type(row['paper_ids_y']) == float else set(row['paper_ids_y']))), axis=1)

author_pub_ids['paper_ids_len'] = author_pub_ids['paper_ids'].apply(len)

author_pub_ids.drop(columns=['paper_ids_x', 'paper_ids_y', 'paper_ids_x_len', 'paper_ids_y_len'], inplace=True)

print(author_pub_ids.head())

author_pub_ids['paper_ids_len'].describe()

pub_info = pub_info.set_index('paper_id')

print(pub_info.head())
# 把pub_info里的信息添加进去，最终包含所有数据集里作者对应的论文和论文元数据，一个作者一行
# author_id paper_ids paper_ids_len abstracts keywords titles venues years authors orgs



author_pub_ids_ = author_pub_ids[['author_id', 'paper_ids']].values
pub_col = ['abstract', 'keywords', 'title', 'venue', 'year', 'authors', 'orgs']
for pc in pub_col:
    print(pc)
    dat = []
    for author_id, paper_ids in tqdm_notebook(author_pub_ids_):
        d = []
        for pid in paper_ids:
            d.append(pub_info.loc[pid, pc])
        dat.append(d)
    author_pub_ids[pc] = dat

print(author_pub_ids.head())

len(author_pub_ids[author_pub_ids['author_id'] == 'sCKCrny5']['abstract'].values[0])

author_pub_ids['year'].apply(len).describe()

author_pub_ids.to_pickle('./pkl/author_pub_detail.pkl')