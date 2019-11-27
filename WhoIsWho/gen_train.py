import pandas as pd
import numpy as np
import os
import sys
from collections import OrderedDict
import json
import random
random.seed(42)
np.random.seed(42)

import pickle
# 'Bin Hu':'bin_hu'
with open('./pkl/author_name_map.pkl', 'rb') as file:
    author_name_map = pickle.load(file)
# 'a_mukherjee':{'0vv5Wa0l': '', '1KAG1yh7': 'Univ. of Central Fl..., Orlando', '1gM4SaCv': 'Centre for Advanced...019 India', '254qd3BZ': '', '26LhIoMX': 'Saha Institute of N...ta, India', '3Io5ZqSs': '', '3LzGIsLf': '', '5hGi8NGc': 'Department of Physi...echnology', '6t0h83qe': 'Department of Physi... of Delhi', '71kcLS45': '', '7HwiJ3Mf': 'Saha Institute of N...ata India', '9Arb2zFK': 'Department of Physi...hi, India', 'A5pYtaL0': '', 'BCy31CKN': 'saha institute of n...r physics', ...}
with open('./pkl/author_org_map.pkl', 'rb') as file:
    author_org_map = pickle.load(file)
# ['author_id', 'author_name', 'paper_ids']
whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')
# author_name, author_ids
train_author_name_ids = pd.read_pickle('./pkl/train_author_name_ids.pkl')
# author_name, author_ids, author_nums
valid_data = pd.read_pickle('./pkl/valid_data.pkl')

print('valid: ', len(set(valid_data['author_name'])))
print('train: ', len(set(train_author_name_ids['author_name'])))
print('whole: ', len(set(whole_author_name_paper_ids['author_name'])))
print('-'*60)
# & 集合的交集
print('valid & train: ', len(set(train_author_name_ids['author_name']) & set(valid_data['author_name'])))
print('whole & valid: ', len(set(whole_author_name_paper_ids['author_name']) & set(valid_data['author_name'])))
print('whole & train: ', len(set(whole_author_name_paper_ids['author_name']) & set(train_author_name_ids['author_name'])))

# author_id, paper_ids
train_author_paper_ids = pd.read_pickle('./pkl/train_author_paper_ids.pkl')
# 一个name下对应的id的数目
train_author_name_ids['author_num'] = train_author_name_ids['author_ids'].apply(len)
# 筛选id数目>=2的name
train_author_name_ids = train_author_name_ids[train_author_name_ids['author_num'] >= 2]

print(train_author_name_ids['author_num'].min())

# 把name和id拓展成一一对应了
train_author_name_ids_ext = []
for author_name, author_ids in train_author_name_ids[['author_name', 'author_ids']].values:
     for aid in author_ids:
            train_author_name_ids_ext.append([author_name, aid])
            
train_author_name_ids_ext = pd.DataFrame(train_author_name_ids_ext, columns=['author_name', 'author_id'])
# 同样拓展成一对一的了
train_author_paper_ids_ext = []
for author_id, paper_ids in train_author_paper_ids[['author_id', 'paper_ids']].values:
     for pid in paper_ids:
            train_author_paper_ids_ext.append([author_id, pid])
train_author_paper_ids_ext = pd.DataFrame(train_author_paper_ids_ext, columns=['author_id', 'paper_id'])
# 拓展完之后应当只有paper_id是唯一的
train_author_paper_ids_ext = train_author_paper_ids_ext.merge(train_author_name_ids_ext, 'left', 'author_id')
# 通过name和paper索引到当时的机构
train_author_paper_ids_ext['author_org'] = train_author_paper_ids_ext.apply(lambda row: author_org_map[row['author_name']][row['paper_id']], axis=1)

train_author_name_ids_ext2 = train_author_name_ids_ext.merge(train_author_name_ids[['author_name', 'author_ids']], 'left', 'author_name')


# sample 
n = 5
# 随机采样5个
train_author_name_ids_ext2['author_ids_sample'] = train_author_name_ids_ext2['author_ids'].apply(lambda x: np.random.permutation(x)[:n])
# 采样的值前面并上真实值
train_author_name_ids_ext2['author_ids_sample'] = train_author_name_ids_ext2.apply(lambda row: {row['author_id']} | set(row['author_ids_sample']), axis=1)
print(train_author_name_ids_ext2.head())

train_author_ids_ext2_sample = []
for author_id, author_ids_sample in train_author_name_ids_ext2[['author_id', 'author_ids_sample']].values:
     for aid in author_ids_sample:
            train_author_ids_ext2_sample.append([author_id, aid])
            
train_author_ids_ext2_sample = pd.DataFrame(train_author_ids_ext2_sample, columns=['author_id', 'author_id_sample'])
train_author_ids_ext2_sample.head()

train_data = train_author_paper_ids_ext.merge(train_author_ids_ext2_sample, 'left', 'author_id')

train_data['label'] = (train_data['author_id'] == train_data['author_id_sample']).astype(int)

train_data.drop(columns=['author_id'], inplace=True)

train_data.columns = ['paper_id', 'author_name', 'author_org', 'author_id', 'label']

train_data.to_pickle('./pkl/train_data.pkl')