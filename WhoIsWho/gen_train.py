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
with open('./pkl/author_name_map.pkl', 'rb') as file:
    author_name_map = pickle.load(file)
with open('./pkl/author_org_map.pkl', 'rb') as file:
    author_org_map = pickle.load(file)

whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')
train_author_name_ids = pd.read_pickle('./pkl/train_author_name_ids.pkl')
valid_data = pd.read_pickle('./pkl/valid_data.pkl')

print('valid: ', len(set(valid_data['author_name'])))
print('train: ', len(set(train_author_name_ids['author_name'])))
print('whole: ', len(set(whole_author_name_paper_ids['author_name'])))
print('-'*60)
print('valid & train: ', len(set(train_author_name_ids['author_name']) & set(valid_data['author_name'])))
print('whole & valid: ', len(set(whole_author_name_paper_ids['author_name']) & set(valid_data['author_name'])))
print('whole & train: ', len(set(whole_author_name_paper_ids['author_name']) & set(train_author_name_ids['author_name'])))

train_author_paper_ids = pd.read_pickle('./pkl/train_author_paper_ids.pkl')

whole_author_name_paper_ids

train_author_name_ids

train_author_paper_ids

train_author_name_ids['author_num'] = train_author_name_ids['author_ids'].apply(len)

train_author_name_ids = train_author_name_ids[train_author_name_ids['author_num'] >= 2]

print(train_author_name_ids['author_num'].min())

train_author_name_ids

train_author_name_ids_ext = []
for author_name, author_ids in train_author_name_ids[['author_name', 'author_ids']].values:
     for aid in author_ids:
            train_author_name_ids_ext.append([author_name, aid])
            
train_author_name_ids_ext = pd.DataFrame(train_author_name_ids_ext, columns=['author_name', 'author_id'])
train_author_name_ids_ext.head()

train_author_paper_ids_ext = []
for author_id, paper_ids in train_author_paper_ids[['author_id', 'paper_ids']].values:
     for pid in paper_ids:
            train_author_paper_ids_ext.append([author_id, pid])
train_author_paper_ids_ext = pd.DataFrame(train_author_paper_ids_ext, columns=['author_id', 'paper_id'])

# train_pub_info = pd.read_pickle('./pkl/train_pub_info.pkl')[['paper_id', 'authors']]

# train_author_paper_ids_ext = train_author_paper_ids_ext.merge(train_pub_info, 'left', 'paper_id')

train_author_paper_ids_ext = train_author_paper_ids_ext.merge(train_author_name_ids_ext, 'left', 'author_id')

train_author_paper_ids_ext.head()


train_author_paper_ids_ext['author_org'] = train_author_paper_ids_ext.apply(lambda row: author_org_map[row['author_name']][row['paper_id']], axis=1)

train_author_paper_ids_ext['author_org'].nunique() / len(train_author_paper_ids_ext)

train_author_paper_ids_ext.head()

train_author_name_ids_ext.head()

train_author_name_ids_ext2 = train_author_name_ids_ext.merge(train_author_name_ids[['author_name', 'author_ids']], 'left', 'author_name')

train_author_name_ids_ext2.head()

# sample 
n = 5
train_author_name_ids_ext2['author_ids_sample'] = train_author_name_ids_ext2['author_ids'].apply(lambda x: np.random.permutation(x)[:n])

train_author_name_ids_ext2['author_ids_sample'] = train_author_name_ids_ext2.apply(lambda row: {row['author_id']} | set(row['author_ids_sample']), axis=1)
train_author_name_ids_ext2.head()

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

train_data['label'].value_counts()

train_data.head()

train_data.to_pickle('./pkl/train_data.pkl')