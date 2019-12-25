import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import sys

import pickle

##
whole_pub_info = pd.read_pickle('./pkl/whole_pub_info.pkl')
whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')
## 
train_pub_info = pd.read_pickle('./pkl/train_pub_info.pkl')
train_author_name_ids = pd.read_pickle('./pkl/train_author_name_ids.pkl')
train_author_paper_ids = pd.read_pickle('./pkl/train_author_paper_ids.pkl')
##
valid_pub_info = pd.read_pickle('./pkl/valid_pub_info.pkl')
valid_data = pd.read_pickle('./pkl/valid_data.pkl')

pub_info = pd.concat([whole_pub_info, train_pub_info, valid_pub_info])
print(pub_info.shape)
pub_info = pub_info.drop_duplicates(subset='paper_id', keep='first')
print(pub_info.shape)

pub_info.head(2)

pub_info = pub_info.set_index('paper_id')
pub_info.head(2)

pub_info.loc['0009qJgC', 'authors']

with open('./pkl/author_name_map.pkl', 'rb') as file:
    author_name_map = pickle.load(file)

whole_author_name_paper_ids.head(2)

train_author_paper_ids.head(2)

valid_data.head(2)

# dict: {author_name:{paper_id:org}}

train_author_name_ids = pd.read_pickle('./pkl/train_author_name_ids.pkl')

train_author_name_ids_ext = []
for author_name, author_ids in train_author_name_ids[['author_name', 'author_ids']].values:
     for aid in author_ids:
            train_author_name_ids_ext.append([author_name, aid])
            
train_author_name_ids_ext = pd.DataFrame(train_author_name_ids_ext, columns=['author_name', 'author_id'])
train_author_name_ids_ext.head()

train_author_name_paper_ids = train_author_paper_ids.merge(train_author_name_ids_ext, 'left', 'author_id')
train_author_name_paper_ids.head(2)

author_name_paper_ids = pd.concat([train_author_name_paper_ids, whole_author_name_paper_ids])
author_name_paper_ids.head(2)

# dict: {author_name:{paper_id:org}}
from tqdm import tqdm_notebook
author_org_map = {}
for author_name, paper_ids in tqdm_notebook(author_name_paper_ids[['author_name', 'paper_ids']].values):
    if not author_name in author_org_map:
        author_org_map[author_name] = {}
    for pid in paper_ids:
        org = np.nan
        author_orgs = pub_info.loc[pid, 'authors']
        for ao in author_orgs:
            if not ao['name'] in author_name_map:
                continue
            if author_name_map[ao['name']] == author_name:
                org = ao.get('org')
        author_org_map[author_name][pid] = org

with open('./pkl/author_org_map.pkl', 'wb') as file:
    pickle.dump(author_org_map, file)