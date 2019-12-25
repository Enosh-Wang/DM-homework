import json
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
### author_pub_detail 作者档案信息
# 将whole_author和train_author两个数据集里相同作者的paper合并了
whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')
train_author_paper_ids = pd.read_pickle('./pkl/train_author_paper_ids.pkl')
pub_info = pd.read_pickle('./pkl/pub_info_lda.pkl')

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
    for author_id, paper_ids in tqdm(author_pub_ids_):
        d = []
        for pid in paper_ids:
            d.append(pub_info.loc[pid, pc])
        dat.append(d)
    author_pub_ids[pc] = dat

print(author_pub_ids.head())

len(author_pub_ids[author_pub_ids['author_id'] == 'sCKCrny5']['abstract'].values[0])

author_pub_ids['year'].apply(len).describe()

author_pub_ids.to_pickle('./pkl/author_pub_detail.pkl')