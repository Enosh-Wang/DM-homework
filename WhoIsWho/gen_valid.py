import pandas as pd
import numpy as np
import os
import sys
from collections import OrderedDict
import json

### valid data
# paper_id, author_names&orgs, title, venue, year, keywords, abstract
valid_pub_info = pd.read_pickle('./pkl/valid_pub_info.pkl')
cna_valid_unass = pd.read_pickle('./pkl/cna_valid_unass.pkl')
# 左连接连接两个表
valid_data = cna_valid_unass.merge(valid_pub_info, 'left', 'paper_id')
# 要匹配的作者序号，转为int
valid_data['author_idx'] = valid_data['author_idx'].astype(int)

# 提取序号指定的作者的名字和工作单位
valid_data['author_name'] = valid_data.apply(lambda row: row['authors'][row['author_idx']]['name'], axis=1)

valid_data['author_org'] = valid_data.apply(lambda row: row['authors'][row['author_idx']].get('org'), axis=1)
# 只用了这三个特征？
valid_data = valid_data[['paper_id', 'author_name', 'author_org']]
# 将作者姓名转为小写，去掉特殊符号
# 特殊处理的那两个不知道干嘛的
def convert(name):
    name = name.lower()
    name = name.replace('. ', '_').replace('.', '_').replace(' ', '_').replace('-', '')
    if name in ['yang_jie', 'jie_yang_0002', 'jie\xa0yang', 'jie_yang_0008']:
        name = 'jie_yang'
    if name in ['liu_bing']:
        name = 'bing_liu'
    return name

valid_data['author_name'] = valid_data['author_name'].apply(convert)

print(valid_data.head())

print(valid_data['author_name'].nunique())
# ['author_id', 'author_name', 'paper_ids']
whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')

print(whole_author_name_paper_ids.head())

valid_data_ = valid_data.merge(whole_author_name_paper_ids[['author_name', 'author_id']], 'left', 'author_name')

print(valid_data_.head())

valid_data_.to_pickle('./pkl/valid_data.pkl')