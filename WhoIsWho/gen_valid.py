import pandas as pd
import numpy as np
import os
import sys
from collections import OrderedDict
import json

### valid data

valid_pub_info = pd.read_pickle('./pkl/valid_pub_info.pkl')
cna_valid_unass = pd.read_pickle('./pkl/cna_valid_unass.pkl')

valid_data = cna_valid_unass.merge(valid_pub_info, 'left', 'paper_id')

valid_data['author_idx'] = valid_data['author_idx'].astype(int)
print(valid_data['author_idx'].max())

valid_data['author_name'] = valid_data.apply(lambda row: row['authors'][row['author_idx']]['name'], axis=1)

valid_data['author_org'] = valid_data.apply(lambda row: row['authors'][row['author_idx']].get('org'), axis=1)

valid_data = valid_data[['paper_id', 'author_name', 'author_org']]

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

valid_data['author_name'].nunique()

valid_data[valid_data['author_org'] == 'South China University of Technology']



whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')

print(whole_author_name_paper_ids.head())

valid_data_ = valid_data.merge(whole_author_name_paper_ids[['author_name', 'author_id']], 'left', 'author_name')

valid_data_.isnull().sum() / len(valid_data_)

print(valid_data_.head())

valid_data_.to_pickle('./pkl/valid_data.pkl')