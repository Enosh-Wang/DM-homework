import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc
import time
import os

feat_dir = './feat/'
if not os.path.exists(feat_dir):
    os.mkdir(feat_dir)

train_data = pd.read_pickle('./pkl/train_data.pkl')
test_data = pd.read_pickle('./pkl/valid_data.pkl')

pub_info = pd.read_pickle('./pkl/pub_info.pkl')
author_pub_detail = pd.read_pickle('./pkl/author_pub_detail.pkl')

train_data.head(3)

test_data.head(3)

print(train_data.shape)
print(test_data.shape)

data = pd.concat([train_data, test_data]).reset_index(drop=True)

data.head()

pub_info.columns = ['abstract_a', 'authors_a', 'keywords_a', 'paper_id', 'title_a', 'venue_a', 'year_a', 'orgs_a']
pub_info.head()

author_pub_detail.columns = ['author_id', 'paper_ids', 'paper_ids_len', 'abstract_b', 'keywords_b', 'title_b', 'venue_b', 'year_b', 'authors_b', 'orgs_b']
author_pub_detail.head()

data = data.merge(pub_info, 'left', 'paper_id').merge(author_pub_detail, 'left', 'author_id')

data.shape

data.head(3)

### delete paper_id in pos sample

def pidx(p, ps):
    ans = np.nan
    for i, p2 in enumerate(ps):
        if p == p2:
            ans = i
            break
    return ans


data['idx'] = data.apply(lambda row: pidx(row['paper_id'], row['paper_ids']) if row['label'] == 1 else np.nan, axis=1)

from tqdm import tqdm_notebook

cols = ['abstract_b', 'keywords_b', 'title_b', 'venue_b', 'year_b', 'authors_b', 'orgs_b']
for i in tqdm_notebook(range(len(data))):
    if pd.isna(data.loc[i, 'idx']):
        continue
    for c in cols:
        v = list(data.loc[i, c])
        del v[data.loc[i, 'idx'].astype(int)]
        data.set_value(i, c, v)
#         data.loc[i, c] = frozenset(v)

data.to_pickle('./pkl/data.pkl')

print(data.columns)

### year

# data = pd.read_pickle('./pkl/data.pkl')

data['year_b'] = data['year_b'].apply(lambda x: [0] if len(x) == 0 else x)

data['year_b_min'] = data['year_b'].apply(np.min)
data['year_b_max'] = data['year_b'].apply(np.max)
data['year_b_mean'] = data['year_b'].apply(np.mean)
data['year_b_std'] = data['year_b'].apply(np.std)

data['year_b_mm2'] = (data['year_b_min'] + data['year_b_max']) / 2

for c in ['year_b_min', 'year_b_max', 'year_b_mean', 'year_b_mm2']:
    data[c + '-year_a'] = data[c] - data['year_a']

data['year_inside_range'] = ((data['year_b_min-year_a'] <= 0) & (data['year_b_max-year_a'] >= 0)).astype(int)

cols = ['year_a', 'year_b_min', 'year_b_max', 'year_b_mean', 'year_b_std', 'year_b_mm2',
       'year_b_min-year_a', 'year_b_max-year_a', 'year_b_mean-year_a',
       'year_b_mm2-year_a', 'year_inside_range']

data[cols].to_pickle('./feat/time_feat_a.pkl')

### 'authors', 'orgs'

tmp = data[['author_org', 'authors_a', 'orgs_a', 'authors_b', 'orgs_b']]

# import multiprocessing as mp

# def split_df(df, n):
#     chunk_size = int(np.ceil(len(df) / n))
#     return [df[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

# chunk_list = split_df(tmp, 100)

def func(a, b):
    cnt = 0
    for x in b:
        for y in x:
            if a == y:
                cnt += 1
    return cnt

# def process(df):
#     return df.apply(lambda row: func(row['author_org'], row['orgs_b']), axis=1)
 
# with mp.Pool(8) as pool:
#     ret = pool.map(process, chunk_list)
#     print(len(ret))


tmp['author_org_in_orgs_b_times'] = tmp.apply(lambda row: func(row['author_org'], row['orgs_b']), axis=1)

def funct(a, b):
    b = set([y for x in b for y in x])
    # b = set([x for x in b])
    a = set(a)
    return len(a & b)


tmp['author_interset_num'] = tmp.apply(lambda row: funct(row['authors_a'], row['authors_b']), axis=1)

tmp['author_interset_num/paper_ids_len'] = (tmp['author_interset_num'] / (data['paper_ids_len'] - (data['label'] == 1).astype(int))).fillna(0)

tmp.head()

tmp['author_interset_num'].value_counts()

tmp[['author_org_in_orgs_b_times', 'author_interset_num', 'author_interset_num/paper_ids_len']].to_pickle('./feat/tmp.pkl')