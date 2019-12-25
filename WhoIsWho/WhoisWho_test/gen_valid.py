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
# 左连接连接两个表 给待匹配的论文添加详细信息
valid_data = cna_valid_unass.merge(valid_pub_info, 'left', 'paper_id')
# 要匹配的作者序号，转为int
valid_data['author_idx'] = valid_data['author_idx'].astype(int)

# 提取序号指定的作者的名字和工作单位
valid_data['author_name'] = valid_data.apply(lambda row: row['authors'][row['author_idx']]['name'], axis=1)

valid_data['author_org'] = valid_data.apply(lambda row: row['authors'][row['author_idx']].get('org'), axis=1)
# 只用了这三个特征？
valid_data = valid_data[['paper_id', 'author_name', 'author_org']]

# 名称映射
name_dict = {'Tong Zhang 0001':'tong_zhang','JIE LIU':'jie_liu','Sheng Xu':'sheng_xu','Yu JianJun':'jianjun_yu',
'Zhang Xian':'xian_zhang','L Deng':'li_deng','Hideyuki Suzuki':'hideyuki_suzuki','Jie liu':'jie_liu',
'Jian Chen':'jian_chen','Bo LI':'bo_li','Hui Zhang':'hui_zhang','Hong-Zhi Wang':'hongzhi_wang',
'Xia Li':'xia_li','Dr. Hui Xiong':'hui_xiong','Ding Jian-Ping':'jianping_ding','Fusheng Wang':'fusheng_wang',
'DING feng':'feng_ding','Hui Zhang':'hui_zhang',' Zhu Wei-Hong':'weihong_zhu','LI BO':'bo_li','LIU LING':'ling_liu',
'Bo Li 0001':'bo_li','Liu, Ling':'ling_liu','Xu Qing-Song':'qingsong_xu','xu qingsong':'qingsong_xu'
}
author_name_map = pd.read_pickle('./pkl/author_name_map.pkl')
print(author_name_map)
author_name_map.update(name_dict)
name_key = author_name_map.keys()
def convert(name):
    if name not in name_key:
        print(name)
        return ''
    name = author_name_map[name]
    return name

valid_data['author_name'] = valid_data['author_name'].apply(convert)
# ['author_id', 'author_name', 'paper_ids']
whole_author_name_paper_ids = pd.read_pickle('./pkl/whole_author_name_paper_ids.pkl')

#print(whole_author_name_paper_ids.head())

valid_data_ = valid_data.merge(whole_author_name_paper_ids[['author_name', 'author_id']], 'left', 'author_name')
#print(valid_data_.head())
print(len(valid_data_))
print(valid_data_)

print(pd.notnull(valid_data_['author_id']))
valid_data_ = valid_data_[pd.notnull(valid_data_['author_id'])]
valid_data_ = valid_data_.reset_index(drop=True)
print(len(valid_data_))
print(valid_data_)
# [ 'paper_ids', 'author_name', 'author_org', 'author_id']
valid_data_.to_pickle('./pkl/valid_data.pkl')