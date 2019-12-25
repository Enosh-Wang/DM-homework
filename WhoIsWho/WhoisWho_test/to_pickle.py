import json
import numpy as np
import pandas as pd
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

import gc
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora,models
import time

with open('data/cna_data/cna_test_pub.json') as file:
    cna_valid_pub = json.load(file, object_pairs_hook=OrderedDict)
with open('data/cna_data/cna_test_unass_competition.json') as file:
    cna_valid_unass = json.load(file, object_pairs_hook=OrderedDict)

### cna_valid_unass 待分配的论文

cna_valid_unass = pd.DataFrame(cna_valid_unass, columns=['cna_valid_unass'])

cna_valid_unass['cna_valid_unass'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x.split('-'))

cna_valid_unass['paper_id'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x[0])
cna_valid_unass['author_idx'] = cna_valid_unass['cna_valid_unass'].apply(lambda x: x[1])

del cna_valid_unass['cna_valid_unass']

cna_valid_unass.to_pickle('./pkl/cna_valid_unass.pkl')

### cna_valid_pub 验证集的论文元信息

# paper_id, author_names&orgs, title, venue, year, keywords, abstract
valid_pub_info = pd.DataFrame.from_dict(cna_valid_pub, orient='index').reset_index(drop=True).rename({'id':'paper_id'}, axis=1)

valid_pub_info.head()

valid_pub_info.to_pickle('./pkl/valid_pub_info.pkl')

### pub info 论文元信息 把三部分论文元信息拼接去重后规范了一下数据格式
# 删除重复的paper_id，保留第一次出现的行

whole_pub_info = pd.read_pickle('./pkl/whole_pub_info.pkl')
train_pub_info = pd.read_pickle('./pkl/train_pub_info.pkl')

pub_info = pd.concat([whole_pub_info, train_pub_info, valid_pub_info]).drop_duplicates(subset='paper_id', keep='first')

pub_info['orgs'] = pub_info['authors'].apply(lambda x: [ao['org'] for ao in x if 'org' in ao])
pub_info['authors'] = pub_info['authors'].apply(lambda x: [ao['name'] for ao in x if 'name' in ao])

pub_info['year'] = pub_info['year'].fillna(0).replace('', 0).astype(int)

pub_info['abstract'] = pub_info['abstract'].fillna(' ').replace('', ' ')
pub_info['keywords'] = pub_info['keywords'].fillna(' ').replace('', ' ')
pub_info['venue'] = pub_info['venue'].fillna(' ').replace('', ' ')
pub_info['title'] = pub_info['title'].fillna(' ').replace('', ' ')
print(pub_info.head())

data = pub_info.reset_index(drop=True)

'''
# 停用词
stoplist = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','<','>','`','©','``',"''"]
stoplist += punctuations

# 词袋
texts = []
for i in tqdm(range(len(data))):
    words = []
    words += [word.lower() for word in word_tokenize(data.loc[i,'abstract']) if word.lower() not in stoplist]
    words += [word.lower() for word in word_tokenize(data.loc[i,'title']) if word.lower() not in stoplist]
    keywords_list = data.loc[i,'keywords']
    if len(keywords_list) >=1 :
        for key in keywords_list:
            words += [word.lower() for word in word_tokenize(key) if word.lower() not in stoplist]
    texts.append(words)

print("遍历结束")
'''
dictionary = corpora.Dictionary.load("all.dict")
print("加载字典")
#corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.serialize('test.mm', corpus)  # 将生成的语料保存成MM文件

corpus = corpora.MmCorpus('test.mm')  # 加载
print("加载语料")

# 加载tf-idf模型
tfidf = models.TfidfModel.load("tfidf.model")
corpus_tfidf = tfidf[corpus]
print("加载tf-idf模型")
# 训练lsi模型
lsi = models.LsiModel.load("lsi20.model")
print("加载lsi模型")

data["lsi"] = list(lsi[corpus_tfidf])
print(data.head())
data.to_pickle('./pkl/pub_info.pkl')



#lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
#lsi.save('lsi20.model')