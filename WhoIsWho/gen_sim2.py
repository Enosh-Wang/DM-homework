import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc
import time
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing

data = pd.read_pickle('./pkl/pub_info.pkl')
data = data.reset_index(drop=True)
stoplist = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','<','>','`','©','``',"''"]
stoplist += punctuations

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
from gensim import corpora,models,similarities

dictionary = corpora.Dictionary(texts)
dictionary.save("all.dict")
print("保存字典")
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('b.mm', corpus)  # 将生成的语料保存成MM文件
print("保存语料")
#corpus = corpora.MmCorpus('ths_corpuse.mm')  # 加载

tfidf = models.TfidfModel(corpus)
tfidf.save("tfidf.model")
print("保存tfidf模型")
corpus_tfidf = tfidf[corpus]
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=100)
lda.save('lda.model')
print("保存lda模型")
'''
index = similarities.Similarity(lda[corpus])
index.save("lda_sim.index")
print("保存sim索引")
'''
data["lda"] = list(lda[corpus])
data.to_pickle('./pkl/pub_info_lda.pkl')