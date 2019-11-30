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

data = pd.read_pickle('./pkl/data.pkl')
data["words"]=''
data["words_a"]=''

stoplist = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','<','>','`','©','``',"''"]
stoplist += punctuations

def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

df_list = split_df(data,100)
print(df_list)
print(df_list[-1])
exit()
def process(data):
    for i in tqdm(range(len(data))):
        words = []
        for ab in data.loc[i,'abstract_b']:
            words += [word.lower() for word in word_tokenize(ab) if word.lower() not in stoplist] 
        for ti in data.loc[i,'title_b']:
            words += [word.lower() for word in word_tokenize(ti) if word.lower() not in stoplist]
        for key_list in data.loc[i,'keywords_b']:
            if len(key_list) >=1 :
                for key in key_list:
                    words += [word.lower() for word in word_tokenize(key) if word.lower() not in stoplist]
        #data.loc[i,'words'] = words
        data.set_value(i, 'words', words)
        words_a = []
        words_a += [word.lower() for word in word_tokenize(data.loc[i,'abstract_a']) if word.lower() not in stoplist]
        words_a += [word.lower() for word in word_tokenize(data.loc[i,'title_a']) if word.lower() not in stoplist]
        keywords_list_a = data.loc[i,'keywords_a']
        if len(keywords_list_a) >=1 :
            for key in keywords_list_a:
                words_a += [word.lower() for word in word_tokenize(key) if word.lower() not in stoplist]
        #data.loc[i,'words_a'] = words_a
        data.set_value(i, 'words_a', words_a)

cores = multiprocessing.cpu_count()
with multiprocessing.Pool(processes=cores) as pool:
    pool.map(process,df_list)

data[['words','words_a']].to_pickle("./pkl/words.pkl")

'''
texts = []
texts_a = []
for i in tqdm(range(len(data))):
    words = []
    abstract_b = data.loc[i,'abstract_b']
    for ab in abstract_b:
        words += [word.lower() for word in word_tokenize(ab) if word.lower() not in stoplist]
    title_b = data.loc[i,'title_b']
    for ti in title_b:
        words += [word.lower() for word in word_tokenize(ti) if word.lower() not in stoplist]
    keywords_b = data.loc[i,'keywords_b']
    for key_list in keywords_b:
        if len(key_list) >=1 :
            for key in key_list:
                words += [word.lower() for word in word_tokenize(key) if word.lower() not in stoplist]
    texts.append(words)
    # 每行只有一篇论文
    words_a = []
    words_a += [word.lower() for word in word_tokenize(data.loc[i,'abstract_a']) if word.lower() not in stoplist]
    words_a += [word.lower() for word in word_tokenize(data.loc[i,'title_a']) if word.lower() not in stoplist]
    keywords_list_a = data.loc[i,'keywords_a']
    if len(keywords_list_a) >=1 :
        for key in keywords_list_a:
            words_a += [word.lower() for word in word_tokenize(key) if word.lower() not in stoplist]
    texts_a.append(words_a)
'''
texts = list(data['words'])
texts_a = list(data['words_a'])
print("遍历结束")
from gensim import corpora,models,similarities

dictionary = corpora.Dictionary(texts+texts_a)
dictionary.save("all.dict")
print("保存字典")
corpus = [dictionary.doc2bow(text) for text in texts]
corpus_a = [dictionary.doc2bow(text) for text in texts_a]
corpora.MmCorpus.serialize('b.mm', corpus)  # 将生成的语料保存成MM文件
corpora.MmCorpus.serialize('a.mm', corpus_a)
print("保存语料")
#corpus = corpora.MmCorpus('ths_corpuse.mm')  # 加载

tfidf = models.TfidfModel(corpus)
tfidf.save("tfidf.model")
print("保存tfidf模型")
corpus_tfidf = tfidf[corpus]
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=100)
lda.save('lda.model')
print("保存lda模型")
index = similarities.Similarity(lda[corpus])
index.save("lda_sim.index")
print("保存sim索引")
