import warnings
warnings.filterwarnings('ignore')

import os
os.environ["CUDA_VISIBLE_DEVICES"] ='2'

import pandas as pd
import numpy as np
import gc
import time

out_dir = './out/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier, Pool

train_data = pd.read_pickle('./pkl/train_data.pkl')
test_data = pd.read_pickle('./pkl/valid_data.pkl')

data = pd.concat([train_data, test_data]).reset_index(drop=True)

data.head()

print(train_data.shape)
print(test_data.shape)
print(data.shape)

##
time_feat_a = pd.read_pickle('./feat/time_feat_a.pkl')
tmp = pd.read_pickle('./feat/tmp.pkl')

feats = [time_feat_a, tmp]

data = pd.concat([data] + feats, axis=1)

data.head()

drop_feat = ['author_id', 'author_name', 'author_org', 'paper_id', 'label']
used_feat = [c for c in data.columns if c not in drop_feat]
print(len(used_feat))
print(used_feat)

train = data[:len(train_data)]
test = data[len(train_data):]

test_x = test[used_feat]

# cv split according to author names
train_author_name = train['author_name'].unique()
print(len(train_author_name))

def gen_dict(df, label):
    df = df[['paper_id', 'author_name', 'author_id', label]]
    res = df.groupby(['paper_id', 'author_name'])[label].apply(np.argmax).reset_index()
    res.columns = ['paper_id', 'author_name', 'index']
    idx_name = df[['author_id']].reset_index()
    res = res.merge(idx_name, 'left', 'index')
    from collections import defaultdict
    res_dict = defaultdict(list)
    for pid, aid in res[['paper_id', 'author_id']].values:
        res_dict[aid].append(pid)
    return res_dict

def f1_score(pred_dict, true_dict):
    total_unassigned_paper = np.sum([len(l) for l in true_dict.values()])
    print('total_unassigned_paper: ', total_unassigned_paper)
    print('true author num: ', len(true_dict))
    author_weight = dict((k, len(v) / total_unassigned_paper) for k, v in true_dict.items())
    author_precision = {}
    author_recall = {}
    for author in author_weight.keys():
        # total pred, total belong, correct pred
        total_belong = len(true_dict[author])
        total_pred = (len(pred_dict[author]) if author in pred_dict else 0)
        correct_pred = len(set(true_dict[author]) & (set(pred_dict[author]) if author in pred_dict else set()))
        author_precision[author] = (correct_pred/total_pred) if total_pred > 0 else 0
        author_recall[author] = correct_pred / total_belong
        
    weighted_precision = 0
    weighted_recall = 0
    for author, weight in author_weight.items():
        weighted_precision += weight * author_precision[author]
        weighted_recall += weight * author_recall[author]
    weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
    print('weighted_precision: %f, weighted_recall: %f, weighted_f1: %f' %(weighted_precision, weighted_recall, weighted_f1))
    return weighted_precision, weighted_recall, weighted_f1

from sklearn.model_selection import KFold

preds = np.zeros((test.shape[0], 2))
scores = []
f1_scores = []
has_saved = False
imp = pd.DataFrame()
imp['feat'] = used_feat

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for index, (tr_idx, va_idx) in enumerate(kfold.split(train_author_name)):
    print('*' * 30)
    trn_aname, val_aname = train_author_name[tr_idx], train_author_name[va_idx]
    trn_dat = train[train['author_name'].isin(trn_aname)]
    val_dat = train[train['author_name'].isin(val_aname)]
    X_train, y_train, X_valid, y_valid = trn_dat[used_feat], trn_dat['label'], val_dat[used_feat], val_dat['label']
    cate_features = []
    train_pool = Pool(X_train, y_train, cat_features=cate_features)
    eval_pool = Pool(X_valid, y_valid,cat_features=cate_features)
    if not has_saved: 
        cbt_model = CatBoostClassifier(iterations=10000,
                           learning_rate=0.1,
                           eval_metric='AUC',
                           use_best_model=True,
                           random_seed=42,
                           logging_level='Verbose',
                           task_type='GPU',
                           devices='0',
                           gpu_ram_part=0.5,
                           early_stopping_rounds=300,
                           loss_function='Logloss',
#                            depth=12,
                           )
        cbt_model.fit(train_pool, eval_set=eval_pool, verbose=100)
#         with open('./models/fold%d_cbt_v1.mdl' % index, 'wb') as file:
#             pickle.dump(cbt_model, file)
#     else:
#         with open('./models/fold%d_cbt_v1.mdl' % index, 'rb') as file:
#             cbt_model = pickle.load(file)
    
    imp['score%d' % (index+1)] = cbt_model.feature_importances_
    
    val_dat['pred'] = cbt_model.predict_proba(X_valid)[:, 1]
    val_pred_dict = gen_dict(val_dat, 'pred')
    val_true_dict = gen_dict(val_dat, 'label')
    precision, recall, f1 = f1_score(val_pred_dict, val_true_dict)
    f1_scores.append(f1)
    
    score = cbt_model.best_score_['validation']['AUC']
    scores.append(score)
    print('fold %d round %d : auc: %.6f | mean auc %.6f | F1: %.6f | mean F1: %.6f' % (index+1, cbt_model.best_iteration_, score,np.mean(scores), f1, np.mean(f1_scores))) 
    preds += cbt_model.predict_proba(test_x)  
#     break
    del cbt_model, train_pool, eval_pool
    del X_train, y_train, X_valid, y_valid
    import gc
    gc.collect()
    
#     mdls.append(cbt_model)

imp.sort_values(by='score1', ascending=False)

test_data['pred'] = preds[:, 1]

test_data.head()

result_dict = gen_dict(test_data, 'pred')

len(result_dict)

import json
import time
localtime = time.localtime(time.time())
save_path = './out/result_%02d%02d%02d%02d.json' % (localtime[1], localtime[2], localtime[3], localtime[4])
with open(save_path, 'w') as file:
    file.write(json.dumps(result_dict))