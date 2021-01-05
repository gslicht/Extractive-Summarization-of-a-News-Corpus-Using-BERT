# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:31:16 2020

@author: gslicht
"""

import pandas as pd
import numpy as np
import random
from rouge_score import rouge_scorer

def gen_x_y_doc_labels_for_mean_num_sent(df):

    #convert tensors to numpy array
    to_array = lambda x: np.array(x)
    s_embed_text = df.embed_text.apply(to_array).reset_index(drop=True)
    s_y_labels= df.y_labels.apply(to_array).reset_index(drop=True)

    #label docs
    s_doc_label = pd.Series(range(df.shape[0]), name = 'doc_label')

    #calculate doc mean
    s_doc_mean = s_embed_text.apply(lambda x: x.mean(axis=0).reshape(1,-1))


    #create values for each sentence in doc 
    X_doc_label_list =[]
    X_doc_mean_list = []
    X_sent_num_list = []

    for j in range(len(df)):
        X_doc_label = s_doc_label[j]
        X_doc_mean = s_doc_mean[j]
        X_text = s_embed_text [j]
        n = X_text.shape[0]
    
        X_doc_label_fixed = X_doc_label
        X_doc_mean_fixed = X_doc_mean
        sent_num = []
        for i in range(n-1): 
            X_doc_label = np.vstack((X_doc_label, X_doc_label_fixed )) 
            X_doc_mean = np.vstack((X_doc_mean, X_doc_mean_fixed )) 
            sent_num.append(i)
        sent_num.append(n-1)
    
        X_doc_label_list.append(X_doc_label)
        X_doc_mean_list.append(X_doc_mean)
        X_sent_num_list.append(np.array(sent_num).reshape(-1,1))
    
    #from list to pandas series
    s_doc_label = pd.Series(X_doc_label_list)
    s_doc_mean = pd.Series(X_doc_mean_list)
    s_sent_num = pd.Series(X_sent_num_list)

    #concatenate documents with rows = sentences
     #intialize
    Xy_doc_label = s_doc_label.values[0]
    X = np.hstack((s_embed_text[0], s_doc_mean[0], s_sent_num[0]))
    y= s_y_labels[0].reshape(-1,1)
     #recursive population
    for j in range(1, len(df)):
        Xy_doc_label_new = s_doc_label.values[j]
    
        X_text_new = s_embed_text [j]
        X_sent_num_new =s_sent_num[j]
        X_doc_mean_new = s_doc_mean[j]
        y_new = s_y_labels[j].reshape(-1,1)
    
        X_new = np.hstack((X_text_new, X_doc_mean_new, X_sent_num_new))
    
        X = np.vstack((X, X_new))
        y = np.vstack((y, y_new))           
    
        Xy_doc_label = np.vstack((Xy_doc_label, Xy_doc_label_new))
        
    return Xy_doc_label, X, y

def gen_train_test_split_doc_level(Xy_doc_label, X, y, 
                                         test_ratio, folds=1, rand_seed=42):
    

    random.seed(rand_seed)
    total_docs = Xy_doc_label.max()
    train_docs_num = int(total_docs*(1-test_ratio))

    rand_state_list = random.sample(range(2*folds), folds)

    train_test_set = []

    for state in rand_state_list:
    
        random.seed(state)

        train_docs = random.sample(range(1, total_docs+1), train_docs_num)
        train_mask = np.array([x in train_docs for x in list(Xy_doc_label)])
    
        X_train = X[train_mask]
        y_train = y[train_mask]
    
        X_test = X[~train_mask]
        y_test = y[~train_mask]
    
        Xy_doc_label_train = Xy_doc_label[train_mask]
        Xy_doc_label_test = Xy_doc_label[~train_mask]

        data_pass = (Xy_doc_label_train, Xy_doc_label_test,
                                             X_train, X_test, y_train, y_test)
    
        train_test_set.append(data_pass)
    
    train_test_set = tuple(train_test_set)

    return train_test_set


def return_greater_than_min_num(arr, thresh=0.5, min_num=1, fix_num_flag=False, fix_num=3):
    
    if fix_num_flag == True:
        idx = np.argsort(arr)[-fix_num:]
        
    else:
        idx_prelim = np.where(arr>= thresh)
    
        if idx_prelim[0].shape[0] <= min_num:
            idx = np.argsort(arr)[-min_num:]
        else:
            idx = idx_prelim
    
    return sorted(idx)

    
def return_df_pred_summaries( Xy_doc_label, y_pred, df_text, thresh, min_num,
                             return_all=False, fix_num_flag=False, fix_num=3):
    
    df_label_pred = pd.DataFrame({'doc_label': Xy_doc_label.flatten(),
                                                 'y_pred': y_pred.flatten()}) 
    df_label_pred = df_label_pred.groupby('doc_label').agg(list) 

    df_label_pred = df_label_pred.applymap(lambda x: np.array(x))

    f = lambda arr: return_greater_than_min_num(arr, thresh=thresh, 
                                    min_num=min_num,fix_num_flag = fix_num_flag, 
                                                            fix_num=fix_num)

    df_label_pred = df_label_pred.applymap(f) 

    #Return predicted summaryy_
    df_doc = df_text[df_label_pred.index]
    
    
    pred_summaries = [np.array(df_doc.iloc[j])
                               [df_label_pred.iloc[j][0]].tolist()                      #???
                                          for j in range(len(df_label_pred))]

    pred_summaries = [summ_list if type(summ_list) == str else 
                      ' '.join(summ_list) for summ_list in pred_summaries]
    
    if return_all == True:
        answer = df_label_pred.values, df_label_pred.index, pred_summaries
    else:
        answer = pred_summaries
    
    return answer
    

def calc_rouge_scores(pred_summaries, gold_summaries, 
                                 keys=['rouge1', 'rougeL'], use_stemmer=True):
    #Calculate rouge scores
    scorer = rouge_scorer.RougeScorer(keys, use_stemmer= use_stemmer)
    
    n = len(pred_summaries)
    
    scores = [scorer.score(pred_summaries[j], gold_summaries[j]) for 
              j in range(n)] 
    
    dict_scores={}                                                            
    for key in keys:
        dict_scores.update({key: {}})
        
    
    for key in keys:
        
        precision_list = [scores[j][key][0] for j in range(len(scores))]
        recall_list = [scores[j][key][1] for j in range(len(scores))]
        f1_list = [scores[j][key][2] for j in range(len(scores))]

        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1 = np.mean(f1_list)
        
        dict_results = {'recall': recall, 'precision': precision, 'f1': f1}
        
        dict_scores[key] = dict_results
        
    return dict_scores


    
    
    
        
        
        

