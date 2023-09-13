import json
import os, sys
import numpy as np
import math
from collections import Counter

import random
# random.seed(312)

UNKNOWN = [0]
PADDING = [0]
LABELPAD = 7

embd_name = ['embd', 'embd_a', 'embd_b', 'embd_c'] 

label_map = {'Introduction': 0, 
             'Thesis': 1, 
             'Main Idea': 2, 
             'Evidence': 3,   
             'Conclusion': 4, 
             'Elaboration': 5,
             'Other': 6, 
             'padding': 7}
             
para_label_map = {'IntroductionPara': 0, 
                 'ThesisPara': 1, 
                 'IdeaPara': 2, 
                 'SupportPara': 3,   
                 'ConclusionPara': 4, 
                 'OtherPara': 5, 'transition': 5,
                 'padding': 7, 'noise': 7}

score_map = {'Bad': 0, 'Medium': 1, 'Great': 2}                     
    
def loadDataAndFeature(in_file, title=False, max_len=99):
    labels = []
    documents = []
    ft_list = ['gid', 'lid', 'pid']
    features = []
    scores = []
    grids = []
    paralabs = []
    with open(in_file, 'r', encoding='utf8') as f:
        for line in f:
            ft = []
            load_dict = json.loads(line)
            if title:
                if ('slen' in load_dict) and ('slen' not in ft_list):
                    ft_list.append('slen')
                    
                load_dict['sents'].insert(0, load_dict['title'])
                load_dict['labels'].insert(0, 'padding')
                
                for k in ft_list:
                    load_dict[k].insert(0, 0)
                    
                if 'slen' in load_dict:    
                    load_dict['slen'][0] = len(load_dict['title'])

            if 'slen' not in load_dict:
                load_dict['slen'] = [len(sent) for sent in load_dict['sents']]
            documents.append(load_dict['sents'][: max_len+title])
            labels.append(load_dict['labels'][: max_len+title])
            paralab = load_dict['paras'] + ['padding']*(20-len(load_dict['paras']))
            paralabs.append(paralab[:20])
            
            grid = []
            # para = []
            for i in load_dict['gid']:
                if i > max_len:
                    break
                ft.append([load_dict[k][i-1+title] for k in ft_list])
                
                if i == 0:
                    continue
                if load_dict['lid'][i] == 1:
                    if i != 1:
                        para = para + [0]*(20-len(para))
                        if len(para) > 20:
                            para = para[:20]
                        # if len(para) > 20:
                            # print(para)
                        grid.append(para)
                    para = []
                slen = max(1, round(load_dict['slen'][i]/20))
                para.extend([load_dict['gid'][i]]*slen)      
            if len(para) > 20:
                para = para[:20]    
            grid.append(para + [0]*(20-len(para)))
            
            if len(grid) < 20:
                grid += [[0]*20] * (20-len(grid))
            elif len(grid) > 20:
                grid = grid[:20]
                    
            features.append(ft)
            grids.append(grid)
            
            scores.append(score_map[load_dict['score']])

    return documents, labels, features, scores, grids, paralabs

    
def labelEncode(labels):
    en_labels = []
    for labs in labels:
        en_labs = []
        for label in labs:
            # if not label in label_map:
                # print(label)
                # continue
            en_labs.append(label_map[label])
        en_labels.append(en_labs)
    return en_labels

def paraEncode(paralabs):
    en_paralabels = []
    for labs in paralabs:
        en_labs = []
        for label in labs:
            # if not label in label_map:
                # print(label)
                # continue
            en_labs.append(para_label_map[label])
        en_paralabels.append(en_labs)
    return en_paralabels
    
def scoreEncode(scores):
    en_scores = []
    for score in scores:
        
        en_scores.append(score/2)
    return en_scores
    
def encode(documents, labels, embed_map, vec_size):
    names = []
    with open('./data/names.txt', 'r', encoding='utf8') as f:
        for line in f:
            names.append(line[:-1])
        
    en_documents = []
    for sentences in documents:
        length = len(sentences)
        out_sentences = []
        for sentence in sentences:
            assert names[0] in embed_map
            # sentence = [w if w not in names else names[0] for w in sentence]
            seq = [embed_map[w] if w in embed_map else UNKNOWN * vec_size for w in sentence]
            out_sentences.append(seq)
        en_documents.append(out_sentences)
        
    en_labels = labelEncode(labels)
    
    return en_documents, en_labels
    
def sentence_padding(en_documents, labels, n_l, vec_size, is_cutoff=True):
    pad_documents = []
    for sentences in en_documents:
        length = len(sentences)
        out_sentences = []
        for sentence in sentences:
            if len(sentence) % n_l:
                sentence = sentence + [PADDING * vec_size] * (n_l - len(sentence) % n_l)
            if is_cutoff:
                out_sentences.append(sentence[0: n_l])
            else:
                for i in range(0, len(sentence), n_l):
                    out_sentences.append(sentence[i: i+n_l])
                    # 还需要label填充
        pad_documents.append(out_sentences)
    pad_labels = labels
    return pad_documents, pad_labels
    
def loadEmbeddings(embed_filename):
    embed_map = {}
    with open(embed_filename, 'r', encoding='utf-8') as f:
        lenth = f.readline()
        for line in f:
            line = line[:-1]
            embed = line.split(' ')
            vec_size = len(embed[1:])
            embed_map[embed[0]] = [float(x) for x in embed[1:]]
    return embed_map, vec_size
    
def getSamplesAndFeatures(in_file, embed_filename, title=False):

    print('load Embeddings...')
    embed_map, vec_size = loadEmbeddings(embed_filename)
    
    documents, labels, features, scores, grids, paralabs = loadDataAndFeature(in_file, title)

    
    en_documents, en_labels = encode(documents, labels, embed_map, vec_size)
    en_paralabels = paraEncode(paralabs)
    
    return en_documents, en_labels, features, scores, vec_size, grids, en_paralabels
    
            
def batchGeneratorSGT(input_data, batch_n, is_random=False):
    en_documents, labels, features, scores, grids, en_paralabels = input_data
    if type(labels[0][0]) in (int, str):
        mutiLabel = 0
    else:
        mutiLabel = len(labels[0])
    data = list(zip(en_documents, labels, features, scores, grids, en_paralabels))
    
    data.sort(key=lambda x: len(x[0]))
    for i in range(0, len(en_documents), batch_n):
        if is_random:
            random.seed()
            mid = random.randint(0,len(en_documents)-1)
            # print(mid)
            start = max(0, mid - int(batch_n/2))
            end = min(len(en_documents), mid + math.ceil(batch_n/2))
        else:
            start = i
            end = i + batch_n
        # print(start, end)
        b_data = data[start: end]

        b_docs, b_labs, b_ft, b_s, b_g, b_p = zip(*b_data)
        b_ft = list(b_ft)
        b_docs = list(b_docs)
        b_labs = list(b_labs)
        b_s = list(b_s)
        b_g = list(b_g)
        b_p = list(b_p)
        max_len = len(b_docs[-1])
        if len(b_docs[0]) == max_len:
            yield b_docs, b_labs, b_ft, b_s, b_g, b_p
        else:
            sen_len = len(b_docs[0][0])
            vec_size = len(b_docs[0][0][0])
            ft_len = len(b_ft[0][0])
            
            for j in range(len(b_docs)):
                if len(b_docs[j]) < max_len:
                    l = len(b_docs[j])
                    b_docs[j] = b_docs[j] + [[PADDING * vec_size] * sen_len] * (max_len - l)
                    if not mutiLabel:
                        b_labs[j] = b_labs[j] + [LABELPAD] * (max_len - l)
                    else:
                        b_labs[j] = [b_labs[j][0] + ([LABELPAD]) * (max_len - l),
                                     b_labs[j][1] + PADDING * (max_len - l)]

                    b_ft[j] = b_ft[j] + [PADDING * ft_len] * (max_len - l)
                else:
                    break
            yield b_docs, b_labs, b_ft, b_s, b_g, b_p
            
    
def dataSplitGT(X, Y, ft, Y2, G, Para, p=0.1):
    # random.seed(312)
    test_idx = []
    score_c = [0, 0, 0]
    for i in range(len(X)):
        s = Y2[i]
        if score_c[s] % int(1/p) == 0:
            test_idx.append(i)
        score_c[s] += 1
    X_test = []
    Y_test = []
    Y2_test = []
    ft_test = []
    G_test = []
    Para_test = []
    
    X_train = []
    Y_train = []
    Y2_train = []
    ft_train = []
    G_train = []
    Para_train = []
    for i in range(len(X)):
        if i in test_idx:
            X_test.append(X[i])
            Y_test.append(Y[i])
            Y2_test.append(Y2[i])
            ft_test.append(ft[i])
            G_test.append(G[i])
            Para_test.append(Para[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
            Y2_train.append(Y2[i])
            ft_train.append(ft[i])
            G_train.append(G[i])
            Para_train.append(Para[i])

    return (X_train, Y_train, ft_train, Y2_train, G_train, Para_train), (X_test, Y_test, ft_test, Y2_test, G_test, Para_test)
    
    
def discretePos(features):
    for feat in features:
        for f in feat:
            f[3] = math.ceil(f[0]*40)
            f[4] = math.ceil(f[1]*20)
            f[5] = math.ceil(f[2]*10)
    return features
    
def loadFolds(folds_file):
    folds = []
    with open(folds_file, 'r') as f:
        for line in f:
            line = line.split('\t')
            folds.append([int(x) for x in line])
    return folds
