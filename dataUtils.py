from EssayBase import Essay
# from EssayBaseE import EssayE
# from EssayBaseS import EssayS

import json
import os, sys
import numpy as np
import math
from collections import Counter

from utils import loadEmbeddings, label_map, LABELPAD, para_label_map

import random
random.seed(312)
             
ne_map = {'Nh': 0, 'Ns': 1, 'Ni': 2}

def getAllWords(data_list, w_filename):
    
    word_count = Counter()

    for newdir in data_list:
        filenames = os.listdir('data/' + newdir)
        for file in filenames:
            fname = os.path.join('data/' + newdir, file)
            essay = Essay()
            essay.parse_file(fname)

            for sent in essay.sentences:
                for token in sent.tokens:
                    word_count[token.string] += 1
                
    wf = open(w_filename,'w',encoding='utf-8')
    for (index, w) in enumerate(word_count):
        wf.write(w + '\n')
    wf.close()
    
def getNames(data_list, w_filename):
    
    word_count = Counter()

    for newdir in data_list:
        filenames = os.listdir('data/' + newdir)
        for file in filenames:
            fname = os.path.join('data/' + newdir, file)
            essay = Essay()
            essay.parse_file(fname)

            for sent in essay.sentences:
                for token in sent.tokens:
                    if token.ne_tag in ['S-Nh', 'B-Nh']:
                        word_count[token.string] += 1
                
    wf = open(w_filename,'w',encoding='utf-8')
    for (index, w) in enumerate(word_count):
        wf.write(w + '\n')
    wf.close()
    
def posFunc(x):
    return (np.arcsin(2*x-1) + np.pi / 2 + 0.1) / (np.pi + 0.1)
    
def posPercent(pos_id, pos_num):
    if (pos_id - 1) / pos_num <= 0.5:
        return posFunc((pos_id-1)/pos_num)
    else:
        return posFunc(pos_id/pos_num)
        
def resave(sorceDir, newDir):
    filenames = os.listdir(sorceDir)
    
    for file in filenames:
        fname = os.path.join(sorceDir, file)
        essay = Essay()
        essay.parse_file(fname)
        nfname = os.path.join(newDir, file)
        with open (nfname, 'w', encoding='utf8') as f:
            # print(type(essay.to_xml()))
            f.write(bytes.decode(essay.to_xml()))
        

# 获取句子及其标签,包括评分
def getSentencesS(newdir, w_file):
    
    filenames = os.listdir(newdir)
    
    wf = open(w_file,'w',encoding='utf-8')
    for file in filenames:
        fname = os.path.join(newdir, file)
        essay = EssayS()
        essay.parse_file(fname)
        
        sent_num = len(essay.sentences) - 1
        para_num = len(essay.paragraphs) - 1
        sentences = []
        labels = []
        global_ids = []
        local_ids = []
        global_pos = []
        local_pos = []
        para_ids = []
        para_pos = []
        sent_lens = []
        nes = []
        
        title_c = 0
        for sent in essay.sentences:
            if sent.label in ['pointTit', 'topicTit', 'Tit', ' ', '  ', 'title']:
                # title = []
                # for token in sent.tokens:
                    # if token.pos_tag in poslist:
                        # title.append(token.string)
                        
                # if len(title) == 0:
                    # print(file, [token.string for token in sent.tokens])
                title = [token.string for token in sent.tokens]
                
                if title_c:
                    print(file, title)
                    sent_num = sent_num - 1
                title_c += 1
                continue
            # if sent.label in ['']:
                # print(file)
                # os.remove(fname)
                # break
            if title_c == 0:
                print(file)
            sentences.append([token.string for token in sent.tokens])
            labels.append(sent.label)
            sent_lens.append(len(sent.tokens))
            sent.global_id = sent.global_id - title_c + 1
            global_ids.append(sent.global_id)
            global_p = sent.global_id / sent_num
            global_pos.append(global_p)
            local_ids.append(sent.local_id + 1)
            local_p = (sent.local_id + 1) / essay.paragraphs[sent.para_id].n_sent
            local_pos.append(local_p)
            para_ids.append(sent.para_id)
            para_pos.append(sent.para_id / para_num)

            ne_l = [token.ne_tag[-2:] for token in sent.tokens 
                    if token.ne_tag in ['S-Nh', 'S-Ns', 'S-Ni', 'B-Ni', 'B-Nh', 'B-Ns']]
            nef = [0, 0, 0]
            for ne in ne_l:
                nef[ne_map[ne]] += 1
            nes.append(nef)
        
        score = int(essay.score)
        
        paras = [para.paratype for para in essay.paragraphs]
        
        json_data = {'file': file, 'title': title, 'score': score, 'sents': sentences, 'labels':labels, 
                     'gpos': global_pos, 'lpos': local_pos, 
                     'gid': global_ids, 'lid': local_ids,
                     'pid': para_ids, 'ppos': para_pos,
                     'slen': sent_lens, 'nes': nes,
                     'paras': paras}
        json.dump(json_data, wf, ensure_ascii=False)
        wf.write('\n')
            
    wf.close() 
    
def getParaLabelPic(in_file, w_file, max_len=20):
    wf = open(w_file, 'w', encoding='utf8')
    with open(in_file, 'r', encoding='utf8') as f:
        for line in f:
            # para = '  '
            para_lab = [7]*2
            # pic = []
            lab_pic = []
            
            load_dict = json.loads(line)
            
            wf.write(load_dict['file'] + '\t' + str(load_dict['score']) + '\n')
            
            title = ''.join(load_dict['title'])
            
            
            paras_lab = load_dict['paras']
            
            grid = []
            para = [para_label_map[paras_lab[load_dict['pid'][0]-1]]]
            # print(load_dict['pid'])
            # print(paras_lab)
            for i in range(len(load_dict['gid'])):
                if i > max_len:
                    break
                
                if load_dict['lid'][i] == 1:
                    if i != 0:
                        para = para + [LABELPAD]*(max_len-len(para)+1)
                        if len(para) > max_len+1:
                            para = para[:max_len+1]
                        # if len(para) > 20:
                            # print(para)
                        grid.append(para)
                        para = [para_label_map[paras_lab[load_dict['pid'][i]-1]]]
                para.append(label_map[load_dict['labels'][i]])      
            para = para + [LABELPAD]*(max_len-len(para)+1)
            if len(para) > max_len+1:
                para = para[:max_len+1]    
            grid.append(para)
            
            if len(grid) < max_len:
                grid += [[LABELPAD]*(max_len+1)] * (max_len-len(grid))
            elif len(grid) > max_len:
                grid = grid[:max_len]
            
            for r in grid:
                for c in r:
                    wf.write(str(c) + '\t')
                wf.write('\n')
                
    wf.close()