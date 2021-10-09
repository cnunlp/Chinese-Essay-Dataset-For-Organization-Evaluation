import utils
# import config
from sModel import STPSPPWithGridCNN

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import os

import logging

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

date_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./log/sent_tag_%s.log' % date_str,
                filemode='w')
    

def list2tensor(x, y, ft, y2, g, para, p_embd, device='cpu'):
    inputs = torch.tensor(x, dtype=torch.float, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)
    paralabs = torch.tensor(para, dtype=torch.long, device=device)
    # print(g)
    gridm = torch.tensor(g, dtype=torch.long, device=device)
    if y2 is not None:
        scores = torch.tensor(y2, dtype=torch.long, device=device)
    else:
        scores = None
    tp = torch.tensor(ft, dtype=torch.float, device=device)
    return inputs, labels, tp, scores, gridm, paralabs

def getMask(ft, device='cpu'):
    slen = torch.tensor(ft, dtype=torch.long)[:, :, 6]
    # print(slen)
    s_n = slen.size(0) * slen.size(1)
    slen = slen.view(s_n)
    # print(slen)
    mask = torch.zeros((s_n, 40)) == 1
    for i, d in enumerate(slen):
        if d < 40:
            mask[i, d:] = 1
    if device == 'cuda':
        return mask.cuda()
    return mask
    
def train(model, X, Y, FT, Y2, G, Para, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, is_mask=False, step=None):
    train_data, test_data = utils.dataSplitGT(X, Y, FT, Y2, G, Para, 0.1)
    # X_train, Y_train, ft_train, Y2_train, G_train, Para_train = train_data
    
    if(is_gpu):
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'
        
    modelName = model.getModelName()
    if title:
        modelName += '_t'
    if step is not None:
        modelName += '_s' + str(step)
        if step in [3, 4, 5]:
            for name, param in model.named_parameters():
                param.requires_grad = True
        else:
            if step == 1:
                step1 = True
            else:
                step1 = False
            for name, param in model.named_parameters():
                if name.split('.')[0] in ['scoreCL', 'scoreLayer']:
                    param.requires_grad = not step1
                else:
                    param.requires_grad = step1
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
    
    modelName += '_' + str(fold_k)
    logging.info(modelName) 
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),   #只更新requires_grad=True的参数
                          lr=lr)  

    w_r = 1  
    w_s = 0.1  
    w_p = 0.1  
    
    loss_function = nn.NLLLoss()
    w = torch.tensor([1.1, 1, 1.1], device=device)
    loss_function2 = nn.NLLLoss(weight=w)

    
    loss_list = []
    loss_list_s = []
    loss_list_p = []
    acc_list = []
    acc_list_s = []
    acc_list_p = []
    last_loss = 100
    c = 0
    c1 = 0

    
    last_acc, _, last_acc_s, _, last_acc_p, _  = test(model, test_data, device, title=title, is_mask=is_mask)
    # acc_list.append(last_acc)
    logging.info('first acc: %f acc_s: %f acc_p: %f' % (last_acc, last_acc_s, last_acc_p))  
    # last_acc = max(0.6, last_acc)
    for epoch in range(epoch_n):
        total_loss = 0
        total_loss_r = 0
        total_loss_s = 0
        total_loss_p = 0
        gen = utils.batchGeneratorSGT(train_data, batch_n, is_random=True)
        i = 0
        model.train()
        for x, y, ft, y2, g, para in gen:
            optimizer.zero_grad()
            
            inputs, labels, tp, labels2, gridm, paralabs = list2tensor(x, y, ft, y2, g, para, model.p_embd, device)
            
            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None

            if title:
                result, scores, paraResult = model(inputs, pos=tp, device=device, mask=mask, grid=gridm)
                result = result[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result, scores, paraResult = model(inputs, pos=tp, device=device, mask=mask, grid=gridm)
            
   
            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            
            r_n = paralabs.size()[0]*paralabs.size()[1]
            # paraResult = paraResult.view(r_n)
            paralabs = paralabs.view(r_n)
                              
            loss = loss_function(result, labels)
            loss_s = loss_function2(scores, labels2)
            loss_p = loss_function(paraResult, paralabs)
            # print(loss)
            # print(loss_s)
            
            if step is not None:
                if step == 1:
                    t_loss = loss + loss_p
                    total_loss += (loss + loss_p).cpu().detach().numpy()
                elif step == 4:
                    t_loss = loss_p
                    total_loss += loss_p.cpu().detach().numpy()
                elif step == 5:
                    t_loss = loss_p + loss_s
                    total_loss += (loss_s + loss_p).cpu().detach().numpy()
                else:
                    t_loss = loss_s
                    total_loss += loss_s.cpu().detach().numpy()
            else:
                t_loss = loss + loss_p + w_s*loss_s

            
                total_loss += (loss + loss_s + loss_p).cpu().detach().numpy()

            t_loss.backward()
            optimizer.step()

            total_loss_r += loss.cpu().detach().numpy()
            total_loss_s += loss_s.cpu().detach().numpy()
            total_loss_p += loss_p.cpu().detach().numpy()
            # print(total_loss)
            i += 1
            
        aver_loss = total_loss/i
        aver_loss_r = total_loss_r/i
        aver_loss_s = total_loss_s/i
        aver_loss_p = total_loss_p/i
        loss_list.append(aver_loss_r)
        loss_list_s.append(aver_loss_s)
        loss_list_p.append(aver_loss_p)
        
        accuracy, _, accuracy2, _, accuracy3, _ = test(model, test_data, device, title=title, is_mask=is_mask)
        acc_list.append(accuracy)
        acc_list_p.append(accuracy3)
        if (step is None) or (step == 1):
            if last_acc < accuracy:
                last_acc = accuracy
                if accuracy > 0.64:
                    s_flag = True
                    torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch/20)*20))

        
        if (step is None) or (step in [4, 5]):
            if last_acc_p < accuracy3:
                last_acc_p = accuracy3
                if accuracy3 > 0.58:
                    torch.save(model, model_dir + '%s_%d_pbest.pk' % (modelName, int(epoch/10)*10))
                    
        acc_list_s.append(accuracy2)            
        if (step is None) or (step in [2, 3, 5]):        
            if last_acc_s < accuracy2:
                last_acc_s = accuracy2
                if accuracy2 > 0.56:
                    torch.save(model, model_dir + '%s_%d_sbest.pk' % (modelName, int(epoch/20)*20))
        if step is None:
            logging.info('%d loss: %.3f, loss_s: %.3f, loss_p: %.3f, acc: %.3f, acc_s: %.3f, acc_p: %.3f, w_r: %.3f, w_s: %.3f, w_p: %.3f' % 
                         (epoch, aver_loss_r, aver_loss_s, aver_loss_p, accuracy, accuracy2, accuracy3, w_r, w_s, w_p))
        elif step == 4:
            logging.info('%d loss_p: %f, acc_p: %f' % 
                         (epoch, aver_loss_p, accuracy3))
        elif step == 5:
            logging.info('%d loss_s: %f, loss_p: %f, acc_s: %f, acc_p: %f, loss_w: %f' % 
                         (epoch, aver_loss_s, aver_loss_p, accuracy2, accuracy3, w_s))
        else:
            logging.info('%d loss: %.3f, loss_s: %.3f, loss_p: %.3f, acc: %.3f, acc_s: %.3f, acc_p: %.3f, loss_w: %.3f, loss_w_s: %.3f, loss_w_p: %.3f' % 
                         (epoch, aver_loss_r, aver_loss_s, aver_loss_p, accuracy, accuracy2, accuracy3, w_r, w_s, w_p))
        
        if step is None:
            w_s = min((aver_loss_s+0.0) / aver_loss_r * w_s + 0.001, 1)
            w_p = min((aver_loss_p+0.0) / aver_loss_r * w_p + 0.001, 1)
            
        
        if(aver_loss > last_loss):
            c += 1
            if c == 10:
                lr = lr * 0.95
                optimizer.param_groups[0]['lr'] = lr
                logging.info('lr: %f, last loss: %f' % (lr, last_loss))
                c = 0
                last_loss = aver_loss
        else:
            c = 0
            last_loss = aver_loss
        torch.save(model, model_dir + '%s_last.pk' % (modelName))

        if(lr < 0.0001) or (aver_loss_r < 0.5) or (aver_loss_s < 0.004):
            break
        if step in [2, 3, 5] and (aver_loss_s < 0.1):
            break
        if step in [4, 5] and (aver_loss_p < 0.2):
            break
    plt.cla()
    loss_n = len(loss_list)
    plt.plot(range(loss_n), acc_list, range(loss_n), acc_list_s, range(loss_n), acc_list_p, 
             range(loss_n), loss_list, range(loss_n), loss_list_s, range(loss_n), loss_list_p)
    plt.legend(['acc_list_r', 'acc_list_s', 'acc_list_p', 'loss_list_r', 'loss_list_s', 'loss_list_p'])
    plt.savefig('./img/'+modelName+'.jpg')
    # plt.show()
    
def test(model, test_data, device='cpu', batch_n=1, title=False, is_mask=False):
    # X, Y, FT, Y2, G = test_data
    result_list = []
    scores_list = []
    para_list = []
    label_list = []
    label2_list = []
    paralabel_list = []
    model.eval()
    with torch.no_grad():
        gen = utils.batchGeneratorSGT(test_data, batch_n)
        for x, y, ft, y2, g, para in gen:
            
            inputs, labels, tp, labels2, gridm, paralabs = list2tensor(x, y, ft, y2, g, para, model.p_embd, device)
            
            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None
                
            if title:
                result, scores, paraResult = model(inputs, pos=tp, device=device, mask=mask, grid=gridm)
                result = result[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result, scores, paraResult = model(inputs, pos=tp, device=device, mask=mask, grid=gridm)
            
            # print(result)
            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            
            r_n = paralabs.size()[0]*paralabs.size()[1]
            # paraResult = paraResult.view(r_n)
            paralabs = paralabs.view(r_n)
            
            result_list.append(result)
            scores_list.append(scores)
            para_list.append(paraResult)
            label_list.append(labels)
            label2_list.append(labels2)
            paralabel_list.append(paralabs)

    preds = torch.cat(result_list)
    labels = torch.cat(label_list)
    t_c = 0
    a = np.zeros((8, 8))
    l = preds.size()[0]
    for i in range(l):
        p = preds[i].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())
        a[r][p] += 1
        if p == r:
            t_c += 1
    accuracy = t_c / l

    preds = torch.cat(scores_list)
    labels = torch.cat(label2_list)
    t_c = 0
    a2 = np.zeros((3, 3))
    l = preds.size()[0]
    for i in range(l):
        p = preds[i].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())

        a2[r][p] += 1
        if p == r:
            t_c += 1
    accuracy2 = t_c / l
    
    preds = torch.cat(para_list)
    labels = torch.cat(paralabel_list)
    t_c = 0
    t = 0
    a3 = np.zeros((8, 8))
    l = preds.size()[0]
    for i in range(l):
        p = preds[i].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())
        if r != 7:
            t += 1
            a3[r][p] += 1
            if p == r:
                t_c += 1
    accuracy3 = t_c / t
   
    return accuracy, a, accuracy2, a2, accuracy3, a3
  

def predict(model, test_data, device='cpu', batch_n=1, title=False, is_mask=False):
    X, FT, G = test_data

    model.eval()
    with torch.no_grad():
            
        inputs, _, tp, _, gridm, _ = list2tensor(X, [], FT, [], G, [], model.p_embd, device)
        
        if is_mask:
            mask = getMask(ft, device)
        else:
            mask = None
            
        if title:
            result, scores, paraResult = model(inputs, pos=tp, device=device, mask=mask, grid=gridm)
            result = result[:, 1:].contiguous()

        else:
            result, scores, paraResult = model(inputs, pos=tp, device=device, mask=mask, grid=gridm)
        
        # print(result)
        r_n = result.size()[0]*result.size()[1]
        result = result.contiguous().view(r_n, -1)
        
        result = result.cpu().argmax(dim=1).tolist()
        scores = scores.cpu().argmax(dim=1).tolist()
        paraResult = paraResult.cpu().argmax(dim=1).tolist()

   
    return result, scores, paraResult
   
def loadPretrainDict(tag_model, dict_name):
    
    pre_model = torch.load(dict_name)
    state_dict = {k:v for k,v in pre_model.state_dict().items() if 'scoreLayer' not in k and 'scoreCL' not in k}
    m_d = tag_model.state_dict()
    m_d.update(state_dict)
    tag_model.load_state_dict(m_d)
    return tag_model
    

    
if __name__ == "__main__":
    
    in_file = './data/c_all2_2.json'
    embed_filename = './embd/new_embeddings3.txt'
    title = True
    max_len = 40
    en_documents, en_labels, features, scores, vec_size, grids, en_paralabels = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)
    pad_documents, pad_labels = utils.sentence_padding(en_documents, en_labels, max_len, vec_size)
    # print(scores)
    
    is_mask = False
    
    class_n = 8
    s_class_n = 3
    batch_n = 50
    is_gpu =  True
    # is_gpu =  False
    
    hidden_dim = 128
    sent_dim = 128
    kernel_size = 5
    
    p_embd = 'embd_c'
    p_embd_dim=16
    
    if p_embd in ['embd_c']:
        p_embd_dim = hidden_dim*2


    ''' step: 
       None: all task
       1: multi role+para task
       2: pipeline score task
       3: single score task
       4: single para task
       5: multi para+score task
    '''
    step = None
    lr = 0.2
    
    model_dir = './model/scores/%s/' % date_str
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    folds_file = './data/c_folds2.txt'
    folds = utils.loadFolds(folds_file)
    
    f_list = range(len(folds))
    # f_list = [0]
    
    for fold_k in f_list:
        tag_model = STPSPPWithGridCNN(vec_size, hidden_dim, sent_dim, kernel_size, class_n, s_class_n, p_embd=p_embd, p_embd_dim=p_embd_dim)
        
        if step == 2:
            dict_name =  './model/scores/SFI_PFI/socp_grid_128_128_embd_c_t_s1_%d_best.pk' % fold_k
            tag_model = loadPretrainDict(tag_model, dict_name)
            lr = 0.003
        elif step in [3, 5]:
            lr = 0.01
            
        train_fold = []
        for i in range(len(folds)):
            if i != fold_k:
                train_fold += folds[i]
        train_docs = [pad_documents[i] for i in train_fold]
        train_labels = [pad_labels[i] for i in train_fold]
        train_features = [features[i] for i in train_fold]
        train_scores = [scores[i] for i in train_fold]
        train_grids = [grids[i] for i in train_fold]
        train_paras = [en_paralabels[i] for i in train_fold]
        
        logging.info('start training model...')
        starttime = datetime.datetime.now()
        train(tag_model, train_docs, train_labels, train_features, train_scores, train_grids, train_paras, is_gpu, epoch_n=700, lr=lr, batch_n=batch_n, title=title, is_mask=is_mask, step=step)
        endtime = datetime.datetime.now()
        logging.info(endtime - starttime)
        

