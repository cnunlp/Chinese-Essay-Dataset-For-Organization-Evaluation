import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayer import *

class STPSPPWithGridCNN(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, kernel_size, class_n, s_class_n, p_embd=None, p_embd_dim=16, pool_type='max_pool', active_func='tanh'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STPSPPWithGridCNN, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.kernel_size = kernel_size
        self.class_n = class_n
        self.s_class_n = s_class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type
        self.active_func = active_func
        
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        
        self.posLayer = PositionLayer(p_embd, p_embd_dim)
        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim*2, pool_type = self.pool_type, active_func=active_func)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim*2, pool_type = self.pool_type, active_func=active_func)
        
        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+p_embd_dim*3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim*2+3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim*2, self.sent_dim, bidirectional=True)
        
        if self.pool_type in ['max_pool', 'avg_pool']:
            sent_dim = self.sent_dim*2 + 32
        else:
            sent_dim = self.sent_dim*2 + 64
            
        self.classifier = nn.Linear(sent_dim, self.class_n)
            
        self.paraLayer = nn.LSTM(sent_dim, int(sent_dim/2), bidirectional=True)
        self.paraLayer2 = nn.MaxPool1d(20)
        self.paraCL = nn.Linear(sent_dim, self.class_n)
        
        self.scoreLayer = nn.Sequential(
                nn.Conv2d(in_channels=sent_dim, out_channels=64, kernel_size=self.kernel_size, 
                          stride=1, padding=int(self.kernel_size/2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 32, self.kernel_size, stride=1, padding=int(self.kernel_size/2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))
        self.scoreCL = nn.Linear(800, self.s_class_n)
        
        self.lossWeightR = nn.Parameter(torch.Tensor([0]))
        self.lossWeightS = nn.Parameter(torch.Tensor([0]))
        
        
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01), 
                                 torch.rand(2, batch_n*doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01), 
                                torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))
        
    def forward(self, documents, pos, grid, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()    # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)  
        documents = documents.view(batch_n*doc_l, sen_l, -1).transpose(0, 1)    # documents: (sen_l, batch_n*doc_l, word_dim)
        
        sent_out, _ = self.sentLayer(documents, self.sent_hidden)   # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        
        if mask is None:
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))      # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) /(sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))
        
        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim*2)   # sentpres: (batch_n, doc_l, hidden_dim*2)
        
        sentFt = self.sfLayer(sentpres)  
        
        sentpres = self.posLayer(sentpres, pos)
        
        sentpres = sentpres.transpose(0, 1)
        
        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)   # tag_out: (doc_l, batch_n, output_dim*2)
        tag_out = torch.tanh(tag_out)
        
        tag_out = tag_out.transpose(0, 1)
        roleFt = self.rfLayer(tag_out) 
        
        tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)
        
        result = self.classifier(tag_out)
        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        
        # grid_onehot = torch.zeros((batch_n, 20, 20, doc_l), device=device).scatter_(-1, grid.unsqueeze(-1), 1)
        grid_onehot = F.one_hot(grid, tag_out.size(1)).float()
        #去除填充部分
        grid_onehot[:, :, :, 0] = 0
        grid_m = torch.matmul(grid_onehot, tag_out.unsqueeze(1))    # grid_m: (batch_n, 20, 20, output_dim*2)
        
        paras_out, _ = self.paraLayer(grid_m.view(batch_n*20, 20, -1).transpose(0, 1))
        paras_out = self.paraLayer2(paras_out.permute(1, 2, 0)).squeeze(-1)
        paras = self.paraCL(paras_out)
        paras = F.log_softmax(paras, dim=1)  # result: (batch_n*20, class_n)
        
        paras_out = paras_out.view(batch_n, 20, -1).unsqueeze(2)
        grid_m = torch.cat((paras_out, grid_m), dim=2)[:, :, :20, :]
        
        score_out = self.scoreLayer(grid_m.permute(0, 3, 1, 2))   # score_out: (batch_n, 32, 5, 5)

        scores = self.scoreCL(torch.tanh(score_out.view(batch_n, -1)))  # scores: (batch_n, s_class_n)
        scores = F.log_softmax(scores, dim=1)  # scores: (batch_n, s_class_n)
        return result, scores, paras
        
    def getModelName(self):
        name = 'socp_spp_grid'
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim) + '_' + str(self.kernel_size)
        if self.p_embd == 'cat':
            name += '_cp'
        elif self.p_embd =='add':
            name += '_ap'
        elif self.p_embd =='embd':
            name += '_em'
        elif self.p_embd == 'embd_a':
            name += '_em_a'
        elif self.p_embd:
            name += '_' + self.p_embd
        return name   
                
                               
