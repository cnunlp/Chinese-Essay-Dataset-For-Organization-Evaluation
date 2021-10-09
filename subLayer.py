import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, pos):
        gp_embds = Variable(self.pe[pos[:, :, 0]], requires_grad=False)
        lp_embds = Variable(self.pe[pos[:, :, 1]], requires_grad=False)
        pp_embds = Variable(self.pe[pos[:, :, 2]], requires_grad=False)
        return gp_embds, lp_embds, pp_embds

class PositionLayer(nn.Module):
    def __init__(self, p_embd=None, p_embd_dim=16, zero_weight=False):
        super(PositionLayer, self).__init__()
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        
        if zero_weight:
            self.pWeight = nn.Parameter(torch.zeros(3))
        else:
            self.pWeight = nn.Parameter(torch.ones(3))

        
        if p_embd == 'embd':
            self.g_embeddings = nn.Embedding(41, p_embd_dim)
            self.l_embeddings = nn.Embedding(21, p_embd_dim)
            self.p_embeddings = nn.Embedding(11, p_embd_dim)
        elif p_embd == 'embd_c':
            self.pe = PositionalEncoding(p_embd_dim, 100)

    def forward(self, sentpres, pos):
        # sentpres: (batch_n, doc_l, output_dim*2)
        if self.p_embd in utils.embd_name:
            pos = pos[:, :, 3:6].long()
        else:
            pos = pos[:, :, :3]
        if self.p_embd == 'embd':
            gp_embds = torch.tanh(self.g_embeddings(pos[:, :, 0]))
            lp_embds = torch.tanh(self.l_embeddings(pos[:, :, 1]))
            pp_embds = torch.tanh(self.p_embeddings(pos[:, :, 2]))
            # print(sentpres.size(), lp_embds.size(), gp_embds.size())
            sentpres = torch.cat((sentpres, gp_embds, lp_embds, pp_embds), dim=2)
        elif self.p_embd == 'embd_c':
            gp_embds, lp_embds, pp_embds = self.pe(pos)
            sentpres = sentpres + self.pWeight[0] * gp_embds + \
                                  self.pWeight[1] * lp_embds + \
                                  self.pWeight[2] * pp_embds                   
        elif self.p_embd == 'cat':
            sentpres = torch.cat((sentpres, pos), dim=2)
        elif self.p_embd =='add':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[1] * pos[:, :, 1:2] + self.pWeight[2] * pos[:, :, 2:3]
            
        return sentpres
        
    def init_embedding(self):
        gp_em_w = [[i/40] * self.p_embd_dim for i in range(41)]
        self.g_embeddings.weight = nn.Parameter(torch.FloatTensor(gp_em_w))
        lp_em_w = [[i/20] * self.p_embd_dim for i in range(21)]
        self.l_embeddings.weight = nn.Parameter(torch.FloatTensor(lp_em_w))
        pp_em_w = [[i/10] * self.p_embd_dim for i in range(11)]
        self.p_embeddings.weight = nn.Parameter(torch.FloatTensor(pp_em_w))
       

# title独立特征，SPP特征不包含title
class InterSentenceSPPLayer(nn.Module):
    def __init__(self, hidden_dim, num_levels=4, pool_type='max_pool', active_func='tanh'):
        super(InterSentenceSPPLayer, self).__init__()
        self.linearK = nn.Linear(hidden_dim, hidden_dim)
        self.linearQ = nn.Linear(hidden_dim, hidden_dim)
        self.num_levels = num_levels
        self.pool_type = pool_type
        if self.pool_type == 'max_pool':
            self.SPP = nn.ModuleList([nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])
        elif self.pool_type == 'avg_pool':
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)])
        else:
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)] + [nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])
        if active_func == 'tanh':
            self.active_func = nn.Tanh()
        elif active_func == 'relu':
            self.active_func = nn.ReLU()
        elif active_func == 'softmax':
            self.active_func = nn.Softmax(dim=2)
        else:
            self.active_func = None
        
    def forward(self, sentpres):
        # sentpres: (batch_n, doc_l, output_dim*2)
        doc_l = sentpres.size(1)
        key = self.linearK(sentpres)
        query = self.linearQ(sentpres)
        d_k = query.size(-1)
        features = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # features: (batch_n, doc_l, doc_l)

        if self.active_func is not None:
            features = self.active_func(features)
        self.ft =  features
        features_no_title = features[:, :, 1:]
        pooling_layers = [features[:, :, 0:1]]
        for pooling in self.SPP:
            tensor = pooling(features_no_title)
            pooling_layers.append(tensor)
            
        # print([x.size() for x in pooling_layers])
        self.features = torch.cat(pooling_layers, dim=-1)
        return self.features
        
        