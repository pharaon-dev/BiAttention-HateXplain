import torch
import torch.nn as nn
import numpy as np


debug=False
# Custom Layers

def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        
        eij = torch.tanh(eij)
        
        eij[~mask] = float('-inf')
        a=torch.softmax(eij, dim=1)
        
#         a = torch.exp(eij)
#         if(debug==True):
#             print("a shape",a.shape)
#             print("mask shape",mask.shape)
#         if mask is not None:
#             a = a * mask

#         a = a /(torch.sum(a, 1, keepdim=True) + 1e-10)
        
        
        
        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        return torch.sum(weighted_input, 1),a

import sys


class Attention_LBSA(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention_LBSA, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, feature_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        context=torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector=nn.Parameter(context)
        if bias:
            self.b = nn.Parameter(torch.zeros(feature_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)
        eij = eij.view(-1, step_dim)

#         a = torch.exp(eij)
#         if(debug==True):
#             print("a shape",a.shape)
#             print("mask shape",mask.shape)
#         if mask is not None:
#             a = a * mask

#         a = a /(torch.sum(a, 1, keepdim=True) + 1e-10)

        eij[~mask] = float('-inf')
        a=torch.softmax(eij, dim=1)
        
        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        
        return torch.sum(weighted_input, 1),a


class Attention_BiRNN_Layer(nn.Module):
    def __init__(self, feature_dim, step_dim, drop_hidden, drop_embed, bias=True, **kwargs):
        super(Attention_BiRNN_Layer, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.output_dim = 1 
        self.maxpool1D = nn.MaxPool1d(self.step_dim, stride=1)
        drop_embed_x = 0.2
        self.dropout_embed = nn.Dropout2d(drop_embed_x)    
        self.seq_model = nn.LSTM(self.feature_dim, self.output_dim, bidirectional=True, num_layers=1, batch_first=True, dropout=drop_hidden)

        self.bias_att = nn.Parameter(torch.zeros(1, 1))
        #self.bias_ctxt = nn.Parameter(torch.zeros(1))
        
    
    def forward(self, x, mask=None):
        y = torch.squeeze(self.dropout_embed(torch.unsqueeze(x, 0))).view(x.shape[0], x.shape[1], x.shape[2])    
        a, _ = self.seq_model(y)
        
        if(debug):
            print("attention", a.shape)

        a = torch.sum(a, dim=2)
        a = a + self.bias_att 
        #a = a.squeeze(2)
        a[~mask] = float('-inf')
        a = torch.softmax(a, dim=1)
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input", weighted_input.shape)
        #r = torch.cat((torch.max(weighted_input, dim=1)[0], torch.sum(weighted_input, dim=1)), dim=1)
        return torch.max(weighted_input, dim=1)[0], a
        #return self.maxpool1D(weighted_input.permute(0, 2, 1)).view(-1, self.feature_dim), a

#@w.pMFj:*6N_aWU
class Attention_BiRNN_Layer_v2(nn.Module):
    def __init__(self, feature_dim, step_dim, drop_hidden, drop_embed, bias=True, **kwargs):
        super(Attention_BiRNN_Layer_v2, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        #self.maxpool1D = nn.MaxPool1d(self.step_dim, stride=1)
        context=torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector=nn.Parameter(context)
        drop_embed_x = 0.5
        self.dropout_embed = nn.Dropout2d(drop_embed_x)
        #self.conv = nn.Conv1d(self.feature_dim, 1, 1)
        self.seq_model = nn.LSTM(1, 1, bidirectional=True, num_layers=1, batch_first=True, dropout=drop_hidden)
        if bias:
            self.b = nn.Parameter(torch.zeros(feature_dim))
        
    def forward(self, x, mask=None):
        temp = x.contiguous().view(-1, self.feature_dim)
        temp = torch.mm(temp, self.context_vector)
        temp = temp.view(-1, self.step_dim, 1)
        temp = torch.squeeze(self.dropout_embed(torch.unsqueeze(temp, 0))).view(temp.shape[0], temp.shape[1], temp.shape[2])
        a, _ = self.seq_model(temp)
        if(debug):
            print("attention", a.shape)
        a = torch.sum(a, dim=2)
        #a = a.squeeze(2)
        a[~mask] = float('-inf')
        a=torch.softmax(a, dim=1)
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        
        #r = torch.cat((torch.max(weighted_input, dim=1)[0], torch.sum(weighted_input, dim=1)), dim=1)
        return torch.min(weighted_input, dim=1)[0], a
        #return self.maxpool1D(weighted_input.permute(0, 2, 1)).view(-1, self.feature_dim), a


class Analyse_Layer(nn.Module):
    def __init__(self, feature_dim, step_dim, drop_hidden, bias=True, **kwargs):
        super(Analyse_Layer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(self.feature_dim,self.feature_dim, 2)
        self.conv2 = nn.Conv1d(self.feature_dim,self.feature_dim, 3,padding=1)
        self.conv3 = nn.Conv1d(self.feature_dim,self.feature_dim, 4,padding=2)
        self.maxpool1D = nn.MaxPool1d(4, stride=4)
        self.seq_model = nn.GRU(self.feature_dim, self.feature_dim, bidirectional=False, batch_first=True, dropout=drop_hidden)


    def forward(self, h_embedding):
        new_conv1=self.maxpool1D(self.conv1(h_embedding.permute(0,2,1)))
        new_conv2=self.maxpool1D(self.conv2(h_embedding.permute(0,2,1)))
        new_conv3=self.maxpool1D(self.conv3(h_embedding.permute(0,2,1)))
        concat=self.maxpool1D(torch.cat([new_conv1, new_conv2,new_conv3], dim=2))
        h_seq, _ = self.seq_model(concat.permute(0,2,1))
        global_h_seq=torch.squeeze(global_max_pooling(h_seq, 1, 1))
        
        return global_h_seq

class Attention_K_Layer(nn.Module):
    def __init__(self, feature_dim, step_dim, drop_hidden, bias=True, **kwargs):
        super(Attention_K_Layer, self).__init__(**kwargs)
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        
        self.seq_model = nn.LSTM(self.feature_dim, 1, bidirectional=True, batch_first=True, dropout=drop_hidden)
        self.seuil_best_contribution = 0.1
        if bias:
            self.b = nn.Parameter(torch.zeros(feature_dim))


    def forward(self, x, mask):    
        att, _ = self.seq_model(x)
        if(debug):
            print("attention", att.shape)
        att = torch.sum(att, dim=2)
        att[~mask] = float('-inf')
        att = torch.softmax(att, dim=1)
        weighted_input = x * torch.unsqueeze(att, -1)
        
        a = dict()
        output = torch.zeros(x.shape[0], x.shape[2])

        for i in range(att.shape[0]):
            b = att[i] >= self.seuil_best_contribution
            a[i] = b.nonzero().squeeze(1)
            if a[i].shape[0] == 0:
                a[i] = (torch.max(att[i], dim=0)[1]).long()

        for i in range(x.shape[0]):
            output[i] = torch.sum(torch.index_select(weighted_input[i], 0, a[i]), 0)

        return output, att



class Attention_LBSA_sigmoid(Attention_LBSA):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super().__init__(feature_dim, step_dim, bias, **kwargs)
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        #eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)
        eij = eij.view(-1, step_dim)
        sigmoid = nn.Sigmoid()
        a=sigmoid(eij)
           
        if(debug==True):
            print("a shape",a.shape)
            print("mask shape",mask.shape)
        if mask is not None:
            a = a * mask

        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        #self.conv(weighted_input).squeeze(1)
        
        return torch.sum(weighted_input, dim=1), a

class Attention_LBSA_VP(Attention_LBSA):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super().__init__(feature_dim, step_dim, bias, **kwargs)
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)

        eij = eij.view(-1, step_dim)

#         a = torch.exp(eij)
#         if(debug==True):
#             print("a shape",a.shape)
#             print("mask shape",mask.shape)
#         if mask is not None:
#             a = a * mask
#         a = a /(torch.sum(a, 1, keepdim=True) + 1e-10)
        
        eij[~mask] = float('-inf')
        
        a=torch.softmax(eij, dim=1)
        if(debug):
            print("attention", a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        
        index = (torch.max(a, dim=1)[1]).long()  
               
        p = torch.zeros(a.shape[0], a.shape[1]) 
        for i in range(weighted_input.shape[0]):
            temp = weighted_input[i]
            p[i] = temp[index[i]]

        if(debug):
            print("weighted input",weighted_input.shape)
        
        
        return p, a


class Attention_LBSA_CONV(Attention_LBSA):
    def __init__(self, feature_dim, step_dim, drop_hidden, bias=True, **kwargs):
        super().__init__(feature_dim, step_dim, bias, **kwargs)
        self.conv1 = nn.Conv1d(self.feature_dim,self.feature_dim, 2)
        self.conv2 = nn.Conv1d(self.feature_dim,self.feature_dim, 3,padding=1)
        self.conv3 = nn.Conv1d(self.feature_dim,self.feature_dim, 4,padding=2)
        self.maxpool1D = nn.MaxPool1d(4, stride=4)
        self.seq_model = nn.GRU(self.feature_dim, self.feature_dim, bidirectional=False, batch_first=True, dropout=drop_hidden)
        self.seq_model_att = nn.LSTM(self.feature_dim, 1, bidirectional=True, num_layers=1, batch_first=True, dropout=drop_hidden)


    def forward(self, x, mask=None):
        a, _ = self.seq_model_att(x)
        if(debug):
            print("attention", a.shape)
        a = torch.sum(a, dim=2)
        #a = a.squeeze(2)
        a[~mask] = float('-inf')
        a=torch.softmax(a, dim=1)
        weighted_input = x * torch.unsqueeze(a, -1)
        h_embedding = weighted_input
        new_conv1=self.maxpool1D(self.conv1(h_embedding.permute(0,2,1)))
        new_conv2=self.maxpool1D(self.conv2(h_embedding.permute(0,2,1)))
        new_conv3=self.maxpool1D(self.conv3(h_embedding.permute(0,2,1)))
        concat=self.maxpool1D(torch.cat([new_conv1, new_conv2,new_conv3], dim=2))
        h_seq, _ = self.seq_model(concat.permute(0,2,1))
        global_h_seq=torch.squeeze(global_max_pooling(h_seq, 1, 1))
        
        return global_h_seq, a





















import sys

class CoAttention_LBSA(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(CoAttention_LBSA, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, feature_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        context=torch.zeros(step_dim, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector=nn.Parameter(context)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
            
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim   
        eij = torch.zeros(x.shape[0], step_dim, step_dim)

        for i in range(x.shape[0]):
            temp = x[i]
            temp1 = torch.mm(temp, self.weight)
            temp = torch.mm(temp1, temp.permute(1, 0))
            eij[i] = temp

        eij = eij.contiguous().view(-1, step_dim)

        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)

        if(debug):
            print("eij step 3",eij.shape)
        eij = eij.view(-1, step_dim)
        
        if(debug==True):
            print("mask shape",mask.shape)

        eij[~mask] = float('-inf')
        a=torch.softmax(eij, dim=1)

        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        
        return torch.sum(weighted_input, 1),a

class CoAttention_LBSA_sigmoid(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(CoAttention_LBSA_sigmoid, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, feature_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        context=torch.zeros(step_dim, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector=nn.Parameter(context)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
            
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim   
        eij = torch.zeros(x.shape[0], step_dim, step_dim)

        for i in range(x.shape[0]):
            temp = x[i]
            temp1 = torch.mm(temp, self.weight)
            temp = torch.mm(temp1, temp.permute(1, 0))
            eij[i] = temp

        eij = eij.contiguous().view(-1, step_dim)

        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)

        if(debug):
            print("eij step 3",eij.shape)
        eij = eij.view(-1, step_dim)
        sigmoid = nn.Sigmoid()
        a=sigmoid(eij)
           
        if(debug==True):
            print("a shape",a.shape)
            print("mask shape",mask.shape)
        if mask is not None:
            a = a * mask

        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        
        return torch.sum(weighted_input, 1),a


class Attention_TCH(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention_TCH, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, feature_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        context=torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector=nn.Parameter(context)
        if bias:
            self.b = nn.Parameter(torch.zeros(feature_dim))
        
    def forward(self, data):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        x, mask, att = data
        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)
        #print("eij ", eij.shape)
        #sys.exit(0)
        eij = eij.view(-1, step_dim)

#         a = torch.exp(eij)
#         if(debug==True):
#             print("a shape",a.shape)
#             print("mask shape",mask.shape)
#         if mask is not None:
#             a = a * mask

#         a = a /(torch.sum(a, 1, keepdim=True) + 1e-10)
        
        eij[~mask] = float('-inf')
        a=torch.softmax(eij, dim=1)
       
        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)

        return weighted_input, mask, a

class Stack_Attention(nn.Module):
    def __init__(self, num_of_layers, feature_dim, step_dim, bias=True, **kwargs):
        super(Stack_Attention, self).__init__(**kwargs)
        
        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = Attention_TCH(
                feature_dim = feature_dim,
                step_dim = step_dim
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

            
    def forward(self, data):
        weighted_input, mask, att = self.gat_net(data)
        return torch.sum(weighted_input, 1), mask, att



class BiAttention_RNN(nn.Module):
    def __init__(self, feature_dim, step_dim, drop_hidden, hidden_dim, bias=True, **kwargs):
        super(BiAttention_RNN, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        self.hidden_dim = hidden_dim

        weight = torch.zeros(feature_dim, feature_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        context=torch.zeros(self.hidden_dim, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector=nn.Parameter(context)

        self.seq_model = nn.LSTM(feature_dim, int(feature_dim//2), bidirectional=True, batch_first=True,dropout=drop_hidden)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
            
    def forward(self, x, emb, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        att,_ = self.seq_model(x)
        #att = torch.squeeze(att, 2)
        #att = att.view(-1, step_dim)
        
        if(debug):
            print("attention",a.shape)
        
        att = torch.sum(att, 2)
        #att = att.contiguous().view(-1, feature_dim)
        #att = torch.mm(att, self.context_vector)
        #att = att.view(-1, step_dim)


        att[~mask] = float('-inf')
        a = torch.softmax(att, dim=1)

        y = emb
        weighted_input = y.contiguous().view(-1, self.hidden_dim)
        weighted_input = torch.mm(weighted_input, self.context_vector)
        weighted_input = weighted_input.view(-1, step_dim)
        weighted_input = weighted_input * a

        if(debug):
            print("weighted input",weighted_input.shape)
        
        return weighted_input, a
