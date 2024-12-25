import torch
import torch.nn as nn
import numpy as np
from Models.GAT import *
from Models.attentionLayer import *
from .utils import masked_cross_entropy

debug =False
#### BiGRUCLassifier model

def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret



class BiRNN(nn.Module):  
    def __init__(self,args,embeddings):
        super(BiRNN, self).__init__()
        self.hidden_size = args['hidden_size']
        self.batch_size = args['batch_size']
        self.drop_embed=args['drop_embed']
        self.drop_fc=args['drop_fc']
        self.embedsize=args["embed_size"]
        self.drop_hidden=args['drop_hidden']
        self.seq_model_name=args["seq_model"]
        self.weights =args["weights"]
        #
        self.vocab_size = args["vocab_size"]
        self.embedding = nn.Embedding(args["vocab_size"], self.embedsize)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings.astype(np.float32), dtype=torch.float32))
        self.embedding.weight.requires_grad = args["train_embed"]
        if(args["seq_model"]=="lstm"):
            self.seq_model = nn.LSTM(args["embed_size"], self.hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=self.drop_hidden)
        elif(args["seq_model"]=="gru"):
            self.seq_model = nn.GRU(args["embed_size"], self.hidden_size, num_layers=1, bidirectional=True, batch_first=True, dropout=self.drop_hidden) 
        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, args['num_classes'])
        self.dropout_embed = nn.Dropout2d(self.drop_embed)
        self.dropout_fc = nn.Dropout(self.drop_fc)
        self.num_labels=args['num_classes']
        
        
        
    def forward(self,input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        batch_size=input_ids.size(0)
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(batch_size,input_ids.shape[1],self.embedsize)
        if(self.seq_model_name=="lstm"):
            _, hidden = self.seq_model(h_embedding)
            hidden=hidden[0]
        else:
            _, hidden = self.seq_model(h_embedding)
            
        if(debug):
            print(hidden.shape)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1) 
        hidden = self.dropout_fc(hidden)
        hidden = torch.relu(self.linear1(hidden))  #batch x hidden_size
        hidden = self.dropout_fc(hidden)
        logits = self.linear2(hidden)
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(logits.view(-1, self.num_labels), labels.view(-1)) 
            return (loss_logits,logits)  
        return (logits,)
    
    
    
    def init_hidden(self, batch_size):
        return cuda_available(torch.zeros(2, self.batch_size, self.hidden_size))


class LSTM_bad(BiRNN):
    def __init__(self,args):
        super().__init__(args)
        self.seq_model = nn.LSTM(args["embed_size"], self.hidden_size, bidirectional=False, batch_first=True,dropout=self.drop_hidden)
    def forward(self,x,x_mask):
        batch_size=x.size(0)
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(batch_size,x.shape[1],self.embedsize)
        _, hidden = self.seq_model(h_embedding)
        hidden=hidden[0]
        if(debug):
            print(hidden.shape)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1) 
        hidden = self.dropout_fc(hidden)
        return (self.linear2(hidden))  
       
    
    

class CNN_GRU(BiRNN):
    def __init__(self,args,embeddings):
        super().__init__(args,embeddings)
        self.conv1 = nn.Conv1d(self.embedsize,100, 2)
        self.conv2 = nn.Conv1d(self.embedsize,100, 3,padding=1)
        self.conv3 = nn.Conv1d(self.embedsize,100, 4,padding=2)
        self.maxpool1D = nn.MaxPool1d(4, stride=4)
        self.seq_model = nn.GRU(100, 100, bidirectional=False, batch_first=True,dropout=self.drop_hidden)
        self.out = nn.Linear(100, args["num_classes"])
        
    def forward(self,input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        batch_size=input_ids.size(0)
        h_embedding = self.embedding(input_ids)
        h_embedding = self.dropout_embed(h_embedding)
        new_conv1=self.maxpool1D(self.conv1(h_embedding.permute(0,2,1)))
        new_conv2=self.maxpool1D(self.conv2(h_embedding.permute(0,2,1)))
        new_conv3=self.maxpool1D(self.conv3(h_embedding.permute(0,2,1)))
        concat=self.maxpool1D(torch.cat([new_conv1, new_conv2,new_conv3], dim=2))
        h_seq, _ = self.seq_model(concat.permute(0,2,1))
        global_h_seq=torch.squeeze(global_max_pooling(h_seq, 1, 1)) 
        global_h_seq = self.dropout_fc(global_h_seq)
        output=self.out(global_h_seq)
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1)) 
            return (loss_logits,output)  
        return (output,)
    
        return output
    
class BiAtt_RNN(BiRNN):
    def __init__(self,args,embeddings,return_att):
        super().__init__(args,embeddings)
        if(args['attention']=='sigmoid'):
            #self.seq_attention = Attention_LBSA_sigmoid(self.hidden_size * 2, args['max_length'])
            self.seq_attention = Attention_BiRNN_Layer(2*self.hidden_size, args['max_length'], self.drop_hidden, self.drop_embed)
        else:
            #self.seq_attention = Attention_LBSA(self.hidden_size * 2, args['max_length'])
            self.seq_attention = Attention_BiRNN_Layer(2*self.hidden_size, args['max_length'], self.drop_hidden, self.drop_embed)
        self.linear = nn.Linear(2*self.hidden_size, args["batch_size"])
        self.relu = nn.ReLU()
        self.out = nn.Linear(args["batch_size"], args["num_classes"])
        self.return_att = False
        self.lam = args['att_lambda']
        self.train_att =args['train_att']
        self.max_length = args['max_length']

    def concatenate_column_index(self, x, y):
        nb_feature = self.hidden_size * 2
        p = torch.zeros(x.shape[0], x.shape[1] + y.shape[1]) 
        for i in range(nb_feature):
            p[:, i] = x[:, i]
            p[:, i+1] = y[:, i]  
        return p

    def forward(self, input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(input_ids.shape[0],input_ids.shape[1],self.embedsize)

        h_seq, _ = self.seq_model(h_embedding)
        if(debug):
            print("output",h_seq.shape)
        #h_seq = self.relu(h_seq)
        h_seq_atten, att = self.seq_attention(h_seq, attention_mask)
        if(debug):
            print("h_seq_atten",h_seq_atten.shape)

        #h_seq_atten = self.dropout_fc(h_seq_atten)
        conc=h_seq_atten
        conc=self.dropout_fc(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout_fc(conc)
        outputs = self.out(conc)
        outputs=(outputs,)
        
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(outputs[0].view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
           
            if(self.train_att):
                loss_atts = self.lam * masked_cross_entropy(att,attention_vals,attention_mask)
                loss = loss+loss_atts
            outputs = (loss,) + outputs     
       
        outputs= outputs+(att,)
        return outputs



class MEDICAL_BiAtt_RNN(BiRNN):
    def __init__(self,args,embeddings,return_att):
        super().__init__(args,embeddings)
       
        self.seq_attention = Attention_BiRNN_Layer(self.hidden_size * 2, args['max_length'], self.drop_hidden)
        #self.seq_analyse = Analyse_Layer(self.embedsize, args['max_length']) 

        self.linear_1 = nn.Linear(self.hidden_size * 2, args["batch_size"])
        self.linear_2 = nn.Linear(self.hidden_size * 2, args["batch_size"])
        self.relu = nn.ReLU()
        self.out_1 = nn.Linear(args["batch_size"], args["num_classes"])
        self.out_2 = nn.Linear(args["batch_size"], args["num_classes"])
        self.out = nn.Linear(args["num_classes"]*2, args["num_classes"])
        self.return_att = False
        self.lam = args['att_lambda']
        self.train_att =args['train_att']
        self.max_length = args['max_length']
        
    def concatenate_column_index(self, x, y):
        nb_feature = self.hidden_size * 2
        p = torch.zeros(x.shape[0], x.shape[1] + y.shape[1]) 
        for i in range(nb_feature):
            p[:, i] = x[:, i]
            p[:, i+1] = y[:, i]  
        return p

    def sum_hidden_state_birnn(self, x):
        p = torch.zeros(x.shape[1], self.hidden_size)
        output = torch.zeros(x.shape[0], x.shape[1], self.hidden_size) 
        for i in range(x.shape[0]):
            temp = x[i]
            for j in range(self.hidden_size): 
                p[:, j] = temp[:, j] + temp[:, j+1]
            output[i] = p
        return output

    def forward(self, input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(input_ids.shape[0],input_ids.shape[1],self.embedsize)

        h_seq, _ = self.seq_model(h_embedding)
        #h_seq = self.sum_hidden_state_birnn(h_seq)

        if(debug):
            print("output",h_seq.shape)
        
        h_seq_atten_1, h_seq_atten_2, att = self.seq_attention(h_seq, attention_mask)
        
        #h_seq_anlyse = h_embedding * torch.unsqueeze(att, -1)
        #h_seq_anlyse = self.seq_analyse(h_seq_anlyse)

        if(debug):
            print("h_seq_atten",h_seq_atten.shape)

        #h_seq_atten_anlyse = torch.cat((h_seq_atten, h_seq_anlyse), dim=1)
        conc_1 = h_seq_atten_1
        conc_2 = h_seq_atten_2
        conc_1 = self.dropout_fc(conc_1)
        conc_2 = self.dropout_fc(conc_2)
        conc_1 = self.relu(self.linear_1(conc_1))
        conc_2 = self.relu(self.linear_2(conc_2))
        conc_1 = self.dropout_fc(conc_1)
        conc_2 = self.dropout_fc(conc_2)
        outputs_1 = self.out_1(conc_1)
        outputs_2 = self.out_2(conc_2)
        outputs = torch.cat((outputs_1, outputs_2), dim=1)
        outputs = self.out(outputs)
        outputs=(outputs,)
        
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(outputs[0].view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
           
            if(self.train_att):
                loss_atts = self.lam * masked_cross_entropy(att,attention_vals,attention_mask)
                loss = loss+loss_atts
            outputs = (loss,) + outputs
       
       
        outputs= outputs+(att,)
        return outputs

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f
"""
class IntraAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights, is_cuda):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.bilstms = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		self.sq = nn.Linear(embedding_dim, embedding_dim, bias=False)
		self.fc = nn.Linear(2*hidden_dim+embedding_dim, output_dim)
		self.is_cuda = is_cuda

	def forward(self, input_sequences):
		embeddings = self.embedding(input_sequences)		
		s = self.bilinear_layer(embeddings)
		embeddings = embeddings.permute(1, 0, 2)
		output, hidden = self.bilstms(embeddings)
		x = torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), dim=1)
		r = torch.cat((s,x), dim=1)
		fc = self.fc(r)
		return fc

	def bilinear_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		mask = torch.ones(max_sequence_length, max_sequence_length)
		if self.is_cuda:
			mask = mask.cuda()
		mask = mask - torch.diag(torch.diag(mask))
		s = self.sq(embeddings)
		s = torch.bmm(s, embeddings.permute(0,2,1))
		s = s*mask
		s = f.avg_pool1d(s, s.size()[2]).squeeze(2)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		s = torch.bmm(s, embeddings)
		s = s.squeeze(1)
		return s

class BiCoAtt_RNN(BiRNN):
    def __init__(self,args,embeddings,return_att):
        super().__init__(args,embeddings)
        if(args['attention']=='sigmoid'):
             self.seq_attention = Attention_LBSA_sigmoid(self.hidden_size * 2, args['max_length'])
        else:
             self.seq_attention = Attention_LBSA(self.hidden_size * 2, args['max_length'])
        self.linear = nn.Linear(self.hidden_size * 2 + self.embedsize, args["batch_size"])
        self.relu = nn.ReLU()
        self.out = nn.Linear(args["batch_size"], args["num_classes"])
        self.return_att=False
        self.lam=args['att_lambda']
        self.train_att =args['train_att']
        #element add
        self.sq = nn.Linear(self.embedsize, self.embedsize, bias=False)
        self.sq1 = nn.Linear(self.embedsize, self.embedsize, bias=False)
        self.kl = args["embed_size"]
        
        
    def forward(self, input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        ###print("embedding_size ", self.kl)
        ###print("hidden_size ", self.hidden_size)
        ###print("input_ids.shape[0] ", input_ids.shape[0], " input_ids.shape[1] ", input_ids.shape[1]) 
        h_embedding = self.embedding(input_ids)
        ###print("h_embedding ", h_embedding.shape)
        s, att = self.bilinear_layer(h_embedding)#add
        #
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(input_ids.shape[0],input_ids.shape[1],self.embedsize)
        
        h_embedding = h_embedding.permute(1, 0, 2)
        ###print("h_embedding1 ", h_embedding.shape)
        
        h_seq, _ = self.seq_model(h_embedding)#BiRNN model
        #associate h_seq and s(r in article) before to pass at different layers(attention layer and fully connected layers)
        #embeddings = self.embedding(input_sequences)		
		#s = self.bilinear_layer(embeddings)
		#embeddings = embeddings.permute(1, 0, 2)
		#output, hidden = self.bilstms(embeddings)
		#x = torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), dim=1)
		#r = torch.cat((s,x), dim=1)
		#fc = self.fc(r)
        #end add
        ###print("output ",h_seq.shape)
        ###print("s ", s.shape)    
        print("h_seq ", h_seq.shape)
        x = torch.cat((h_seq[-1, :, :self.hidden_size], h_seq[0, :, self.hidden_size:]), dim=1)#add
        ###print("x ", x.shape)
        r = torch.cat((s,x), dim=1)#add
        ###print("r ", r.shape)

        conc=r
        conc=self.dropout_fc(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout_fc(conc)
        outputs = self.out(conc)
        outputs=(outputs,)

        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(outputs[0].view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
           
            if(self.train_att):
                loss_atts = self.lam*masked_cross_entropy(att,attention_vals,attention_mask)
                loss = loss+loss_atts
            outputs = (loss,) + outputs
       
       
        outputs= outputs+(att,)
        return outputs

    def bilinear_layer(self, embeddings):
        max_sequence_length = embeddings.shape[1]
        mask = torch.ones(max_sequence_length, max_sequence_length)
        #if self.is_cuda:
        #	mask = mask.cuda()
        ###mask = mask - torch.diag(torch.diag(mask))
        #print(mask)
        #sys.exit(0)
        s = self.relu(self.sq(embeddings))
        s = self.dropout_fc(s)
        s = self.relu(self.sq1(s))
        s = torch.bmm(s, embeddings.permute(0,2,1))
        s = s*mask
        #
        g = s
        g = g[0]
        ###g = g[1:]
        #
        s = f.avg_pool1d(s, s.size()[2]).squeeze(2)
        s = f.softmax(s, dim=1)
        s = s.unsqueeze(dim=1)
        s = torch.bmm(s, embeddings)
        ###print("s-bil ", s.shape)
        s = s.squeeze(1)
        return s, g


"""
import sys
import math

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0., emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0., maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class BiCoAtt_RNN(BiRNN):
    def __init__(self,args,embeddings,return_att):
        super().__init__(args,embeddings)
        if(args['attention']=='sigmoid'):
             self.seq_attention = CoAttention_LBSA_sigmoid(self.hidden_size * 2, args['max_length'])
             self.seq_attention_word = CoAttention_LBSA_sigmoid(self.embedsize, args['max_length'])
        else:
             self.seq_attention = CoAttention_LBSA(self.hidden_size * 2, args['max_length'])
             self.seq_attention_word = CoAttention_LBSA(self.embedsize, args['max_length'])
        self.linear = nn.Linear(self.hidden_size * 2 + self.embedsize, args["batch_size"])
        self.relu = nn.ReLU()
        self.out = nn.Linear(args["batch_size"], args["num_classes"])
        self.return_att=False
        self.lam=args['att_lambda']
        self.train_att =args['train_att']
        self.pos_encoder = PositionalEncoding(self.embedsize, self.drop_embed)

        
        
    def forward(self, input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(input_ids.shape[0],input_ids.shape[1],self.embedsize)
        
        h_seq, _ = self.seq_model(h_embedding)
        if(debug):
            print("output",h_seq.shape)

        h_embedding = self.pos_encoder(h_embedding)
        
        h_seq_atten, att_2 = self.seq_attention(h_seq, attention_mask)
        h_seq_atten_1, att_1 = self.seq_attention_word(h_embedding, attention_mask)
        h_seq_atten = torch.cat((h_seq_atten, h_seq_atten_1), dim=1)
        att_2 = att_2 + att_1
        sigmoid = nn.Sigmoid()
        att = sigmoid(att_2) 

        if(debug):
            print("h_seq_atten",h_seq_atten.shape)

        conc=h_seq_atten
        conc=self.dropout_fc(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout_fc(conc)
        outputs = self.out(conc)
        outputs=(outputs,)
        
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(outputs[0].view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
           
            if(self.train_att):
                loss_atts = self.lam*masked_cross_entropy(att,attention_vals,attention_mask)
                loss = loss+loss_atts
            outputs = (loss,) + outputs
       
       
        outputs= outputs+(att,)
        return outputs
    
    def bilinear_layer(self, embeddings):
        max_sequence_length = embeddings.shape[1]
        mask = torch.ones(max_sequence_length, max_sequence_length)
        #if self.is_cuda:
        #	mask = mask.cuda()
        ###mask = mask - torch.diag(torch.diag(mask))
        #print(mask)
        #sys.exit(0)
        sigmoid = nn.Sigmoid()
        s = sigmoid(torch.mm(embeddings, self.sq))
        #s = sigmoid(torch.mm(s, self.sq1))
        s = torch.bmm(s, embeddings.permute(0,2,1))
        s = s*mask
        #
        g = s
        g = g[0]
        ###g = g[1:]
        #
        s = f.avg_pool1d(s, s.size()[2]).squeeze(2)
        s = f.softmax(s, dim=1)
        s = s.unsqueeze(dim=1)
        s = torch.bmm(s, embeddings)
        ###print("s-bil ", s.shape)
        s = s.squeeze(1)
        return s, g

    def simplified_intra_attention_layer(self, embeddings):
        # (WW)
        max_sequence_length = embeddings.shape[1]
        mask = torch.ones(max_sequence_length,max_sequence_length)
        #mask = mask - torch.diag(torch.diag(mask))
        e = embeddings.permute(0,2,1)
        s = torch.bmm(embeddings, e)
        #(bts, max_sequence_length, max_sequence_length)
        s = s*mask
        g = s
        g = g[0]
        #doing masking to make values where word pairs are same(i == j), zero
        s = f.avg_pool1d(s, s.size()[2]).squeeze(2)
        #(bts, max_sequence_length)
        s = f.softmax(s, dim=1)
        s = s.unsqueeze(dim=1)
        #(bts, 1, max_sequence_length)
        s = torch.bmm(s, embeddings)
        #(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
        s = s.squeeze(1)
        return s, g

    def singular_intra_attention_layer(self, embeddings):
        # from Tay paper on sarcasm detection
        max_sequence_length = embeddings.shape[1]
        batch_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[2]

        #(bts, msl, dim)
        b = embeddings
        d = embeddings
        b = b.repeat(1,1,max_sequence_length)
        #(bts, max_sequence_length, max_sequence_length*embedding_dim)
        b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
        d = d.unsqueeze(1)
        #(bts, 1, max_sequence_length, embedding_dim)
        d = d.repeat(1,max_sequence_length,1,1)
        #(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
        concat = torch.cat((b,d), dim=3)
        #batch_size, max_sequence_length, max_sequence_length, 2*embedding_dim

        '''Lets assume that there are 2 words and each word embedding has 3 dimension
        So embedding matrix would look like this
        [[----1----]
        [----2----]].  (2*3)

        b =
        ----1----,----1----
        ----2----,----2----


        d=
        ----1----,----2----
        ----1----,----2----

        Now if you concatenate both, we get all the combinations of word pairs'''


        s = self.dropout_fc(self.w(concat))
        #batch_size, max_sequence_length, max_sequence_length, 1
        s = s.squeeze(3)
        #batch_size, max_sequence_length, max_sequence_length
        mask = torch.ones(max_sequence_length,max_sequence_length)
        #if self.is_cuda:
        #    mask = mask.cuda()
        #mask = mask - torch.diag(torch.diag(mask))
        #(bts, max_sequence_length, max_sequence_length)
        s = s*mask
        
        g = s
        g = g[0]

        s = f.max_pool1d(s, s.size()[2]).squeeze(2)
        #(bts, max_sequence_length)
        s = f.softmax(s, dim=1)
        s = s.unsqueeze(dim=1)
        #(bts, 1, max_sequence_length)
        s = self.dropout_fc(torch.bmm(s, embeddings))
        #(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
        s = s.squeeze(1)
        return s, g

    def co_attention_layer(self, embeddings):
        # model from BIDAF paper
        max_sequence_length = embeddings.shape[1]
        batch_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[2]

        #(bts, msl, dim)
        b = embeddings
        d = embeddings
        b = b.repeat(1,1,max_sequence_length)
        #(bts, max_sequence_length, max_sequence_length*embedding_dim)
        b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
        d = d.unsqueeze(1)
        #(bts, 1, max_sequence_length, embedding_dim)
        d = d.repeat(1,max_sequence_length,1,1)
        #(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
        bd = b*d
        concat = torch.cat((b,d,bd), dim=3)
        #batch_size, max_sequence_length, max_sequence_length, 3*embedding_dim

        '''Lets assume that there are 2 words and each word embedding has 3 dimension
        So embedding matrix would look like this
        [[----1----]
        [----2----]].  (2*3)

        b =
        ----1----,----1----
        ----2----,----2----


        d=
        ----1----,----2----
        ----1----,----2----

        Now if you concatenate both, we get all the combinations of word pairs'''


        s = self.wCo(concat)
        #batch_size, max_sequence_length, max_sequence_length, 1
        s = s.squeeze(3)
        #batch_size, max_sequence_length, max_sequence_length
        mask = torch.ones(max_sequence_length,max_sequence_length)
        #if self.is_cuda:
        #	mask = mask.cuda()
        #mask = mask - torch.diag(torch.diag(mask))
        #(bts, max_sequence_length, max_sequence_length)
        s = s*mask
        print("s ", s.shape)
        g = s
        g = g[0]
        #s = f.max_pool1d(s, s.size()[2]).squeeze(2)
        s = self.Wmy(s)
        print("s squeeze ", s.shape)
        sys.exit(0)
        ##s = s.squeeze(2)
        #(bts, max_sequence_length)
        s = f.softmax(s, dim=1)
        
        ##s = s.unsqueeze(dim=1)
        #(bts, 1, max_sequence_length)
        s = torch.bmm(s, embeddings)
        print("s ", s.shape)
        #(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
        s = s.squeeze(1)
        print("s squeeze ", s.shape)
        sys.exit(0)
        #print("s ", s.shape)
        #sys.exit(0)
        return s, g

    def multi_dimensional_intra_attention_layer(self, embeddings):
        # from Tay paper on sarcasm detection
        max_sequence_length = embeddings.shape[1]
        batch_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[2]

        #(bts, msl, dim)
        b = embeddings
        d = embeddings
        b = b.repeat(1,1,max_sequence_length)
        #(bts, max_sequence_length, max_sequence_length*embedding_dim)
        b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
        d = d.unsqueeze(1)
        #(bts, 1, max_sequence_length, embedding_dim)
        d = d.repeat(1,max_sequence_length,1,1)
        #(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
        concat = torch.cat((b,d), dim=3)
        #batch_size, max_sequence_length, max_sequence_length, 2*embedding_dim

        '''Lets assume that there are 2 words and each word embedding has 3 dimension
        So embedding matrix would look like this
        [[----1----]
        [----2----]].  (2*3)

        b =
        ----1----,----1----
        ----2----,----2----


        d=
        ----1----,----2----
        ----1----,----2----

        Now if you concatenate both, we get all the combinations of word pairs

        ----1--------1----,----1--------2----
        ----2--------1----,----2--------2----

        '''

        s = self.Wq(concat)
        #batch_size, max_sequence_length, max_sequence_length, par
        s = f.relu(s)
        s = self.Wp(s)
        #batch_size, max_sequence_length, max_sequence_length, 1
        s = s.squeeze(3)
        #batch_size, max_sequence_length, max_sequence_length
        mask = torch.ones(max_sequence_length,max_sequence_length)
        if self.is_cuda:
            mask = mask.cuda()
        mask = mask - torch.diag(torch.diag(mask))
        #s = torch.bmm(embeddings, embeddings.permute(0,2,1))
        #(bts, max_sequence_length, max_sequence_length)
        s = s*mask
        s = f.max_pool1d(s, s.size()[2]).squeeze(2)
        #(bts, max_sequence_length)
        s = f.softmax(s, dim=1)
        s = s.unsqueeze(dim=1)
        #(bts, 1, max_sequence_length)
        s = torch.bmm(s, embeddings)
        #(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
        s = s.squeeze(1)
        return s

class BiGAT_RNN(BiRNN):
    def __init__(self,args,embeddings,return_att):
        super().__init__(args,embeddings)
        
        self.CORA_NUM_INPUT_FEATURES = self.hidden_size*2
        self.CORA_NUM_CLASSES = 1
        self.CORA_NUM_INPUT = args['max_length']

        self.config = {
                "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
                "num_heads_per_layer": [8, 1],
                "num_features_per_layer": [self.CORA_NUM_INPUT_FEATURES, 8, self.CORA_NUM_CLASSES],
                "add_skip_connection": False,  # hurts perf on Cora
                "bias": True,  # result is not so sensitive to bias
                "dropout": 0.6,  # result is sensitive to dropout
                "layer_type": LayerType.IMP3,  # fastest implementation enabled by default
                "num_nodes": args["max_length"]
        }

        self.gat = GAT(
                num_of_layers = self.config['num_of_layers'],
                num_heads_per_layer = self.config['num_heads_per_layer'],
                num_features_per_layer = self.config['num_features_per_layer'],
                num_nodes = self.config['num_nodes'],
                add_skip_connection = self.config['add_skip_connection'],
                bias = self.config['bias'],
                dropout = self.config['dropout'],
                layer_type = self.config['layer_type'],
                log_attention_weights=False # no need to store attentions, used only in playground.py for visualizations      
        )
        if(args['attention']=='sigmoid'):
             self.seq_attention = Attention_LBSA_sigmoid(self.hidden_size * 2, args['max_length'])
        else:
             self.seq_attention = Attention_LBSA(self.hidden_size * 2, args['max_length'])

#        self.linear = nn.Linear(self.hidden_size * 2 + self.embedsize, args["batch_size"])
        self.linear = nn.Linear(self.hidden_size * 2 , args["batch_size"])
        self.relu = nn.ReLU()
        self.out = nn.Linear(args["batch_size"], args["num_classes"])
        self.return_att=False
        self.lam=args['att_lambda']
        self.train_att =args['train_att']
        self.max_length = args['max_length']
        
        context = torch.zeros(self.max_length, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector = nn.Parameter(context)
        
    def forward(self, input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(input_ids.shape[0],input_ids.shape[1],self.embedsize)

        h_seq, _ = self.seq_model(h_embedding)
        if(debug):
            print("output",h_seq.shape)
        
#        h_seq_atten, att_1 = self.seq_attention(h_seq_1, attention_mask)
##        h_seq = h_embedding
        out_put_gat = torch.zeros(h_seq.shape[0], self.CORA_NUM_INPUT, self.CORA_NUM_CLASSES)
        att = torch.zeros(h_seq.shape[0], self.CORA_NUM_INPUT, self.CORA_NUM_INPUT)
        for i in range(h_seq.shape[0]):
            node_features = h_seq[i]
            edge_index = torch.transpose(torch.ones(node_features.shape[0], node_features.shape[0]).nonzero(), 0, 1)
            graph_data = (node_features, edge_index, None)  

            out_put_gat[i], _, att[i] = self.gat(graph_data)
            #node_features =   torch.arange(self.CORA_NUM_INPUT * self.CORA_NUM_INPUT_FEATURES, dtype=torch.floawt32).view(self.CORA_NUM_INPUT, self.CORA_NUM_INPUT_FEATURES)
        #print("att ", att.shape, " \n", att)
        #print("mask ", attention_mask.shape, " \n", attention_mask)
        #print("out_put_gat ", out_put_gat.shape, " \n", out_put_gat)
        #sys.exit(0)

        att = att.contiguous().view(-1, 2*self.hidden_size)
        att = torch.mm(att, self.context_vector)
        att = att.view(-1, self.max_length)


        att[~attention_mask] = float('-inf')
        att = torch.softmax(att, dim=1)
    
        weighted_input = h_seq * torch.unsqueeze(att, -1)

        #att = att * attention_mask
        #h_seq_atten,att = self.seq_attention(h_seq,attention_mask)
        if(debug):
            print("h_seq_atten",h_seq_atten.shape)
        
        out_put_gat = torch.max(weighted_input, 1)[0]
#        out_put_gat = torch.cat((out_put_gat, h_seq_atten), dim=1)
#        out_put_gat = out_put_gat.contiguous().view(-1, self.max_length * self.CORA_NUM_CLASSES)
        conc=out_put_gat
        conc=self.dropout_fc(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout_fc(conc)
        outputs = self.out(conc)
        outputs=(outputs,)
        
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(outputs[0].view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
           
            if(self.train_att):
                loss_atts = self.lam*masked_cross_entropy(att,attention_vals,attention_mask)
                loss = loss+loss_atts
            outputs = (loss,) + outputs
       
       
        outputs= outputs+(att,)
        return outputs

class BiStackAtt_RNN(BiRNN):
    def __init__(self,args,embeddings,return_att):
        super().__init__(args,embeddings)
        if(args['attention']=='sigmoid'):
             self.seq_attention = Stack_Attention(5, self.hidden_size * 2, args['max_length'])             
        else:
             self.seq_attention = Stack_Attention(5, self.hidden_size * 2, args['max_length'])
        self.linear = nn.Linear(self.hidden_size * 2, args["batch_size"])
        self.relu = nn.ReLU()
        self.out = nn.Linear(args["batch_size"], args["num_classes"])
        self.return_att=False
        self.lam=args['att_lambda']
        self.train_att =args['train_att']
        
        
    def forward(self, input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(input_ids.shape[0],input_ids.shape[1],self.embedsize)

        h_seq, _ = self.seq_model(h_embedding)
        if(debug):
            print("output",h_seq.shape)
        data = h_seq, attention_mask, None
        h_seq_atten, _, att = self.seq_attention(data)
        if(debug):
            print("h_seq_atten",h_seq_atten.shape)
        
        conc=h_seq_atten
        conc=self.dropout_fc(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout_fc(conc)
        outputs = self.out(conc)
        outputs=(outputs,)
        
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(outputs[0].view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
           
            if(self.train_att):
                p = masked_cross_entropy(att,attention_vals,attention_mask)
                loss_atts = self.lam*p
                #;print("Lc = ", loss, " La = ", p)
                loss = loss+loss_atts
            outputs = (loss,) + outputs
       
       
        outputs= outputs+(att,)
        return outputs



if __name__ == '__main__':
    args_dict = {
        "batch_size":10,
        "hidden_size":256,
        "epochs":10,
        "embed_size":300,
        "drop":0.1,
        "learning_rate":0.001,
        "vocab_size":10000,
        "num_classes":3,
        "embeddings":np.array([]),
        "seq_model":"lstm",
        "drop_embed":0.1,
        "drop_fc":0.1,
        "drop_hidden":0.1,
        "train_embed":False
        
        }
#     BiRNN(args_dict)
#     BiAtt_RNN(args_dict)
#     BiSCRAT_RNN(args_dict)
    CNN_GRU(args_dict)
