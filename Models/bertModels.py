from transformers.modeling_bert import *
from .utils import masked_cross_entropy
import torch
import torch.nn as nn
import sys 

class SC_weighted_BERT(BertPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights=params['weights']
        self.train_att= params['train_att']
        self.lam = params['att_lambda']
        self.num_sv_heads=params['num_supervised_heads']
        self.sv_layer = params['supervised_layer_pos']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.softmax=nn.Softmax(config.num_labels)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        attention_vals=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        #logits = self.softmax(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is not None:
            loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            loss_logits =  loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
            if(self.train_att):
                
                loss_att=0
                for i in range(self.num_sv_heads):
                    attention_weights=outputs[1][self.sv_layer][:,i,0,:]
                    loss_att +=self.lam*masked_cross_entropy(attention_weights,attention_vals,attention_mask)
                loss = loss + loss_att
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BiAtt_BertModels(BertPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights=params['weights']
        self.train_att= params['train_att']
        self.lam = params['att_lambda']
        self.num_sv_heads=params['num_supervised_heads']
        self.sv_layer = params['supervised_layer_pos']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.drop_embed_x = 0.2
        self.dropout_embed = nn.Dropout2d(self.drop_embed_x)
        #self.softmax=nn.Softmax(config.num_labels)
        self.output_dim = 1
        self.seq_model = nn.LSTM(config.hidden_size, self.output_dim, bidirectional=True, num_layers=1, batch_first=True, dropout=config.hidden_dropout_prob)
        self.bias_att = nn.Parameter(torch.zeros(1, 1))
        self.init_weights()        

    def forward(self,
        input_ids=None,
        attention_mask=None,
        attention_vals=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        
        # add hidden states and attention if they are here
        ##################
        #outputs.last_hidden_state
        last_hidden_state = outputs[0]
        shape_token = last_hidden_state.shape[1]
        last_hidden_state = last_hidden_state[0:shape_token-1, : ]
        
        y = torch.squeeze(self.dropout_embed(torch.unsqueeze(last_hidden_state, 0))).view(last_hidden_state.shape[0], last_hidden_state.shape[1], last_hidden_state.shape[2])    
        att, _ = self.seq_model(y)
        
        att = torch.sum(att, dim=2)
        att = att + self.bias_att 
        #a = a.squeeze(2)
        att[~attention_mask] = float('-inf')
        att = torch.softmax(att, dim=1)
        
        weighted_input = last_hidden_state * torch.unsqueeze(att, -1)
        pooled_output, attention_weights = torch.max(weighted_input, dim=1)[0], att        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        #logits = self.softmax(logits)

        outputs = (logits,) + outputs[2:]
        ####################

        if labels is not None:
            loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            loss_logits =  loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_logits
            if(self.train_att): 
                loss_att = self.lam*masked_cross_entropy(attention_weights,attention_vals,attention_mask)
                loss = loss + loss_att
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)