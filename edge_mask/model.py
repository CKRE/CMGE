import torch
import dgl
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import networkx as nx
import sklearn
from transformers import *
from dgl import DGLGraph
from gat import GAT
from math import sqrt


class HGModel(nn.Module):

    coeffs = {
        'edge_size': 0.05,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_desc_label, num_entity_label, num_graph_label):
        super(HGModel, self).__init__()
        # self.BertEncode = BertModel.from_pretrained('./model/pretrain_model/')
        tokenizer_class = BertTokenizer
        pretrained_weights = 'bert-base-chinese'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        vocab_size = tokenizer.vocab_size
        self.dot_attn = ScaledDotProductAttention(attention_dropout=0.1)
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, 300)
        self.gru = nn.GRU(300, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.end_node = nn.GRU(1024, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.GAT = GAT(in_dim, 512, out_dim, num_heads)
        self.fc_desc = nn.Sequential(nn.Linear(out_dim, out_dim),
                                     nn.ReLU(),
                                     nn.Linear(out_dim, num_desc_label))
        self.fc_entity = nn.Sequential(nn.Linear(out_dim, out_dim),
                                       nn.ReLU(),
                                       nn.Linear(out_dim, num_entity_label))
        self.fc_g = nn.Sequential(nn.Linear(out_dim, out_dim),
                                  nn.ReLU(),
                                  nn.Linear(out_dim, num_graph_label))
        self.p_d = nn.Linear(1024, 128)
        self.p_e = nn.Linear(1024, 128)
        self.p_g = nn.Linear(1024, 128)
        self.ce_loss = nn.CrossEntropyLoss()
        self.edge_del = edge_del(2)
        self.m = nn.LayerNorm(128)


    def forward(self, input_data, dataset=None, epochs=None, explain=False):
        age = self.gru(self.embed(input_data['age'].cuda()))[1].transpose(0, 1).reshape(len(input_data['age']), self.hidden_dim * 4)
        gender = self.gru(self.embed(input_data['gender'].cuda()))[1].transpose(0, 1).reshape(len(input_data['gender']), self.hidden_dim * 4)
        desc = self.gru(self.embed(input_data['desc'].cuda()))[1].transpose(0, 1).reshape(len(input_data['desc']), self.hidden_dim * 4)
        raw = self.gru(self.embed(input_data['raw'].cuda()))[1].transpose(0, 1).reshape(len(input_data['raw']), self.hidden_dim * 4)
        entity = self.gru(self.embed(input_data['entity'].cuda()))[1].transpose(0, 1).reshape(len(input_data['entity']), self.hidden_dim * 4)
        raw_start = torch.cat((torch.tensor([0]), input_data['raw_num'][:-1].cumsum(dim=0)), dim=0)
        entity_start = torch.cat((torch.tensor([0]), input_data['entity_num'][:-1].cumsum(dim=0)), dim=0)
        desc_start = torch.cat((torch.tensor([0]), input_data['desc_num'][:-1].cumsum(dim=0)), dim=0)
        raw = [raw.narrow(0, s, l) for s, l in zip(raw_start, input_data['raw_num'])]
        desc = [desc.narrow(0, s, l) for s, l in zip(desc_start, input_data['desc_num'])]
        entity = [entity.narrow(0, s, l) for s, l in zip(entity_start, input_data['entity_num'])]
        graph = input_data['graph']
        desc_loss_all = 0.0
        entity_loss_all = 0.0
        graph_loss_all = 0.0
        desc_acc_num = 0
        entity_acc_num = 0
        graph_acc_num = 0
        g_out = []
        pred_graph = []
        pred_desc = []
        pred_entity = []
        for index, g in enumerate(graph):
            g = g.to('cuda')
            # end_node_feature = self.dot_attn(raw_hidden[index], raw_hidden[index], raw_hidden[index])
            end_node_feature = self.end_node(raw[index].unsqueeze(0))[1].transpose(0, 1).reshape(1, self.hidden_dim * 4)
            feature = torch.cat((age[index: index + 1], gender[index: index + 1], raw[index], desc[index], entity[index], end_node_feature), dim=0)
            attn_output = self.GAT(feature, g)
            
            g_out.append(g)
            desc_output = attn_output[2 + raw[index].size(0) : 2 + raw[index].size(0) + desc[index].size(0)] + self.p_d(desc[index])
            entity_output = attn_output[2 + raw[index].size(0) + desc[index].size(0): -1] + self.p_e(entity[index])
            graph_output = attn_output[-1:] + self.p_g(end_node_feature)
            desc_output = self.m(desc_output)
            entity_output = self.m(entity_output)
            graph_output = self.m(graph_output)
            #desc_pred = self.fc_desc2(self.fc_desc1(desc_output)) # 这里可能要加一维
            #entity_pred = self.fc_entity2(self.fc_entity1(entity_output))
            #graph_pred = self.fc_g2(self.fc_g1(graph_output))
            desc_pred = self.fc_desc(desc_output)
            entity_pred = self.fc_entity(entity_output)
            graph_pred = self.fc_g(graph_output)
            entity_loss = self.ce_loss(entity_pred, torch.tensor(input_data['label'][index]).cuda())
            desc_loss = self.ce_loss(desc_pred, torch.tensor(input_data['semantic'][index]).cuda())
            graph_loss = self.ce_loss(graph_pred, torch.tensor([int(input_data['type'][index]) - 1]).cuda())
            desc_pred = torch.argmax(F.log_softmax(desc_pred, dim=1), dim=1)
            entity_pred = torch.argmax(F.log_softmax(entity_pred, dim=1), dim=1)
            graph_pred = torch.argmax(F.log_softmax(graph_pred, dim=1), dim=1)
            desc_acc_num += (desc_pred == torch.tensor(input_data['semantic'][index]).cuda()).sum().item()
            entity_acc_num += (entity_pred == torch.tensor(input_data['label'][index]).cuda()).sum().item()
            graph_acc_num += (graph_pred == torch.tensor([int(input_data['type'][index]) - 1]).cuda()).sum().item()
            desc_loss_all += desc_loss
            entity_loss_all += entity_loss
            graph_loss_all += graph_loss
            pred_graph.append(graph_pred)
            pred_desc.append(desc_pred)
            pred_entity.append(entity_pred)
        return {'desc_loss': desc_loss_all, 'entity_loss': entity_loss_all, 'desc_node': input_data['desc'].size(0),
                'entity_node': input_data['entity'].size(0), 'desc_acc': desc_acc_num, 'entity_acc': entity_acc_num, 
                'graph_loss': graph_loss_all, 'graph_acc': graph_acc_num, 'desc_pred': pred_desc, 'entity_pred': pred_entity,
                'graph_pred': pred_graph, 'graph': g_out}
        
                
    
    def __set_masks__(self, graph, init='normal'):
        N, F, E = graph.number_of_nodes(), graph.ndata['h'].size(1), graph.number_of_edges()

        std = 0.1
        self.node_feat_mask = nn.Parameter(torch.randn(F) * 0.1)

        std = nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = nn.Parameter(torch.randn(E) * std)

    def __clear_masks__(self):
        self.node_feat_mask = None
        self.edge_mask = None

    def __loss__(self, log_logits, pred_label):
        EPS = 1e-15
        loss = -log_logits[0, pred_label]

        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """

        :param q:[B,l,d]
        :param k: [B, l, d]
        :param v: [B, l , d]=k
        :param scale:
        :param attn_mask:
        :return:
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class edge_del(nn.Module):
    def __init__(self, max_edges_num):
        super(edge_del, self).__init__()
        self.max_edges_num = max_edges_num

    def message_func(self, edges):
        pass
        

    def edge_normal(self, edges):
        pass

    def forward(self, g, desc_start, desc_end):
        for dst_node in range(desc_start, desc_end): # dst_node 是当前计算的点
            
            src = g.edges()[0]
            dst = g.edges()[1]
            src_node = g.adjacency_matrix()[dst_node].to_dense().nonzero().tolist() # src_node 是远处的点
            edges_value = []
            for node in src_node:
                edge_value = g.edges[node[0], dst_node].data['e'] # 从远处到当前点的边
                if len(edge_value) == 0:
                    for i in range(src.size(0)):
                        if src[i].item() == node[0] and dst[i].item() == dst_node:
                            edge_value = g.edata['e'][i]
                            break
                edges_value.append(edge_value)
            edges_value = F.softmax(torch.tensor(edges_value), dim=0)
            if edges_value.size(0) > self.max_edges_num:
                cut_index = edges_value.topk(2, largest=False)[1] # 远处的点
                for index in cut_index:
                    cut_node = src_node[index][0] # 远处的点
                    for i in range(src.size(0)):
                        if src[i].item() == cut_node and dst[i].item() == dst_node:    
                            g.remove_edges(i)
                            break
        return g