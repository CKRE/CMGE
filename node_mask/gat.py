import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, explain=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)
        self.explain = explain

    def forward(self, h, g, explain=False):
        h = self.layer1(h, g, explain)
        h = F.elu(h)
        h = self.layer2(h, g, explain)
        return h


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, h, g, explain=False):
        head_outs = [attn_head(h, g, explain) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1) #  src, dst代表双向的两条边，一个过去一个回来。
        a = self.attn_fc(z2) # 一种加法注意力机制
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        if self.explain:
            return {'z': edges.src['z'], 'e': edges.data['e'], 'mask': edges.data['edge_mask']}
        else:
            return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        if self.explain:
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        else:
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h, g, explain=False):
        self.explain = explain
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        #return g.ndata.pop('h')
        # update_all = send(self, self.edges(), message_func) + recv(self, self.nodes(), reduce_func, apply_node_func)
        return g.ndata['h']