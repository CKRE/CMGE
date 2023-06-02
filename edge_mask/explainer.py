import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from copy import copy
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class GNNExplainer(nn.Module):

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=200, lr=0.001, num_hops=2, log=True):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.log = log
        self.edge_mask = None
        self.node_feat_mask = None

    def explain_node(self, input_data):
        self.model.eval()
        with torch.no_grad():
            age = self.model.gru(self.model.embed(input_data['age'].cuda()))[1].transpose(0, 1).reshape(len(input_data['age']), self.model.hidden_dim * 4)
            gender = self.model.gru(self.model.embed(input_data['gender'].cuda()))[1].transpose(0, 1).reshape(len(input_data['gender']), self.model.hidden_dim * 4)
            desc = self.model.gru(self.model.embed(input_data['desc'].cuda()))[1].transpose(0, 1).reshape(len(input_data['desc']), self.model.hidden_dim * 4)
            raw = self.model.gru(self.model.embed(input_data['raw'].cuda()))[1].transpose(0, 1).reshape(len(input_data['raw']), self.model.hidden_dim * 4)
            entity = self.model.gru(self.model.embed(input_data['entity'].cuda()))[1].transpose(0, 1).reshape(len(input_data['entity']), self.model.hidden_dim * 4)
            raw_start = torch.cat((torch.tensor([0]), input_data['raw_num'][:-1].cumsum(dim=0)), dim=0)
            entity_start = torch.cat((torch.tensor([0]), input_data['entity_num'][:-1].cumsum(dim=0)), dim=0)
            desc_start = torch.cat((torch.tensor([0]), input_data['desc_num'][:-1].cumsum(dim=0)), dim=0)
            raw = [raw.narrow(0, s, l) for s, l in zip(raw_start, input_data['raw_num'])]
            desc = [desc.narrow(0, s, l) for s, l in zip(desc_start, input_data['desc_num'])]
            entity = [entity.narrow(0, s, l) for s, l in zip(entity_start, input_data['entity_num'])]
            graph = input_data['graph'][0].to('cuda')
            graph.ndata['node_id'] = torch.tensor([i for i in range(graph.number_of_nodes())]).cuda()
            num_row = len(input_data['raw'])
            index = 0
            end_node_feature = self.model.end_node(raw[index].unsqueeze(0))[1].transpose(0, 1).reshape(1, self.model.hidden_dim * 4)
            feature = torch.cat((age[index: index + 1], gender[index: index + 1], raw[index], desc[index], entity[index], end_node_feature), dim=0)
            graph.ndata['h'] = feature
            output = self.model.GAT(graph.ndata['h'], graph)
            graph_output = output[-1:]
            row_log_logits = F.log_softmax(self.model.fc_g(graph_output), dim=1)
            pred_label = row_log_logits.argmax(dim=-1)
            num_row = len(input_data['raw'])
            num_desc = len(input_data['desc'])
            num_entity = len(input_data['entity'])

        graph.ndata['h'] = feature
        graph = dgl.node_subgraph(graph, [i for i in range(2 + raw[0].size(0), 2 + raw[0].size(0) + desc[0].size(0))] + [graph.number_of_nodes() - 1])
        # graph.remove_nodes([i for i in range(2 + num_row)] + [i for i in range(2 + num_row + num_desc, 2 + num_row + num_desc + num_entity)])

        self.__set_masks__(graph)
        self.cuda()
        sub_feature = graph.ndata['h']
        optimizer = torch.optim.Adam([self.edge_mask], lr=0.01)
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            graph.edata['edge_mask'] = self.edge_mask
            # graph.ndata['h'] = sub_feature * self.node_feat_mask.view(1, -1).sigmoid()
            graph.ndata['h'] = sub_feature
            graph_output = self.model.GAT(graph.ndata['h'], graph, explain=True)
            log_logits = F.log_softmax(self.model.fc_g(graph_output[-1:]), dim=1)
            loss = self.__loss__(log_logits, pred_label) # logits不是原来的logits, 原来的logits只是用来获取pred_label
            loss.backward()
            optimizer.step()
        
        node_feat_mask = self.node_feat_mask.sigmoid()
        edge_mask = self.edge_mask.sigmoid()
        self.__clear_masks__()

        return node_feat_mask, edge_mask, graph, pred_label


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
    
    def visualize_subgraph(self, graph, edge_mask, threshold=None, label=None, top=None, **kwargs):
        assert edge_mask.size(0) == graph.number_of_edges()
        
        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        values, indices = edge_mask.topk(int(top))
        
        graph = graph.cpu()
        graph.edata['edge_mask'] = edge_mask.cpu()
        graph = dgl.edge_subgraph(graph, indices.cpu())
        mapping = {k: i.item() for k, i in enumerate(graph.ndata['node_id'])}

        G = dgl.to_networkx(graph, node_attrs=['h', 'node_id'], edge_attrs=['edge_mask'])
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = 800
        node_kwargs['cmap'] = 'cool'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = 10

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle='-',
                    alpha=max(data['edge_mask'].item(), 0.1),
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle='arc3,rad=0.1'
                )
            )
        nx.draw_networkx_nodes(G, pos, node_color=[1 for i in range(graph.number_of_nodes())], **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)
        
        return ax, G

        
    
