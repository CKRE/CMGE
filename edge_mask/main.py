import os
import argparse
import torch
import sklearn
import pickle
import dgl
import networkx as nx
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import Medical_Data, data_loader, idx2desc, idx2entity, desc2idx
from model import HGModel
from collections import Counter
from explainer import GNNExplainer
import seaborn as sns


def model_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data/benchmark数据集v1.2_0628/')
    parser.add_argument('--output_dir', type=str, default='./edge_mask_output/top')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--num_entity_label', type=int, default=10)
    parser.add_argument('--num_desc_label', type=int, default=33)
    parser.add_argument('--num_graph_label', type=int, default=4)
    parser.add_argument('--max_sentence_length', type=int, default=200)
    parser.add_argument('--max_entity_length', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./model/')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--explainer', type=bool, default=True)
    parser.add_argument('--explain_epoch', type=int, default=500)
    parser.add_argument('--threshold', type=float, default=0)

    config = parser.parse_args()
    return config

    
def main():
    config = model_config()
    train_dataset = Medical_Data(os.path.join(config.data_dir, 'train'))
    dev_dataset = Medical_Data(os.path.join(config.data_dir, 'valid'))
    test_dataset = dev_dataset
    # test_dataset = Medical_Data(os.path.join(config.data_dir, 'test'))
    train_dataset.tokenize()
    dev_dataset.tokenize()
    train_data_loader = data_loader(train_dataset, config.batch_size, shuffle=True)
    dev_data_loader = data_loader(dev_dataset, config.batch_size)
    model = HGModel(config.in_dim, config.hidden_dim, config.out_dim, config.num_head,
                    config.num_desc_label, config.num_entity_label, config.num_graph_label)
    
    # torch.distributed.init_process_group(backend="nccl")
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    
    precision_score_graph_max = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if config.train:
        for epoch in range(config.epoch):
            model.train()
            entity_loss = 0.0
            desc_loss = 0.0
            graph_loss = 0.0
            num_desc = 0
            num_entity = 0
            desc_acc = 0
            entity_acc = 0
            graph_acc = 0
            graph_pred = []
            desc_pred = []
            entity_pred = []
            graph_label = []
            desc_label = []
            entity_label = []
            for idx, batch_data in tqdm(enumerate(train_data_loader)):
                optimizer.zero_grad()
                output = model(batch_data, train_dataset, epoch)
                loss = output['desc_loss'] + output['entity_loss'] + output['graph_loss']
                # loss = output['graph_loss'] + output['entity_loss']
                loss.backward()
                optimizer.step()
                entity_loss += output['entity_loss'].item()
                desc_loss += output['desc_loss'].item()
                graph_loss += output['graph_loss'].item()
                num_desc += output['desc_node']
                num_entity += output['entity_node']
                desc_acc += output['desc_acc']
                entity_acc += output['entity_acc']
                graph_acc += output['graph_acc']

                graph_pred.append(torch.cat(output['graph_pred'], dim=0))
                desc_pred.append(torch.cat(output['desc_pred'], dim=0))
                entity_pred.append(torch.cat(output['entity_pred'], dim=0))

                graph_label.append(batch_data['type'])
                desc_label.append(sum(batch_data['semantic'], []))
                entity_label.append(sum(batch_data['label'], []))

            graph_pred = torch.cat(graph_pred, dim=0).cpu().numpy()
            desc_pred = torch.cat(desc_pred, dim=0).cpu().numpy()
            entity_pred = torch.cat(entity_pred, dim=0).cpu().numpy()
            
            graph_label = sum(graph_label, [])
            graph_label = np.array([int(i) - 1 for i in graph_label])
            desc_label = np.array(sum(desc_label, []))
            entity_label = np.array(sum(entity_label, []))

            precision_score_graph = sklearn.metrics.precision_score(graph_label, graph_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
            accuracy_score_graph = sklearn.metrics.accuracy_score(graph_label, graph_pred, sample_weight=None)
            recall_score_graph = sklearn.metrics.recall_score(graph_label, graph_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
            f1_score_graph = sklearn.metrics.f1_score(graph_label, graph_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)

            precision_score_desc = sklearn.metrics.precision_score(desc_label, desc_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
            accuracy_score_desc = sklearn.metrics.accuracy_score(desc_label, desc_pred, sample_weight=None)
            recall_score_desc = sklearn.metrics.recall_score(desc_label, desc_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
            f1_score_desc = sklearn.metrics.f1_score(desc_label, desc_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)

            precision_score_entity = sklearn.metrics.precision_score(entity_label, entity_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
            accuracy_score_entity = sklearn.metrics.accuracy_score(entity_label, entity_pred, sample_weight=None)
            recall_score_entity = sklearn.metrics.recall_score(entity_label, entity_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
            f1_score_entity = sklearn.metrics.f1_score(entity_label, entity_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)

            print('Epoch: {}, Desc_accuracy {:.4f}, Entity_accuracy {:.4f}, Graph_accuracy {:.4f}, Desc_loss {:.4f}, Entity_loss {:.4f}, Graph_loss {:.4f}'.format(
                epoch, desc_acc / num_desc, entity_acc / num_entity, graph_acc / train_dataset.__len__(), desc_loss / train_dataset.__len__(),
                entity_loss / train_dataset.__len__(), graph_loss / train_dataset.__len__()))
            print("Graph: Precision: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_score_graph, accuracy_score_graph, recall_score_graph, f1_score_graph))   
            print("Desc: Precision: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_score_desc, accuracy_score_desc, recall_score_desc, f1_score_desc))
            print("Entity: Precision: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_score_entity, accuracy_score_entity, recall_score_entity, f1_score_entity)) 
            print('\n')
            model.eval()
            with torch.no_grad():
                entity_loss = 0.0
                desc_loss = 0.0
                graph_loss = 0.0
                num_desc = 0
                num_entity = 0
                desc_acc = 0
                entity_acc = 0
                graph_acc = 0
                graph_pred = []
                desc_pred = []
                entity_pred = []
                graph_label = []
                desc_label = []
                entity_label = []
                for idx, batch_data in enumerate(dev_data_loader):
                    output = model(batch_data, dev_dataset, epoch)
                    loss = output['desc_loss'] + output['entity_loss'] + output['graph_loss']
                    entity_loss += output['entity_loss'].item()
                    desc_loss += output['desc_loss'].item()
                    graph_loss += output['graph_loss'].item()
                    num_desc += output['desc_node']
                    num_entity += output['entity_node']
                    desc_acc += output['desc_acc']
                    entity_acc += output['entity_acc']
                    graph_acc += output['graph_acc']
                    
                    graph_pred.append(torch.cat(output['graph_pred'], dim=0))
                    desc_pred.append(torch.cat(output['desc_pred'], dim=0))
                    entity_pred.append(torch.cat(output['entity_pred'], dim=0))

                    graph_label.append(batch_data['type'])
                    desc_label.append(sum(batch_data['semantic'], []))
                    entity_label.append(sum(batch_data['label'], []))

                graph_pred = torch.cat(graph_pred, dim=0).cpu().numpy()
                desc_pred = torch.cat(desc_pred, dim=0).cpu().numpy()
                entity_pred = torch.cat(entity_pred, dim=0).cpu().numpy()
            
                graph_label = sum(graph_label, [])
                graph_label = np.array([int(i) - 1 for i in graph_label])
                desc_label = np.array(sum(desc_label, []))
                entity_label = np.array(sum(entity_label, []))

                precision_score_graph = sklearn.metrics.precision_score(graph_label, graph_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
                accuracy_score_graph = sklearn.metrics.accuracy_score(graph_label, graph_pred, sample_weight=None)
                recall_score_graph = sklearn.metrics.recall_score(graph_label, graph_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
                f1_score_graph = sklearn.metrics.f1_score(graph_label, graph_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)

                precision_score_desc = sklearn.metrics.precision_score(desc_label, desc_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
                accuracy_score_desc = sklearn.metrics.accuracy_score(desc_label, desc_pred, sample_weight=None)
                recall_score_desc = sklearn.metrics.recall_score(desc_label, desc_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
                f1_score_desc = sklearn.metrics.f1_score(desc_label, desc_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)

                precision_score_entity = sklearn.metrics.precision_score(entity_label, entity_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
                accuracy_score_entity = sklearn.metrics.accuracy_score(entity_label, entity_pred, sample_weight=None)
                recall_score_entity = sklearn.metrics.recall_score(entity_label, entity_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
                f1_score_entity = sklearn.metrics.f1_score(entity_label, entity_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)

                model_file = os.path.join(config.save_dir, 'trained_model')
                model_file = model_file + str(epoch)
                with open(model_file, 'wb') as fin:
                    torch.save(model, model_file)

                print('Epoch: {}, Desc_accuracy {:.4f}, Entity_accuracy {:.4f}, Graph_accuracy {:.4f}'.format(
                    epoch, desc_acc / num_desc, entity_acc / num_entity, graph_acc / dev_dataset.__len__())) 
                print("Graph: Precision: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_score_graph, accuracy_score_graph, recall_score_graph, f1_score_graph))   
                print("Desc: Precision: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_score_desc, accuracy_score_desc, recall_score_desc, f1_score_desc))
                print("Entity: Precision: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_score_entity, accuracy_score_entity, recall_score_entity, f1_score_entity)) 
                print('\n\n\n')   
    

    if config.explainer:
        model = torch.load(os.path.join(config.save_dir, 'trained_model_v1.2.pt2'))
        test_data_loader = data_loader(test_dataset, 1)

        explainer = GNNExplainer(model, config.explain_epoch)
        for idx, batch_data in tqdm(enumerate(test_data_loader)):
            node_feat_mask, edge_mask, graph, pred_label = explainer.explain_node(batch_data)
            for top in range(1, 6, 2):
                plt.figure(top)
                ax, G = explainer.visualize_subgraph(graph, edge_mask, label=batch_data['type'][0], top=top)

                sample_id = batch_data['sample_id'][0]
                output_dir_path = config.output_dir + str(top)
                dir_path = os.path.join(output_dir_path, sample_id)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                png_path = os.path.join(dir_path, sample_id + '.png')
                result_path = os.path.join(dir_path, sample_id + '.txt')
                plt.savefig(png_path)
                plt.close(top)
                
                
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write('The type of this record:  {}'.format(pred_label.tolist()[0]  + 1))
                    f.write('\n')
                    for node_id in G.nodes():
                        desc_id = node_id - 2 - len(test_dataset.dataset[idx]['raw'])
                        if desc_id > len(test_dataset.dataset[idx]['desc']):
                            continue
                        else:
                            f.write(str(node_id))
                            f.write(': ')
                            f.write(test_dataset.dataset[idx]['desc'][desc_id]['row'])
                            f.write('\t')
                            f.write(test_dataset.dataset[idx]['desc'][desc_id]['semantic'])
                            f.write('\n')

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    try:
        main()
    except KeyboardInterrupt:
        print('\nExited from the program ealier!')
