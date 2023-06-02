import os
import re
import torch
import dgl
import numpy as np
import networkx as nx
from transformers import *
from torch.utils.data import Dataset, DataLoader


desc2idx = {'null':0, '症状的特点_性质':1, '症状的特点_加重或缓解的因素':2, '起病情况_可能的原因和诱因':3, '其它_其它':4, '症状的特点_部位':5,
            '诊疗经过_地点':6, '诊疗经过_治疗效果':7, '症状的特点_程度':8, '诊疗经过_检查及诊断结果':9, '阴性资料_阴性资料':10, '通用描述_官话套话':11,
            '一般情况_睡眠':12, '一般情况_大小便':13, '一般情况_体重':14, '诊疗经过_治疗手段':15, '症状的特点_症状之间的相互关系':16, '收治目的_收治目的':17,
            '病情的发展和演变_症状的变化':18, '一般情况_精神':19, '一般情况_食欲':20, '诊疗经过_检查':21, '症状的特点_持续时间或频率':22, '症状的特点_症状出现的时间':23,
            '通用描述_时间数量描述':24, '诊疗经过_药物名称':25, '一般情况_体力':26, '诊疗经过_时间':27, '诊疗经过_药物剂量':28, '诊疗经过_诊疗原因':29,
            '起病情况_患病时间':30, '病情的发展和演变_新近出现的症状':31, '起病情况_起病缓急':32}
entity2idx = {'其他':0, '部位':1, '药品':2, '时长':3, '疾病':4, '医院':5, '手术':6, '数字':7, '检查':8, '时间':9}
idx2desc = {0:'null', 1:'症状的特点_性质', 2:'症状的特点_加重或缓解的因素', 3:'起病情况_可能的原因和诱因', 4:'其它_其它', 5:'症状的特点_部位',
            6:'诊疗经过_地点', 7:'诊疗经过_治疗效果', 8:'症状的特点_程度', 9:'诊疗经过_检查及诊断结果', 10:'阴性资料_阴性资料', 11:'通用描述_官话套话',
            12:'一般情况_睡眠', 13:'一般情况_大小便', 14:'一般情况_体重', 15:'诊疗经过_治疗手段', 16:'症状的特点_症状之间的相互关系', 17:'收治目的_收治目的',
            18:'病情的发展和演变_症状的变化', 19:'一般情况_精神', 20:'一般情况_食欲', 21:'诊疗经过_检查', 22:'症状的特点_持续时间或频率', 23:'症状的特点_症状出现的时间',
            24:'通用描述_时间数量描述', 25:'诊疗经过_药物名称', 26:'一般情况_体力', 27:'诊疗经过_时间', 28:'诊疗经过_药物剂量', 29:'诊疗经过_诊疗原因',
            30:'起病情况_患病时间', 31:'病情的发展和演变_新近出现的症状', 32:'起病情况_起病缓急'}
idx2entity = {0:'其他', 1:'部位', 2:'药品', 3:'时长', 4:'疾病', 5:'医院', 6:'手术', 7:'数字', 8:'检查', 9:'时间'}


class Medical_Data(Dataset):
    def __init__(self, data_dir):
        files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.max_raw_length = 250 # 242
        self.max_entity_length = 50 # 43
        self.max_desc_length = 100 # 86

        self.dataset = []
        for filename in files:
            sample = {}
            with open('./data/benchmark数据集v1.2_0628/sid_content.txt', 'r', encoding='utf-8') as fin:
                raw_data = fin.readlines()
                with open(filename, 'r', encoding='utf-8') as f:
                    data = f.readlines()
                    sample['sample_id'] = data[0].split('<->')[1].replace('\n', '')
                    sample['type'] = data[1].split('<->')[1].replace('\n', '')
                    sample['diagnose'] = data[2].split('<->')[1].replace('\n', '')
                    sample['age'] = data[3].split('<->')[1].replace('\n', '')
                    sample['gender'] = data[4].split('<->')[1].replace('\n', '')
                    sample['desc'] = []
                    entity = []
                    label = []
                    entity_idx = []
                    semantic = []
                    for i in range(5, len(data), 3):
                        desc = {}
                        desc['desc'] = data[i].split('<->')[1].replace('\n', '')
                        pattern = data[i + 1].split('<->')[1].replace('\n', '')
                        entity_list, label_list = self.entity_extract(desc['desc'], pattern)
                        entity += entity_list
                        label += label_list
                        entity_idx += [int((i - 5) / 3)] * len(entity_list)
                        desc['entity'] = entity_list
                        desc['entity_label'] = label_list
                        desc['semantic'] = data[i + 2].split('<->')[1].replace('\n', '')
                        sample['desc'].append(desc)
                        semantic.append(desc['semantic'])
                    sample['raw'] = raw_data[int(sample['sample_id']) - 1].split('<->')[1].replace(' \n', '').split('。') # 记得得把句号加回去
                    sample['entity'], sample['label'], sample['entity_idx'], sample['semantic'] = entity, label, entity_idx, semantic # 对应的是哪一个子句
                    desc_label = []
                    index = 0
                    raw_all = []
                    for i, sentence in enumerate(sample['raw']):
                        sentence = sentence + '。'
                        raw_sentence = ''
                        for j in range(index, len(sample['desc'])):
                            desc = sample['desc'][j]['desc']
                            if desc in sentence:
                                desc_label.append(i)
                                raw_sentence += desc
                            else:
                                index = j
                                break
                        raw_all.append(raw_sentence)
                    sample['raw'] = raw_all
                    sample['raw_text'] = raw_all
                    assert len(desc_label) == len(sample['desc'])
                    num_node = len(sample['raw']) + len(sample['desc']) + len(sample['entity']) + 2
                    src = [0, 1]
                    dst = [1, 2]
                    for i in range(2, 1 + len(sample['raw'])):
                        src.append(i)
                        dst.append(i + 1)
                    desc_label = [idx + 2 for idx in desc_label]
                    for i in range(len(sample['desc'])):
                        index = i + 2 + len(sample['raw'])
                        src.append(desc_label[i])
                        dst.append(index)
                    for i in range(len(sample['desc']) - 1):
                        for j in range(1, len(sample['desc']) - i):
                            index = i + 2 + len(sample['raw'])
                            src.append(index)
                            dst.append(index + j)
                    entity_label = [idx + 2 + len(sample['raw']) for idx in entity_idx]
                    for i in range(len(sample['entity'])):
                        index = i + 2 + len(sample['raw']) + len(sample['desc'])
                        src.append(entity_label[i])
                        dst.append(index)
                    for i in range(len(sample['desc'])):
                        index = i + 2 + len(sample['raw'])
                        src.append(index)
                        dst.append(num_node)
                    src = np.array(src)
                    dst = np.array(dst)
                    u = np.concatenate([src, dst])
                    v = np.concatenate([dst, src])
                    G = dgl.graph((u, v))
                    sample['Graph'] = G
                    sample['label'] = [entity2idx[lab] for lab in sample['label']]
                    sample['semantic'] = [desc2idx[lab] for lab in sample['semantic']]
            self.dataset.append(sample)

    def tokenize(self):
        tokenizer_class = BertTokenizer
        pretrained_weights = 'bert-base-chinese'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        for sample in self.dataset:
            descs = sample['desc']
            for desc in sample['desc']:
                desc['row'] = desc['desc']
                desc['desc'] = tokenizer.encode(desc['desc'], add_special_tokens=True)
                desc['desc'] = desc['desc'] + [tokenizer.pad_token_id] * (self.max_desc_length - len(desc['desc']))
                entity_list = []
                for entity in desc['entity']:
                    token = tokenizer.encode(entity, add_special_tokens=True)
                    entity_list.append(token + [tokenizer.pad_token_id] * (self.max_entity_length - len(token)))
                desc['entity'] = entity_list
            token = tokenizer.encode(sample['diagnose'], add_special_tokens=True)
            sample['diagnose'] = token + [tokenizer.pad_token_id] * (self.max_raw_length - len(token))
            sample['age'] = tokenizer.encode(sample['age'], add_special_tokens=True)
            sample['gender'] = tokenizer.encode(sample['gender'], add_special_tokens=True)
            for i, sentence in enumerate(sample['raw']):
                token = tokenizer.encode(sentence, add_special_tokens=True)
                if len(token) > self.max_raw_length:
                    token = token[:self.max_raw_length]
                sample['raw'][i] = token + [tokenizer.pad_token_id] * (self.max_raw_length - len(token))
            for i, entity in enumerate(sample['entity']):
                token = tokenizer.encode(entity, add_special_tokens=True)
                sample['entity'][i] = token + [tokenizer.pad_token_id] * (self.max_entity_length - len(token))    

    def entity_extract(self, entity_sentence, lable_sentence):
        entity_list = []
        label_list = re.findall('<(.*?)>', lable_sentence)
        for label in label_list:
            l = lable_sentence.find('<')
            r = lable_sentence.find('>')
            length = r - l + 1
            assert length == (len(label) + 2)
            if r < len(lable_sentence) - 1:
                char = lable_sentence[r + 1]
                r_a = entity_sentence.find(char)
                entity_list.append(entity_sentence[l: r_a])
                entity_sentence = entity_sentence[r_a:]
                lable_sentence = lable_sentence[r + 1:]
            else:
                entity_list.append(entity_sentence[l:])
        return entity_list, label_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def collate_fn(batch):
    age = []
    sample_id = []
    gender = []
    raw = []
    graph = []
    diagnose = []
    desc = []
    entity = []
    raw_num = []
    desc_num = []
    entity_num = []
    label = []
    semantic = []
    type_id = []
    for sample in batch:
        age.append(sample['age'])
        gender.append(sample['gender'])
        diagnose.append(sample['diagnose'])
        sample_id.append(sample['sample_id'])
        for sentence in sample['raw']:
            raw.append(sentence)
        raw_num.append(len(sample['raw']))
        for sentence in sample['desc']:
            desc.append(sentence['desc'])
        desc_num.append(len(sample['desc']))
        for sentence in sample['entity']:
            entity.append(sentence)
        entity_num.append(len(sample['entity']))
        graph.append(sample['Graph'])
        label.append(sample['label'])
        semantic.append(sample['semantic'])
        type_id.append(sample['type'])
    age = torch.tensor(age)
    gender = torch.tensor(gender)
    diagnose = torch.tensor(diagnose)
    raw = torch.tensor(raw)
    desc = torch.tensor(desc)
    entity = torch.tensor(entity)
    entity_num = torch.tensor(entity_num)
    raw_num = torch.tensor(raw_num)
    desc_num = torch.tensor(desc_num)
    return {'age': age, 'gender': gender, 'diagnose': diagnose, 'raw': raw, 'desc': desc,
            'entity': entity, 'entity_num': entity_num, 'raw_num': raw_num, 'desc_num': desc_num,
            'graph': graph, 'label': label, 'semantic': semantic, 'type': type_id, 'sample_id': sample_id}


def data_loader(dataset, batch_size=8, shuffle=False):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn 
    )
    return data_loader


if __name__ == "__main__":
    train_dataset = Medical_Data('data/benchmark数据集v1.2_0628/valid/')
    train_dataset.tokenize()
    train_data_loader = data_loader(train_dataset)
    for i, data in enumerate(train_data_loader):
        print('Done!')
        print('Success')
