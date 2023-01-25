"""
Name   : state_abstraction_toxic.py
Author : Zhijie Wang
Time   : 2021/7/28
"""

import argparse
import pandas as pd
import torch
import numpy as np
import joblib
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from torchtext.datasets import AG_NEWS

from data.data_utils import CommentDataset
from model.simple_rnn import SimpleGRUMultiClassification
from abstraction.profiling import DeepStellar
from utils.state_acquisition import gather_word_state_text, gather_state_labels
from graphics.state_graph import StateGraph


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', dest='train_file', default='./file/data/train_subset_quora.csv', help='path to data file')
parser.add_argument('--test_file', dest='test_file', default='./file/data/test_quora.csv', help='test_set')
parser.add_argument('--out_path', dest='out_path', default='./file/profile/agnews/', help='output path')
parser.add_argument('--export_path', dest='export_path', default='./file/export/agnews/', help='output path')
parser.add_argument('--checkpoint', dest='checkpoint', default='./file/checkpoints/agnews_ckpt_best.pth', help='checkpoint')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--pca_component', dest='pca_components', default=20, type=int, help='pca components')
parser.add_argument('--state_num', dest='state_num', default=39, type=int, help='num of abstract states')
parser.add_argument('--reprofiling', dest='reprofiling', default=False, type=bool, help='reprofiling or not')
parser.add_argument('--edge_topk', dest='edge_topk', default=2, type=int, help='topk output edge')


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    export_path = args.export_path
    os.makedirs(export_path, exist_ok=True)
    train_iter = AG_NEWS(root='./file/data/', split='train')
    X_train = {'comment_text': [], 'target': []}
    for label, line in train_iter:
        X_train['comment_text'] += [line]
        X_train['target'] += [label - 1]
    X_train = pd.DataFrame(data=X_train)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    max_sentence_length = 256

    train_dataset = CommentDataset(df=X_train, tokenizer=tokenizer, max_length=max_sentence_length,
                                   data_col="comment_text", target="target", is_testing=False)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if args.usegpu else 'cpu')

    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])

    model = SimpleGRUMultiClassification(*ckpt['model_args'])
    model = model.to(device)
    model.eval()

    model.load_state_dict(ckpt['model'])

    pca_data_path = '%spca_%d.ptr' % (args.out_path, args.pca_components)
    deep_stellar_path = '%sdeep_stellar_p_%d_s_%d.profile' % (args.out_path, args.pca_components, args.state_num)

    if os.path.exists(pca_data_path) and os.path.exists(deep_stellar_path) and not args.reprofiling:
        (pca_data, embedding, text, seq_labels, label, pred) = joblib.load(pca_data_path)
        deep_stellar_model = joblib.load(deep_stellar_path)
    else:
        state_vec = []
        embedding = []
        text = []
        label = []
        pred = []
        seq_labels = []

        for batch in tqdm(train_dataloader):
            input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).float()
            hidden_states, pred_tensor = model.profile(input_tensor)
            batch_size = hidden_states.size(0)
            for i in range(batch_size):
                embedding_ = batch['input_ids'][i].cpu().numpy()
                mask_ = batch['attn_mask'][i].cpu().numpy()
                text_ = np.array(train_dataset.tokenizer.convert_ids_to_tokens(embedding_))
                label_ = batch['y'][i].cpu().numpy()
                label_ = label_.astype(int)
                state_ = hidden_states[i].cpu().numpy()
                prediction_ = torch.argmax(pred_tensor[i], dim=-1, keepdim=True).cpu().numpy()
                prediction_ = prediction_.astype(int)
                state_vec.append(state_[mask_ == 1.])
                embedding.append(embedding_[mask_ == 1.])
                text.append(text_[mask_ == 1.])
                label.append(label_)
                seq_labels.append(prediction_[mask_ == 1.])
                pred.append(seq_labels[-1][-1][0])

        deep_stellar_model = DeepStellar(args.pca_components, args.state_num, state_vec)
        pca_data = deep_stellar_model.pca.do_reduction(state_vec)
        joblib.dump((pca_data, embedding, text, seq_labels, label, pred), pca_data_path)
        joblib.dump(deep_stellar_model, deep_stellar_path)

    train_trace = deep_stellar_model.get_trace(pca_data)
    (word_state_text, state_word_text) = gather_word_state_text(text, train_trace)
    state_labels = gather_state_labels(train_trace, seq_labels, num_classes=4)

    training_data = {'id': [v for v in range(len(train_trace))],
                     'trace': [v.tolist() for v in train_trace],
                     'seq_label': [v.reshape(-1).tolist() for v in seq_labels],
                     'pred': [int(v) for v in pred],
                     'label': [int(v) for v in label],
                     'text': [v.tolist() for v in text]}
    pd.DataFrame(training_data).to_json(export_path + 'training_data.json', orient='records')
    # for i in range(len(train_trace)):
    #     training_data[i] = {}
    #     training_data[i]['trace'] = train_trace[i].tolist()
    #     training_data[i]['seq_label'] = seq_labels[i].reshape(-1).tolist()
    #     training_data[i]['label'] = int(label[i])
    #     training_data[i]['text'] = text[i].tolist()
    # with open(export_path + 'training_data.json', 'w') as f:
    #     json.dump(training_data, f)

    transitions, transition_dict = deep_stellar_model.ast_model.update_transitions(train_trace, embedding)

    visualizer = StateGraph(state_word_text, word_state_text, transition_dict.keys(), state_labels, base_node_size=8.)

    basic_graph_data = {'node_size': {int(key): visualizer.node_size[key] for key in visualizer.node_size}}
    for topk in range(1, 11):
        edge_list = set()
        for state in visualizer.states:
            # in edge
            in_edge_num = len(visualizer.topk_edges[state]['in'])
            for i in range(min(in_edge_num, topk)):
                edge_width = round((visualizer.edges[(
                visualizer.topk_edges[state]['in'][i][0], state)] / visualizer.most_popular_trace) * 9. + 1., 2)
                if (visualizer.topk_edges[state]['in'][i][0], int(state), edge_width) not in edge_list:
                    edge_list.add((visualizer.topk_edges[state]['in'][i][0], int(state), edge_width))
            # out edge
            out_edge_num = len(visualizer.topk_edges[state]['out'])
            for i in range(min(out_edge_num, topk)):
                edge_width = round((visualizer.edges[(
                state, visualizer.topk_edges[state]['out'][i][0])] / visualizer.most_popular_trace) * 9. + 1., 2)
                if (int(state), visualizer.topk_edges[state]['out'][i][0], edge_width) not in edge_list:
                    edge_list.add((int(state), visualizer.topk_edges[state]['out'][i][0], edge_width))
        edge_list = list(edge_list)
        basic_graph_data[topk] = [list(v) for v in edge_list]
    # with open(export_path + 'basic_graph_data.json', 'w') as f:
    #     json.dump(basic_graph_data, f)
    # pass
    node_data = {'id': [i for i in range(1, 40)] + [0],
                 'size': [basic_graph_data['node_size'][i] for i in range(1, 40)] + [8],
                 'world': [0 for _ in range(40)],
                 'sports': [0 for _ in range(40)],
                 'business': [0 for _ in range(40)],
                 'sci-tech': [0 for _ in range(40)]}
    for i in range(len(train_trace)):
        for j in range(len(train_trace[i])):
            if seq_labels[i][j] == 0.:
                node_data['world'][train_trace[i][j] - 1] += 1
            elif seq_labels[i][j] == 1:
                node_data['sports'][train_trace[i][j] - 1] += 1
            elif seq_labels[i][j] == 2:
                node_data['business'][train_trace[i][j] - 1] += 1
            else:
                node_data['sci-tech'][train_trace[i][j] - 1] += 1
    node_data['world'][-1] = 1
    pd.DataFrame(node_data).to_json(export_path + 'node_data.json', orient='records')

    edge_data = basic_graph_data[args.edge_topk]
    source, target, width = [], [], []
    for edge in edge_data:
        source += [edge[0]]
        target += [edge[1]]
        width += [edge[2]]
    pd.DataFrame({'source': source, 'target': target, 'width': width}).to_json(export_path + 'edge_data.json', orient='records')


    if args.test_file:
        test_iter = AG_NEWS(root='./file/data/', split='test')
        X_test = {'comment_text': [], 'target': []}
        for label, line in test_iter:
            X_test['comment_text'] += [line]
            X_test['target'] += [label - 1]
        X_test = pd.DataFrame(data=X_test)

        test_dataset = CommentDataset(df=X_test, tokenizer=tokenizer, max_length=max_sentence_length,
                                      data_col="comment_text", target="target", is_testing=False)

        test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

        state_vec_test = []
        embedding_test = []
        text_test = []
        label_test = []
        pred_test = []
        seq_labels_test = []
        failed_cases = []
        for batch in tqdm(test_dataloader):
            input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).float()
            hidden_states, pred_tensor = model.profile(input_tensor)
            batch_size = hidden_states.size(0)
            for i in range(batch_size):
                embedding_ = batch['input_ids'][i].cpu().numpy()
                mask_ = batch['attn_mask'][i].cpu().numpy()
                text_ = np.array(train_dataset.tokenizer.convert_ids_to_tokens(embedding_))
                label_ = batch['y'][i].cpu().numpy()
                label_ = label_.astype(int)
                state_ = hidden_states[i].cpu().numpy()
                prediction_ = torch.argmax(pred_tensor[i], dim=-1, keepdim=True).cpu().numpy()
                prediction_ = prediction_.astype(int)
                state_vec_test.append(state_[mask_ == 1.])
                embedding_test.append(embedding_[mask_ == 1.])
                text_test.append(text_[mask_ == 1.])
                label_test.append(label_)
                seq_labels_test.append(prediction_[mask_ == 1.])
                pred_test.append(seq_labels_test[-1][-1][0])
                if pred_test[-1] != label_test[-1] and label_test[-1] == 0.:
                    failed_cases.append(len(pred_test) - 1)

        pca_test_data = deep_stellar_model.pca.do_reduction(state_vec_test)
        test_trace = deep_stellar_model.get_trace(pca_test_data)

        test_data = {'id': [v for v in range(len(test_trace))],
                         'trace': [v.tolist() for v in test_trace],
                         'seq_label': [v.reshape(-1).tolist() for v in seq_labels_test],
                         'pred': [int(v) for v in pred_test],
                         'label': [int(v) for v in label_test],
                         'text': [v.tolist() for v in text_test]}
        pd.DataFrame(test_data).to_json(export_path + 'test_data.json', orient='records')