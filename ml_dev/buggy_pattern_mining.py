"""
Name   : buggy_pattern_mining.py
Author : Zhijie Wang
Time   : 2021/8/10
"""

import argparse
import pandas as pd
import torch
import numpy as np
import joblib
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from collections import Counter

from data.data_utils import CommentDataset
from model.simple_rnn import SimpleGRU
from abstraction.profiling import DeepStellar
from utils.state_acquisition import gather_word_state_text, gather_state_labels, find_exact_pattern
from graphics.state_graph import StateGraph

parser = argparse.ArgumentParser()
parser.add_argument('--profile_path', dest='profile_path', default='./file/profile/toxic/', help='profile path')
parser.add_argument('--out_path', dest='out_path', default='./file/cache/toxic/', help='output path')
parser.add_argument('--export_path', dest='export_path', default='./file/export/toxic/', help='output path')
parser.add_argument('--checkpoint', dest='checkpoint', default='./file/checkpoints/toxic_ckpt_best.pth',
                    help='checkpoint')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--pca_component', dest='pca_components', default=20, type=int, help='pca components')
parser.add_argument('--state_num', dest='state_num', default=39, type=int, help='num of abstract states')

if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    max_sentence_length = 256

    device = torch.device('cuda' if args.usegpu else 'cpu')

    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])

    model = SimpleGRU(*ckpt['model_args'])
    model = model.to(device)
    model.eval()

    model.load_state_dict(ckpt['model'])

    pca_data_path = '%spca_%d.ptr' % (args.profile_path, args.pca_components)
    deep_stellar_path = '%sdeep_stellar_p_%d_s_%d.profile' % (args.profile_path, args.pca_components, args.state_num)

    (pca_data, embedding, text, seq_labels, label, pred) = joblib.load(pca_data_path)
    deep_stellar_model = joblib.load(deep_stellar_path)

    failed_cases = [i for i in range(len(pred)) if pred[i] != label[i]]

    train_trace = deep_stellar_model.get_trace(pca_data)

    train_trace = [v for (ind, v) in enumerate(train_trace) if ind in failed_cases]
    text = [v for (ind, v) in enumerate(text) if ind in failed_cases]
    pred = [v for (ind, v) in enumerate(pred) if ind in failed_cases]

    os.makedirs(args.out_path + 'failed/', exist_ok=True)
    os.makedirs(args.out_path + 'correct/', exist_ok=True)
    os.makedirs(args.out_path + 'unique/', exist_ok=True)

    min_length, max_length, topk = 3, 10, 50
    '''
    There are some bugs in spmf when running TKS from command, you might have to use 
    graphic interface to generate the output manually. (set max gap as 1.)
    '''
    if not os.path.exists(args.out_path + 'failed/mined_top%d_%d_%d.txt' % (topk, min_length, max_length)):
        with open(args.out_path + 'failed/train_trace.txt', 'w') as f:
            for trace in train_trace:
                tmp = [str(v) + ' -1' for v in trace]
                tmp = ' '.join(tmp) + ' -2\n'
                f.writelines(tmp)
        command = 'java -jar ./file/java/spmf.jar run TKS ' + args.out_path + 'failed/train_trace.txt'
        command += ' ' + args.out_path + 'failed/mined_top%d_%d_%d.txt' % (topk, min_length, max_length)
        command += ' %d %d %d "" 1 true' % (topk, min_length, max_length)
        os.system(command)

    train_trace = deep_stellar_model.get_trace(pca_data)
    train_trace = [v for (ind, v) in enumerate(train_trace) if ind not in failed_cases]

    if not os.path.exists(args.out_path + 'correct/mined_top%d_%d_%d.txt' % (topk, min_length, max_length)):
        with open(args.out_path + 'correct/train_trace.txt', 'w') as f:
            for trace in train_trace:
                tmp = [str(v) + ' -1' for v in trace]
                tmp = ' '.join(tmp) + ' -2\n'
                f.writelines(tmp)
        command = 'java -jar ./file/java/spmf.jar run TKS ' + args.out_path + 'correct/train_trace.txt'
        command += ' ' + args.out_path + 'correct/mined_top%d_%d_%d.txt' % (topk, min_length, max_length)
        command += ' %d %d %d "" 1 true' % (topk, min_length, max_length)
        os.system(command)

    train_trace = deep_stellar_model.get_trace(pca_data)

    train_trace = [v for (ind, v) in enumerate(train_trace) if ind in failed_cases]

    with open(args.out_path + 'correct/mined_top%d_%d_%d.txt' % (topk, min_length, max_length), 'r') as f:
        mined_data = f.readlines()
    mined_patterns_correct = {}
    for line in mined_data:
        line_tmp = line.split('#')
        key = line_tmp[0].split(' ')[:-1]
        key = [int(v) for v in key if v != '-1']
        key = tuple(key)
        if len(key) < 3:
            continue
        support = int(line_tmp[1][5:-1])
        ids = line_tmp[2][5:].split(' ')
        ids = [int(v) for v in ids]
        assert len(ids) == support
        mined_patterns_correct[key] = ids
    del mined_data

    with open(args.out_path + 'failed/mined_top%d_%d_%d.txt' % (topk, min_length, max_length), 'r') as f:
        mined_data = f.readlines()
    mined_patterns = {}
    for line in mined_data:
        line_tmp = line.split('#')
        key = line_tmp[0].split(' ')[:-1]
        key = [int(v) for v in key if v != '-1']
        key = tuple(key)
        if len(key) < 3:
            continue
        support = int(line_tmp[1][5:-1])
        ids = line_tmp[2][5:].split(' ')
        ids = [int(v) for v in ids]
        assert len(ids) == support
        if key not in mined_patterns_correct.keys():
            mined_patterns[key] = ids
    del mined_data

    keys = [v[0] for v in sorted(mined_patterns.items(), key=lambda x:len(x[1]), reverse=True)]

    if os.path.exists(args.out_path + 'unique/mined_top%d_%d_%d.pattern' % (topk, min_length, max_length)):
        mined_text = joblib.load(args.out_path + 'unique/mined_top%d_%d_%d.pattern' % (topk, min_length, max_length))
    else:
        mined_text = {}
        for pattern in keys:
            mined_text[pattern] = [[] for _ in range(len(pattern))]
            for id in mined_patterns[pattern]:
                trace = train_trace[id]
                result = find_exact_pattern(trace.tolist(), list(pattern))
                assert result != False
                for i in range(len(result)):
                    v = result[i]
                    mined_text[pattern][i] += [text[id][v]]
        joblib.dump(mined_text, args.out_path + 'unique/mined_top%d_%d_%d.pattern' % (topk, min_length, max_length))
    text_results = {}
    idx = 0
    for key in mined_text.keys():
        group_ids = mined_patterns[key]
        sub_text = np.array(mined_text[key]).T
        sub_text = [' '.join(v) for v in sub_text]
        counter = Counter(sub_text)
        idx += 1
        text_results[key] = [counter.most_common(10), []]
        for (text_pattern, _) in text_results[key][0]:
            pattern_index = [ind for (ind, v) in enumerate(sub_text) if v == text_pattern]
            ori_index = np.array(group_ids)[pattern_index]
            pattern_prediction = np.array(pred)[ori_index]
            text_results[key][1].append(pattern_prediction)
    joblib.dump(text_results, args.out_path + 'unique/mined_top%d_%d_%d.text' % (topk, min_length, max_length))

    keys = text_results.keys()
    buggy_pattern = []
    for key in keys:
        tmp = {'key': list(key), "mined_results": [{'text': v[0], 'freq': v[1]} for v in text_results[key][0]]}
        buggy_pattern.append(tmp)

    import json
    with open(args.export_path + 'buggy_pattern.json', 'w') as f:
        json.dump(buggy_pattern, f)

