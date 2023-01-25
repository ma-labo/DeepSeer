"""
Name   : influential_pattern_mining.py
Author : ZHIJIE WANG
Time   : 2021/8/27
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--export_path', dest='export_path', default='./file/export/toxic/', help='output path')


if __name__ == '__main__':
    args = parser.parse_args()
    data = pd.read_json(args.export_path + 'training_data.json')
    trace = data['trace']
    seq_label = data['seq_label']
    text = data['text']

    mined_results = {}

    for i in range(len(trace)):
        if data['pred'][i] != data['label'][i]:
            continue
        for j in range(len(trace[i]) - 3):
            if len(set(seq_label[i][j:j + 3])) > 1:
                key = tuple(trace[i][j:j + 3])
                if key not in mined_results.keys():
                    mined_results[key] = {}
                phrase = ' '.join(text[i][j:j + 3])
                if phrase not in mined_results[key].keys():
                    mined_results[key][phrase] = 0
                mined_results[key][phrase] += 1

    for key in mined_results.keys():
        mined_results[key] = sorted(mined_results[key].items(), key=lambda x: x[1], reverse=True)

    for key in mined_results.keys():
        mined_results[key] = {v[0]: v[1] for v in mined_results[key]}

    sentimental_pattern = [{'key': key,
                            'mined_results': [{'text': text_key, 'freq': mined_results[key][text_key]} for text_key in
                                              mined_results[key].keys()]} for key in mined_results.keys()]

    sentimental_pattern = [v for v in sentimental_pattern if v['mined_results'] != []]

    sentimental_pattern = sorted(sentimental_pattern, key=lambda x: (sum([y['freq'] for y in x['mined_results']])),
                                 reverse=True)

    with open(args.export_path + 'sentimental_pattern.json', 'w') as f:
        json.dump(sentimental_pattern, f)


