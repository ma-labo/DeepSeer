"""
Name   : mining_state_words.py
Author : Zhijie Wang
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
    trace = data['trace'].values
    text = data['text'].values
    state_text = {i: [] for i in range(1, 40)}

    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))
    import string
    punch = set(string.punctuation)
    for i in range(len(trace)):
        for j in range(len(trace[i])):
            if text[i][j] not in stops and text[i][j] not in punch and text[i][j][:2] != '##':
                state_text[trace[i][j]] += [text[i][j]]

    state_text_count = {i: Counter(state_text[i]).most_common(10) for i in range(1, 40)}

    state_trans_text = {i: {j: [] for j in range(1, 40) if i != j} for i in range(1, 40)}

    for i in range(len(trace)):
        for j in range(1, len(trace[i])):
            if trace[i][j] != trace[i][j - 1]:
                if not (text[i][j - 1] in stops or text[i][j - 1] in punch or text[i][j - 1][:2] == '##') or not (text[i][j] in stops or text[i][j] in punch or text[i][j][:2] == '##'):
                    state_trans_text[trace[i][j]][trace[i][j - 1]] += [' '.join(text[i][j - 1:j + 1])]
    state_trans_text_count = {i: {j: Counter(state_trans_text[i][j]).most_common(10) for j in range(1, 40) if j != i} for i in range(1, 40)}

    output = []
    for i in range(1, 40):
        tmp = state_trans_text_count[i]
        tmp = tmp.items()
        tmp = [v for v in tmp if v[1] != []]
        tmp = sorted(tmp, key=lambda x: x[1][0][1], reverse=True)
        output.append({'state': i, 'words': tmp})

    export_data = []
    for tmp in output:
        tmp['words'] = [[v[0], [list(b) for b in v[1]]] for v in tmp['words']]
        export_data.append(tmp)
    with open(args.export_path + 'state_words.json', 'w') as f:
        json.dump(export_data, f)
    pass

