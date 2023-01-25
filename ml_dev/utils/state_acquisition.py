"""
Name   : state_acquisition.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

from tqdm import tqdm
from collections import Counter


def gather_state_labels(traces, labels, num_classes=2):
    """
    sort the predict labels of states.
    :param traces: [[1,2,3], [1,2,3,4], ...]
    :param labels: [[0,1,1]], [1,0,0,0], ...]
    :param num_classes: number of predict classes
    :return: state -> label -> freq
    """
    state_labels = {}
    for i in tqdm(range(len(traces))):
        for j in range(len(traces[i])):
            if traces[i][j] not in state_labels.keys():
                state_labels[traces[i][j]] = {v: 0 for v in range(num_classes)}
            state_labels[traces[i][j]][labels[i][j][0]] += 1
    return state_labels


def gather_word_state_text(text, x_trace):
    """
    sort the trace by state -> word -> statistics or word -> trace -> statistics
    :param text: [['a','b','c'], ['a','b','c','d'], ...]
    :param x_trace: [[1,2,3], [1,2,3,4], ...]
    :return: (state -> word -> statistics, word -> trace -> statistics)
    """
    word_state_text = {}
    state_word_text = {}
    for i in tqdm(range(len(text))):
        trace = x_trace[i]
        abst_states = trace
        for j in range(len(abst_states)):
            state = abst_states[j]
            word = text[i][j]
            if word not in word_state_text.keys():
                word_state_text[word] = {}
                word_state_text[word][state] = {}
                word_state_text[word][state]['position'] = [(i, j)]
                word_state_text[word][state]['freq'] = 0
            else:
                if state not in word_state_text[word].keys():
                    word_state_text[word][state] = {}
                    word_state_text[word][state]['position'] = [(i, j)]
                    word_state_text[word][state]['freq'] = 0
                else:
                    word_state_text[word][state]['position'].append((i, j))
                    word_state_text[word][state]['freq'] += 1
            if state not in state_word_text.keys():
                state_word_text[state] = {}
                state_word_text[state][word] = {}
                state_word_text[state][word]['position'] = [(i,j)]
                state_word_text[state][word]['freq'] = 0
            else:
                if word not in state_word_text[state].keys():
                    state_word_text[state][word] = {}
                    state_word_text[state][word]['position'] = [(i, j)]
                    state_word_text[state][word]['freq'] = 0
                else:
                    state_word_text[state][word]['position'].append((i, j))
                    state_word_text[state][word]['freq'] += 1
    return word_state_text, state_word_text


def find_exact_pattern(target_trace, querry_trace):
    querry_trace_len = len(querry_trace)
    this_trace_len = len(target_trace)
    if this_trace_len < querry_trace_len:
        return False
    i = target_trace.index(querry_trace[0])
    while i < (this_trace_len - querry_trace_len + 1):
        k = 0
        while k < querry_trace_len and target_trace[i + k] == querry_trace[k]:
            k += 1
        if k == querry_trace_len:
            return [i + v for v in range(k)]
        i += 1


def find_similar_pattern(target_trace, querry_trace):
    """
    find similar pattern in a trace
    :param target_trace: [1, 2, 3, 4, 5, 6, ...]
    :param querry_trace: [2, 3, 4]
    :return:
    """
    # TODO
    result = []
    querry_trace_len = len(querry_trace)
    this_trace_len = len(target_trace)
    if this_trace_len < querry_trace_len:
        return False
    i = target_trace.index(querry_trace[0])
    while i < (this_trace_len - querry_trace_len):
        while i < (this_trace_len - querry_trace_len) and target_trace[i] != querry_trace[0]:
            i += 1
        j = 0
        k = 0
        id = []
        while (i + j) < this_trace_len:
            if target_trace[i + j] == querry_trace[k]:
                id.append(i + j)
                k += 1
            j += 1
            if k == querry_trace_len:
                result.append(id)
                break
        i = i + 1 if len(id) != querry_trace_len else id[0] + 1
    return result


def find_similar_trace(train_trace, test_trace, fuzz=False):
    """
    Find similar trace from training samples
    :param train_trace: [[1,2,3,4], [4,5,6,7], ...]
    :param test_trace: [2,3,4]
    :param fuzz: whether to do accurate search or not
    :return: [(training_idx, start_idx)]
    """
    result = []
    if fuzz:
        new_test_trace = [v for i, v in enumerate(test_trace) if i == 0 or v != test_trace[i-1]]
        test_trace_len = len(new_test_trace)
        for i in range(len(train_trace)):
            this_trace = train_trace[i]
            this_new_train_trace = [v for j, v in enumerate(this_trace) if j == 0 or v != this_trace[j-1]]
            idx = [j for j, v in enumerate(this_trace) if j == 0 or v != this_trace[j - 1]]
            this_trace_len = len(this_new_train_trace)
            if this_trace_len < test_trace_len:
                continue
            for j in range(this_trace_len - test_trace_len):
                k = 0
                while k < test_trace_len and this_new_train_trace[j + k] == new_test_trace[k]:
                    k += 1
                if k == test_trace_len:
                    result.append((i, idx[j], idx[j + k] + 1))
    else:
        test_trace_len = len(test_trace)
        for i in range(len(train_trace)):
            this_trace_len = len(train_trace[i])
            if this_trace_len < test_trace_len:
                continue
            for j in range(this_trace_len - test_trace_len):
                k = 0
                while k < test_trace_len and train_trace[i][j + k] == test_trace[k]:
                    k += 1
                if k == test_trace_len:
                    result.append((i, j))
    return result


def trace_cluster(trace, text):
    """
    Cluster instances according to traces.
    :param trace: [[1,2,3,4], [4,5,6,7], ...]
    :param text: [['a','b','c'], ['a','b','c','d'], ...]
    :return:
    """
    result = {}
    for i in range(len(trace)):
        tr = trace[i]
        new_tr = tuple([v for j, v in enumerate(tr) if j == 0 or v != tr[j - 1]])
        if new_tr not in result.keys():
            result[new_tr] = []
        result[new_tr] += [text[i]]
    return result


def trace_k_cluster(trace, text, k=4):
    """
    Cluster instances according to traces.
    :param trace: [[1,2,3,4], [4,5,6,7], ...]
    :param text: [['a','b','c'], ['a','b','c','d'], ...]
    :param k: maximum states number
    :return:
    """
    result, index = {}, {}
    for i in range(len(trace)):
        tr = trace[i]
        new_tr = [v for j, v in enumerate(tr) if j == 0 or v != tr[j - 1]]
        idx = [j for j, v in enumerate(tr) if j == 0 or v != tr[j - 1]]
        start, end = 0, len(tr)
        if len(new_tr) > k:
            count = Counter(tr)
            k_common = count.most_common(k)
            k_common = [v[0] for v in k_common]
            new_tr = [(v, new_tr.index(v)) for v in k_common]
            new_tr = sorted(new_tr, key=lambda item: item[1])
            start = idx[new_tr[0][1]]
            end = len(idx) if new_tr[-1][1] + 1 >= len(idx) else idx[new_tr[-1][1] + 1]
            idx = [idx[v[1]] for v in new_tr]
            new_tr = [v[0] for v in new_tr]
        new_tr = tuple(new_tr)
        if new_tr not in result.keys():
            result[new_tr] = []
            index[new_tr] = []
        index[new_tr] += [(i, tuple(idx))]
        idx += [end]
        split_text = [text[i][idx[ind-1]:idx[ind]] for ind in range(1, len(idx))]
        result[new_tr] += [split_text]
    return result, index


def trace_k_cluster_edge(trace, text, k=4):
    """
    Cluster instances according to traces (edges).
    :param trace: [[1,2,3,4], [4,5,6,7], ...]
    :param text: [['a','b','c'], ['a','b','c','d'], ...]
    :param k: maximum states number
    :return:
    """
    result, index = {}, {}
    for i in range(len(trace)):
        tr = trace[i]
        new_tr = [v for j, v in enumerate(tr) if j == 0 or v != tr[j - 1]]
        idx = [j for j, v in enumerate(tr) if j == 0 or v != tr[j - 1]]
        start, end = 0, len(tr)
        if len(new_tr) <= 1:
            continue
        if len(new_tr) > k:
            count = Counter(tr)
            k_common = count.most_common(k)
            k_common = [v[0] for v in k_common]
            new_tr = [(v, new_tr.index(v)) for v in k_common]
            new_tr = sorted(new_tr, key=lambda item: item[1])
            start = idx[new_tr[0][1]]
            end = len(idx) if new_tr[-1][1] + 1 >= len(idx) else idx[new_tr[-1][1] + 1]
            idx = [idx[v[1]] for v in new_tr]
            new_tr = [v[0] for v in new_tr]
        new_tr = tuple(new_tr)
        if new_tr not in result.keys():
            result[new_tr] = []
            index[new_tr] = []
        index[new_tr] += [(i, tuple(idx))]
        split_text = [text[i][idx[ind] - 1:idx[ind] + 1] for ind in range(1, len(idx))]
        result[new_tr] += [split_text]
    return result, index


def weights_cal(trace, state_num):
    edge_mat = [[0 for _ in range(state_num)] for _ in range(state_num)]
    max_edge = 0
    for i in range(len(trace)):
        tr = trace[i]
        for j in range(1, len(tr)):
            edge_mat[tr[j-1]-1][tr[j]-1] += 1
            if tr[j-1] != tr[j]:
                max_edge = max(edge_mat[tr[j-1]-1][tr[j]-1], max_edge)
    edge_weights = []
    for i in range(len(trace)):
        tr = trace[i]
        new_tr = [v for j, v in enumerate(tr) if j == 0 or v != tr[j - 1]]
        weight = 0
        for j in range(1, len(new_tr)):
            weight += (edge_mat[new_tr[j-1]-1][new_tr[j]-1] / max_edge)
        weight = weight / (len(new_tr) - 1) if (len(new_tr) - 1) > 1 else weight
        edge_weights.append((i, weight))
    edge_weights = sorted(edge_weights, key=lambda item: item[1], reverse=True)
    return edge_weights
