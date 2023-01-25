"""
Name   : deep_stellar_example.py
Author : Zhijie Wang
Time   : 2021/7/7
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

from data.data_utils import CommentDataset
from model.simple_rnn import SimpleGRU
from abstraction.profiling import DeepStellar
from utils.state_acquisition import gather_word_state_text, gather_state_labels
from graphics.state_graph import StateGraph


parser = argparse.ArgumentParser()
parser.add_argument('--train_file', dest='train_file', default='./file/data/train_subset.csv', help='path to data file')
parser.add_argument('--out_path', dest='out_path', default='./file/profile/toxic/', help='output path')
parser.add_argument('--checkpoint', dest='checkpoint', default='./file/checkpoints/toxic_ckpt_best.pth', help='checkpoint')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--pca_component', dest='pca_components', default=20, type=int, help='pca components')
parser.add_argument('--state_num', dest='state_num', default=39, type=int, help='num of abstract states')
parser.add_argument('--reprofiling', dest='reprofiling', default=False, type=bool, help='reprofiling or not')


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    file_name = args.train_file
    X_train = pd.read_csv(file_name, index_col="id")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    max_sentence_length = 256

    train_dataset = CommentDataset(df=X_train, tokenizer=tokenizer, max_length=max_sentence_length,
                                   data_col="comment_text", target="target", is_testing=False)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if args.usegpu else 'cpu')

    ckpt = torch.load(args.checkpoint)
    print('Loaded checkpoint, Acc: %.4f' % ckpt['acc'])

    model = SimpleGRU(*ckpt['model_args'])
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
                label_ = batch['y'][i].cpu().numpy() >= 0.5
                label_ = label_.astype(int)
                state_ = hidden_states[i].cpu().numpy()
                prediction_ = pred_tensor[i].cpu().numpy()
                prediction_ = prediction_ >= 0.5
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
    state_labels = gather_state_labels(train_trace, seq_labels)

    transitions, transition_dict = deep_stellar_model.ast_model.update_transitions(train_trace, embedding)

    visualizer = StateGraph(state_word_text, word_state_text, transition_dict.keys(), state_labels)
    all_state_graph = visualizer.draw_states()
    all_state_graph.render('%sall_states.gv' % out_path)
    trace_graph = visualizer.draw_single_transition(trace=train_trace[0], text=text[0])
    trace_graph.render('%strace_example.gv' % out_path)
    state_trans_graph = visualizer.draw_state_transition()
    state_trans_graph.render('%sstate_transition.gv' % out_path, format='png', view=True)
    word_graph = visualizer.draw_word_stat(word='the')
    word_graph.render('%sword_example.gv' % out_path)
    sequence_graph = visualizer.draw_sequence(trace=train_trace[0], text=text[0])
    sequence_graph.render('%ssequence_example.gv' % out_path)
    trace_graph_w_label = visualizer.draw_single_transition_w_labels(trace=train_trace[0], text=text[0],
                                                                     seq_labels=seq_labels[0])
    trace_graph_w_label.render('%strace_example_label.gv' % out_path)
    pass