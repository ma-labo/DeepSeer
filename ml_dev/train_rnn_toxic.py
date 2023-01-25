"""
Name   : train_rnn_toxic.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

from data.data_utils import CommentDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import argparse
import pandas as pd
import numpy as np
from model.simple_rnn import SimpleGRU


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', dest='dataset_name', default='toxic', help='name of dataset')
parser.add_argument('--train_file', dest='train_file', default='./file/data/train_subset.csv', help='path to data file')
parser.add_argument('--out_path', dest='out_path', default='./file/checkpoints/', help='output path')
parser.add_argument('--usegpu', dest='usegpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--batch_size', dest='batch_size', default=10, type=int, help='batch size')
parser.add_argument('--num_epochs', dest='num_epochs', default=40, type=int, help='num of epochs')
parser.add_argument('--lr', dest='lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_layers', dest='num_layers', default=2, type=int, help='num of rnn layers')
parser.add_argument('--word_vec_size', dest='word_vec_size', default=200, type=int, help='word embedding size')
parser.add_argument('--rnn_size', dest='rnn_size', default=256, type=int, help='hidden state dim')
parser.add_argument('--dense_hidden_dim', dest='dense_hidden_dim', default=[64], type=list, help='hidden state dim')


if __name__ == '__main__':
    args = parser.parse_args()
    out_path = args.out_path
    file_name = args.train_file
    X = pd.read_csv(file_name, index_col="id")

    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.7)
    X_train = X.iloc[idx[:split]]
    X_val = X.iloc[idx[split:]]

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    max_sentence_length = 256

    train_dataset = CommentDataset(df=X_train, tokenizer=tokenizer, max_length=max_sentence_length,
                                   data_col="comment_text", target="target", is_testing=False)

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)

    val_dataset = CommentDataset(df=X_val, tokenizer=tokenizer, max_length=max_sentence_length,
                                 data_col="comment_text", target="target", is_testing=False)

    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if args.usegpu else 'cpu')

    model = SimpleGRU(word_vec_size=args.word_vec_size, rnn_size=args.rnn_size, embedding_size=tokenizer.vocab_size,
                      num_layers=args.num_layers, dense_hidden_dim=args.dense_hidden_dim)
    model = model.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = 0
        batch_num = 0
        for batch in train_dataloader:
            input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).float()
            y_pred = model(input_tensor)
            loss = loss_function(y_pred, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()  # / len(train_loader)
            batch_num += 1
            if batch_num > -1 and batch_num % 100 == 0:
                print('Train Epoch: %d, Batch: %d/%d, Loss: %6f' % (
                epoch + 1, batch_num + 1, len(train_dataloader), avg_loss / (batch_num + 1)))
        print('Train Epoch: %d, Loss: %6f' % (epoch + 1, avg_loss / (len(train_dataloader))))
        best_loss = float('inf')
        best_acc = 0
        model.eval()
        with torch.no_grad():
            avg_loss = 0
            correct = 0
            all_pred = 0
            for batch in val_dataloader:
                input_tensor, target_tensor = batch['input_ids'].to(device).long(), batch['y'].to(device).float()
                y_pred = model(input_tensor)
                loss = loss_function(y_pred, target_tensor)
                avg_loss += loss.item()  # / len(train_loader)
                y_pred = y_pred >= 0.5
                target_tensor = target_tensor >= 0.5
                correct += torch.sum(y_pred == target_tensor).item()
                all_pred += y_pred.size(0)
            acc = correct / all_pred
            print('Val Epoch: %d, Loss: %6f, Acc: %4f' % (epoch + 1, avg_loss / (len(val_dataloader)), acc))
            if (avg_loss / (len(val_dataloader))) < best_loss or acc > best_acc:
                best_loss = avg_loss / (len(val_dataloader))
                best_acc = acc
                result = {'model': model.state_dict(),
                          'loss': avg_loss / (len(val_dataloader)),
                          'acc': acc,
                          'model_args': (model.rnn_size, model.word_vec_size, model.embedding_size,
                                         model.dense_hidden_dim, model.dropoutProb, model.num_layers,
                                         model.target_size)}
                torch.save(result, '%s%s_ckpt_best.pth' % (out_path, args.dataset_name))

