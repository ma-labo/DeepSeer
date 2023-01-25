"""
Name   : simple_rnn.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGRU(nn.Module):
    def __init__(self, rnn_size, word_vec_size, embedding_size, dense_hidden_dim=None, dropout_prob=0.3, num_layers=1, target_size=1):
        super(SimpleGRU, self).__init__()
        self.rnn_size = rnn_size
        self.word_vec_size = word_vec_size
        self.num_layers = num_layers
        self.target_size = target_size
        self.dropoutProb = dropout_prob
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_size, word_vec_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.GRU(word_vec_size, rnn_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.dense_hidden_dim = dense_hidden_dim
        if dense_hidden_dim is None:
            self.dense = nn.Linear(rnn_size, target_size)
        else:
            layers = [nn.Linear(rnn_size, dense_hidden_dim[0])]
            for i in range(1, len(dense_hidden_dim)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(dense_hidden_dim[i-1], dense_hidden_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dense_hidden_dim[-1], target_size))
            self.dense = nn.Sequential(*layers)
        self.sigomid = torch.sigmoid

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        prediction = self.dense(lstm_out)
        prediction = self.sigomid(prediction)
        output = prediction[:, -1, :].squeeze()
        return output

    def profile(self, x):
        with torch.no_grad():
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            prediction = self.dense(lstm_out)
            prediction = self.sigomid(prediction)
        return lstm_out, prediction


class SimpleGRUTranslation(nn.Module):
    def __init__(self, rnn_size, word_vec_size, embedding_size, dense_hidden_dim=None, dropout_prob=0.3, num_layers=1, target_size=1):
        super(SimpleGRUTranslation, self).__init__()
        self.rnn_size = rnn_size
        self.word_vec_size = word_vec_size
        self.num_layers = num_layers
        self.target_size = target_size
        self.dropoutProb = dropout_prob
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_size, word_vec_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.GRU(word_vec_size, rnn_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.dense_hidden_dim = dense_hidden_dim
        if dense_hidden_dim is None:
            self.dense = nn.Linear(rnn_size, target_size)
        else:
            layers = [nn.Linear(rnn_size, dense_hidden_dim[0])]
            for i in range(1, len(dense_hidden_dim)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(dense_hidden_dim[i-1], dense_hidden_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dense_hidden_dim[-1], target_size))
            self.dense = nn.Sequential(*layers)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        prediction = self.dense(lstm_out)
        output = prediction
        return output

    def profile(self, x):
        with torch.no_grad():
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            prediction = self.dense(lstm_out)
            prediction = self.softmax(prediction)
        return lstm_out, prediction


class SimpleGRUMultiClassification(nn.Module):
    def __init__(self, rnn_size, word_vec_size, embedding_size, dense_hidden_dim=None, dropout_prob=0.3, num_layers=1, target_size=2):
        super(SimpleGRUMultiClassification, self).__init__()
        self.rnn_size = rnn_size
        self.word_vec_size = word_vec_size
        self.num_layers = num_layers
        self.target_size = target_size
        self.dropoutProb = dropout_prob
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_size, word_vec_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.GRU(word_vec_size, rnn_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.dense_hidden_dim = dense_hidden_dim
        if dense_hidden_dim is None:
            self.dense = nn.Linear(rnn_size, target_size)
        else:
            layers = [nn.Linear(rnn_size, dense_hidden_dim[0])]
            for i in range(1, len(dense_hidden_dim)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(dense_hidden_dim[i-1], dense_hidden_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dense_hidden_dim[-1], target_size))
            self.dense = nn.Sequential(*layers)
        self.softmax = nn.Softmax()

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        prediction = self.dense(lstm_out)
        output = prediction[:, -1, :].squeeze()
        return output

    def profile(self, x):
        with torch.no_grad():
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            prediction = self.dense(lstm_out)
        return lstm_out, prediction


class SimpleLSTM(nn.Module):
    def __init__(self, rnn_size, word_vec_size, embedding_size, dense_hidden_dim=None, dropout_prob=0.3, num_layers=1, target_size=1):
        super(SimpleLSTM, self).__init__()
        self.rnn_size = rnn_size
        self.word_vec_size = word_vec_size
        self.num_layers = num_layers
        self.target_size = target_size
        self.dropoutProb = dropout_prob
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(embedding_size, word_vec_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(word_vec_size, rnn_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.dense_hidden_dim = dense_hidden_dim
        if dense_hidden_dim is None:
            self.dense = nn.Linear(rnn_size, target_size)
        else:
            layers = [nn.Linear(rnn_size, dense_hidden_dim[0])]
            for i in range(1, len(dense_hidden_dim)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(dense_hidden_dim[i - 1], dense_hidden_dim[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dense_hidden_dim[-1], target_size))
            self.dense = nn.Sequential(*layers)
        self.sigomid = torch.sigmoid

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        prediction = self.dense(lstm_out)
        prediction = self.sigomid(prediction)
        output = prediction[:, -1, :].squeeze()
        return output

    def profile(self, x):
        with torch.no_grad():
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            prediction = self.dense(lstm_out)
            prediction = self.sigomid(prediction)
        return lstm_out, prediction
