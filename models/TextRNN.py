import torch
import torch.nn as nn
import numpy as np


class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.pretrained = config.pretrained
        self.pretrained_path = config.pretrained_path
        self.n_vocab = config.n_vocab
        self.dim_embed = config.dim_embed
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.num_classes = config.num_classes

        # if self.pretrained: 
        #     self.embedding = nn.Embedding.from_pretrained(self.pretrained_path, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(self.n_vocab, self.dim_embed, padding_idx=self.n_vocab - 1)

        self.lstm = nn.LSTM(self.dim_embed, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        # x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x