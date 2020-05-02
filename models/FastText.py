import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    def __init__(self):
        self.dropout = 0.4                                      
        self.dim_embed = 300
        self.hidden_size = 256
        self.pretrained = False
        self.pretrained_path = None


class FastText(nn.Module):
    def __init__(self, config, vocab_size, num_classes):
        super(FastText, self).__init__()
        self.dim_embed = config.dim_embed
        self.hidden_size = config.hidden_size
        self.n_vocab = vocab_size
        self.num_classes = num_classes
        self.pretrained = config.pretrained
        self.pretrained_path = config.pretrained_path

        if self.pretrained: 
            self.embedding = nn.Embedding.from_pretrained(self.pretrained_path, freeze=False)
        else:
            self.embedding = nn.Embedding(self.n_vocab, self.dim_embed)

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(self.dim_embed, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    config = Config()
    max_len = 100
    batch_size = 8
    vocab_size = 500
    num_classes = 2

    fasttext = FastText(config, vocab_size, num_classes)

    x = torch.LongTensor(batch_size, max_len).random_(0, vocab_size)
    print(x.size())  # [8, 100]

    x = fasttext(x)
    print(x.size())