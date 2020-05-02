import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    def __init__(self):                                    
        self.dim_embed = 300
        self.num_filters = 256


class DPCNN(nn.Module):
    def __init__(self, config, vocab_size, num_classes):
        super(DPCNN, self).__init__()
        self.dim_embed = config.dim_embed
        self.num_filters = config.num_filters
        self.n_vocab = vocab_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding(self.n_vocab, self.dim_embed)

        self.conv_region = nn.Conv2d(1, self.num_filters, (3, self.dim_embed), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] > 2:
            x = self._block(x)

        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = self.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


if __name__ == "__main__":
    config = Config()
    dpcnn = DPCNN(config, 500, 2)

    max_len = 100
    batch_size = 8

    x = torch.LongTensor(batch_size, max_len).random_(0, 2000)
    print(x.size())  # [8, 100]

    x = dpcnn(x)
    print(x.size())