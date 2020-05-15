import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    def __init__(self):                                    
        self.dim_embed = 300
        self.num_filters = 256
        self.kernel_size = 3
        self.pretrained = False
        self.pretrained_path = None


class DPCNN(nn.Module):
    def __init__(self, config, vocab_size, num_classes):
        super(DPCNN, self).__init__()
        self.dim_embed = config.dim_embed
        self.num_filters = config.num_filters
        self.kernel_size = config.kernel_size
        self.n_vocab = vocab_size
        self.num_classes = num_classes
        self.pretrained = config.pretrained
        self.pretrained_path = config.pretrained_path

        if self.pretrained: 
            self.embedding = nn.Embedding.from_pretrained(self.pretrained_path, freeze=False)
        else:
            self.embedding = nn.Embedding(self.n_vocab, self.dim_embed)

        self.conv_region = nn.Conv2d(1, self.num_filters, (self.kernel_size, self.dim_embed), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (self.kernel_size, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(self.kernel_size, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.num_filters, self.num_classes)

    def forward(self, x):
        x = self.embedding(x)     # [batch_size, max_len, dim_embed]
        x = x.unsqueeze(1)        # [batch_size, 1, max_len, dim_embed]
        x = self.conv_region(x)   # [batch_size, num_filters, max_len-kernel_size, 1]
        x = self.padding1(x)      # [batch_size, num_filters, max_len, 1]
        x = self.relu(x)
        x = self.conv(x)          # [batch_size, num_filters, max_len-kernel_size, 1]
        x = self.padding1(x)      # [batch_size, num_filters, max_len, 1]
        x = self.relu(x)
        x = self.conv(x)          # [batch_size, num_filters, max_len-kernel_size, 1]
        while x.size()[2] > 2:
            x = self._block(x)    # [batch_size, num_filters, 1, 1]
        x = x.squeeze()           # [batch_size, num_filters]
        x = self.fc(x)            # [batch_size, num_classes]
         
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
    max_len = 1600
    batch_size = 8
    vocab_size = 500
    num_classes = 2

    dpcnn = DPCNN(config, vocab_size, num_classes)

    x = torch.LongTensor(batch_size, max_len).random_(0, vocab_size)  # [8, 100]

    x = dpcnn(x)
    print(x.size())