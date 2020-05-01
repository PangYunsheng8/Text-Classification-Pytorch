import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    def __init__(self):
        self.dropout = 0.5                                             # 随机失活
        self.dim_embed = 200
        self.kernel_size = (4, 5, 6)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


class TextCNN(nn.Module):
    def __init__(self, args, config, num_classes):
        super(TextCNN, self).__init__()
        self.dim_embed = config.dim_embed
        self.num_filters = config.num_filters
        self.kernel_size = config.kernel_size
        self.n_vocab = args.max_vocab_size
        self.max_len = args.max_seq_size
        self.num_classes = num_classes
        # self.pretrained = config.pretrained
        # self.pretrained_path = config.pretrained_path

        # if self.pretrained: 
        #     self.embedding = nn.Embedding.from_pretrained(self.pretrained_path, freeze=False)
        # else:
        self.embedding = nn.Embedding(self.n_vocab, self.dim_embed, padding_idx=self.n_vocab - 1)

        self.conv1 = nn.Conv2d(1, self.num_filters, (self.kernel_size[0], self.dim_embed))
        self.conv2 = nn.Conv2d(1, self.num_filters, (self.kernel_size[1], self.dim_embed))
        self.conv3 = nn.Conv2d(1, self.num_filters, (self.kernel_size[2], self.dim_embed))
        self.max_pool1 = nn.MaxPool2d((self.max_len - self.kernel_size[0] + 1, 1))
        self.max_pool2 = nn.MaxPool2d((self.max_len - self.kernel_size[1] + 1, 1))
        self.max_pool3 = nn.MaxPool2d((self.max_len - self.kernel_size[1] + 1, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.num_filters * 3, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.embedding(x)
        x = x.unsqueeze(1)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        x1 = self.max_pool1(x1)
        x2 = self.max_pool2(x2)
        x3 = self.max_pool3(x3)

        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch_size, 1, -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(-1, self.num_classes)

        return x
