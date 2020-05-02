import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    def __init__(self):
        self.dropout = 0.4                                             
        self.dim_embed = 300
        self.kernel_size = (4, 5, 6)                                  
        self.num_filters = 256     
        self.pretrained = False
        self.pretrained_path = None


class TextCNN(nn.Module):
    def __init__(self, config, max_len, vocab_size, num_classes):
        super(TextCNN, self).__init__()
        self.dim_embed = config.dim_embed
        self.num_filters = config.num_filters
        self.kernel_size = config.kernel_size
        self.max_len = max_len
        self.n_vocab = vocab_size
        self.num_classes = num_classes
        self.pretrained = config.pretrained
        self.pretrained_path = config.pretrained_path

        if self.pretrained: 
            self.embedding = nn.Embedding.from_pretrained(self.pretrained_path, freeze=False)
        else:
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

        x = self.embedding(x)              # [batch_size, max_len, dim_embed]
        x = x.unsqueeze(1)                 # [batch_size, 1, max_len, dim_embed]

        x1 = F.relu(self.conv1(x))         # [batch_size, num_filters, max_len-kernel_size[0], 1]
        x2 = F.relu(self.conv2(x))         # [batch_size, num_filters, max_len-kernel_size[1], 1]
        x3 = F.relu(self.conv3(x))         # [batch_size, num_filters, max_len-kernel_size[2], 1]

        x1 = self.max_pool1(x1)            # [batch_size, num_filters, 1, 1]
        x2 = self.max_pool2(x2)            # [batch_size, num_filters, 1, 1]
        x3 = self.max_pool3(x3)            # [batch_size, num_filters, 1, 1]

        x = torch.cat((x1, x2, x3), -1)    # [batch_size, num_filters, 1, 3]
        x = x.view(batch_size, 1, -1)      # [batch_size, 1, num_filters*3]

        x = self.dropout(x)
        x = self.fc(x)                     # [batch_size, 1, 2]
        x = x.view(-1, self.num_classes)   # [batch_size, 2]

        return x


if __name__ == "__main__":
    config = Config()
    max_len = 100
    batch_size = 8
    vocab_size = 500
    num_classes = 2

    textcnn = TextCNN(config, max_len, vocab_size, num_classes)

    x = torch.LongTensor(batch_size, max_len).random_(0, vocab_size)  # [8, 100]

    x = textcnn(x)
    print(x.size())
