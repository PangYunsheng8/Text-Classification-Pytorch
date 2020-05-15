import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):
    def __init__(self):            
        self.dropout = 0.5  
        self.dim_embed = 300
        self.dim_model = self.dim_embed
        self.hidden_size = 1024
        self.num_head = 5
        self.num_encoder = 1
        self.pretrained = False
        self.pretrained_path = None


class Transformer(nn.Module):
    def __init__(self, config, max_len, vocab_size, num_classes):
        super(Transformer, self).__init__()
        self.dim_embed = config.dim_embed
        self.dim_model = config.dim_model
        self.num_head = config.num_head
        self.num_encoder = config.num_encoder
        self.hidden_size = config.hidden_size
        self.max_len = max_len
        self.n_vocab = vocab_size
        self.num_classes = num_classes
        self.pretrained = config.pretrained
        self.pretrained_path = config.pretrained_path

        if self.pretrained: 
            self.embedding = nn.Embedding.from_pretrained(self.pretrained_path, freeze=False)
        else:
            self.embedding = nn.Embedding(self.n_vocab, self.dim_embed)

        self.postion_embedding = Positional_Encoding(self.dim_embed, self.max_len, config.dropout)
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden_size, config.dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(self.num_encoder)])
        self.fc = nn.Linear(self.max_len * self.dim_model, self.num_classes)

    def forward(self, x):
        x = self.embedding(x)                                                    # [batch_size, max_len, dim_embed]
        for encoder in self.encoders:
            x = encoder(x)                                                       # [batch_size, max_len, dim_embed]
        x = x.view(x.size(0), -1)                                                # [batch_size, max_len * dim_embed]
        x = self.fc(x)                                                           # [batch_size, num_classes]
        return x


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        # self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False)#.to(self.device)
        out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden_size, dropout)

    def forward(self, x):
        x = self.attention(x)                                                    # [batch_size, max_len, dim_embed]
        x = self.feed_forward(x)                                                 # [batch_size, max_len, dim_embed]
        return x


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size*num_head, max_len, dim_head]
            K: [batch_size*num_head, max_len, dim_head]
            V: [batch_size*num_head, max_len, dim_head]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))                          # [batch_size*num_head, max_len, max_len]                     
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=1)
        context = torch.matmul(attention, V)                                     # [batch_size*num_head, max_len, dim_head]
        return context


class Multi_Head_Attention(nn.Module):
    '''Multi Head Attention '''
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = self.dim_model // self.num_head

        self.fc_Q = nn.Linear(self.dim_model, self.num_head*self.dim_head)
        self.fc_K = nn.Linear(self.dim_model, self.num_head*self.dim_head)
        self.fc_V = nn.Linear(self.dim_model, self.num_head*self.dim_head)

        self.attention = Scaled_Dot_Product_Attention()

        self.fc = nn.Linear(self.num_head*self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)                                                         # [batch_size, max_len, dim_embed]
        K = self.fc_K(x)                                                         # [batch_size, max_len, dim_embed]
        V = self.fc_V(x)                                                         # [batch_size, max_len, dim_embed]

        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)                # [batch_size*num_head, max_len, dim_head]
        K = K.view(batch_size * self.num_head, -1, self.dim_head)                # [batch_size*num_head, max_len, dim_head]
        V = V.view(batch_size * self.num_head, -1, self.dim_head)                # [batch_size*num_head, max_len, dim_head]

        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)                                 # [batch_size*num_head, max_len, dim_head]
        context = context.view(batch_size, -1, self.dim_head * self.num_head)    # [batch_size, max_len, dim_embed]

        out = self.fc(context)                                                   # [batch_size, max_len, dim_embed]
        out = self.dropout(out)

        x = out + x
        x = self.layer_norm(x)
        return x


class Position_wise_Feed_Forward(nn.Module):
    '''Position wise Feed Forward'''
    def __init__(self, dim_model, hidden_size, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.dim_model = dim_model
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(dim_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, dim_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)                                                        # [batch_size, max_len, hidden_size]
        out = self.relu(out)
        out = self.fc2(out)                                                      # [batch_size, max_len, dim_embed]
        out = self.dropout(out)
        x = out + x
        x = self.layer_norm(x)
        return x


if __name__ == "__main__":
    config = Config()
    max_len = 100
    batch_size = 2
    vocab_size = 300
    num_classes = 2

    transformer = Transformer(config, max_len, vocab_size, num_classes)

    x = torch.LongTensor(batch_size, max_len).random_(0, vocab_size)  # [2, 100]

    x = transformer(x)
    print(x.size())
