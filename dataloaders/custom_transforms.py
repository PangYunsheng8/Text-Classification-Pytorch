import torch
import numpy as np
from config import DEVICE

UNK, PAD = "<UNK>", "<PAD>"


class PadAndCut(object):
    def __init__(self, max_seq_size):
        self.max_seq_size = max_seq_size

    def __call__(self, sample):
        text = sample['text']
        label = sample['label']

        seq_len = len(text)
        if seq_len >= self.max_seq_size:
            text = text[:self.max_seq_size]
        else:
            text = [PAD] * (self.max_seq_size - seq_len) + text

        return {'text': text, 'label': label}


class WordToId(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, sample):
        text = sample['text']
        label = sample['label']

        _text = []
        for word in text:
            _text.append(self.vocab.get(word, self.vocab.get(UNK)))
        
        return {'text': _text, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        text = sample['text']
        label = sample['label']

        text = np.array(text)
        label = np.array(label)
        text = torch.from_numpy(text).to(torch.int64).to(DEVICE)
        label = torch.from_numpy(label).to(torch.int64).to(DEVICE)

        return {'text': text, 'label': label}