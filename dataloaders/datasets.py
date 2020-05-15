import json
import numpy as np
import pickle as pkl
from torch.utils import data
from dataloaders import custom_transforms as tr
from config import DEVICE
from dataloaders.custom_transforms import UNK, PAD


def build_vocab(file_path, tokenizer, max_len, min_freq=3):
    vocab_dic = {}
    with open(file_path, 'r') as f:
        data = json.load(f)
    texts = [v["text"] for v in data.values()]
    for text in texts:
        for word in tokenizer.tokenize(text):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_len - 2]
    vocab_dic = {word[0]: idx for idx, word in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    return vocab_dic


class ClimateData(data.Dataset):
    NUM_CLASSES = 2
    def __init__(self, args, tokenizer, vocab, split='train'):
        self.texts = []
        self.labels = []
        self.args = args
        self.split = split
        self.vocab = vocab

        if self.split == "train":
            with open(self.args.train_path, 'r') as f:
                train = json.load(f)
            self.texts = [v["text"] for v in train.values()]
            self.labels = [v["label"] for v in train.values()]
        elif self.split == "val":
            with open(self.args.val_path, 'r') as f:
                val = json.load(f)
            self.texts = [v["text"] for v in val.values()]
            self.labels = [v["label"] for v in val.values()]
        elif self.split == "test":
            with open(self.args.test_path, 'b') as f:
                test = json.load(f)
            self.texts = [v["text"] for v in test.values()]
            self.labels = [0 for i in range(len(test))]
        else:
            raise Exception("No file for split %s" % self.split)

        assert len(self.texts) == len(self.labels)

        self.texts = [tokenizer.tokenize(text) for text in self.texts]
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        _text = self.texts[index]
        _label = self.labels[index]

        sample = {'text': _text, 'label': _label}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def transform(self, trans, sample):
        for obj in trans:
            sample = obj(sample)
        return sample

    def transform_tr(self, sample):
        trans = [
            tr.PadAndCut(self.args.max_len),
            tr.WordToId(self.vocab),
            tr.ToTensor()]
        return self.transform(trans, sample)

    def transform_val(self, sample):
        trans = [
            tr.PadAndCut(self.args.max_len),
            tr.WordToId(self.vocab),
            tr.ToTensor()]
        return self.transform(trans, sample)
    
    def transform_ts(self, sample):
        trans = [
            tr.PadAndCut(self.args.max_len),
            tr.WordToId(self.vocab),
            tr.ToTensor()]
        return self.transform(trans, sample)
        