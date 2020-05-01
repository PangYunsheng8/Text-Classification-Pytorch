import numpy as np
import pickle as pkl
from torch.utils import data
from dataloaders import custom_transforms as tr
from config import DEVICE


class ClimateData(data.Dataset):
    NUM_CLASSES = 2
    def __init__(self, args, vocab, split='train'):
        self.texts = []
        self.labels = []
        self.args = args
        self.split = split
        self.vocab = vocab

        if self.split == "train":
            train_pos_path = self.args.train_path[0]
            train_neg_path = self.args.train_path[1]
            with open(train_pos_path, 'rb') as f:
                train_pos = pkl.load(f)
            with open(train_neg_path, 'rb') as f:
                train_neg = pkl.load(f)
            self.texts = train_pos + train_neg
            self.labels = [1 for _ in range(len(train_pos))] + [0 for _ in range(len(train_neg))]
            # with open(self.args.train_path, 'rb') as f:
            #     train = pkl.load(f)
            # self.texts = [data[0] for data in train]
            # self.labels = [data[1] for data in train]
        elif self.split == "val":
            with open(self.args.val_path, 'rb') as f:
                val = pkl.load(f)
            self.texts = [data[0] for data in val]
            self.labels = [data[1] for data in val]
        elif self.split == "test":
            with open(self.args.test_path, 'rb') as f:
                test = pkl.load(f)
            self.texts = test
            self.labels = [0 for i in range(len(test))]
        else:
            raise Exception("No file for split %s" % self.split)

        assert len(self.texts) == len(self.labels)
    
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
            tr.PadAndCut(self.args.max_seq_size),
            tr.WordToId(self.vocab),
            tr.ToTensor()]
        return self.transform(trans, sample)

    def transform_val(self, sample):
        trans = [
            tr.PadAndCut(self.args.max_seq_size),
            tr.WordToId(self.vocab),
            tr.ToTensor()]
        return self.transform(trans, sample)
    
    def transform_ts(self, sample):
        trans = [
            tr.PadAndCut(self.args.max_seq_size),
            tr.WordToId(self.vocab),
            tr.ToTensor()]
        return self.transform(trans, sample)
        