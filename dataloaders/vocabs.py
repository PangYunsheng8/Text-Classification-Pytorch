import pickle as pkl
from dataloaders.custom_transforms import UNK, PAD


def build_vocab(file_path, max_size, min_freq=3):
    vocab_dic = {}
    lines = []
    if isinstance(file_path, list):
        with open(file_path[0], 'rb') as f:
            train_pos = pkl.load(f)
        with open(file_path[1], 'rb') as f:
            train_neg = pkl.load(f)
        lines = train_pos + train_neg
    elif isinstance(file_path, str):
        with open(file_path, 'rb') as f:
            lines = pkl.load(f)
    for line in lines:
        # line = line[0]
        for word in line:
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    print(len(vocab_dic))
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size-2]
    vocab_dic = {word[0]: idx for idx, word in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic