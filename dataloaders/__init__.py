import os
import nltk
import pickle as pkl
from dataloaders.datasets import ClimateData, build_vocab
from torch.utils.data import DataLoader

def make_data_loader(args):
    tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

    if args.dataset == 'climate':
        if os.path.exists(args.vocab_path):
            vocab = pkl.load(open(args.vocab_path, 'rb'))
        else:
            vocab = build_vocab(args.train_path, tokenizer=tokenizer, max_len=args.max_vocab_size, min_freq=args.min_freq)
            pkl.dump(vocab, open(args.vocab_path, 'wb'))
        print("vocab size: ", len(vocab))

        if args.mode == "train":
            train_set = ClimateData(args, tokenizer=tokenizer, vocab=vocab, split='train')
            val_set = ClimateData(args, tokenizer=tokenizer, vocab=vocab, split='val')

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
            test_loader = None
            
            num_class = train_set.NUM_CLASSES
        elif args.mode == "inference":
            train_loader, val_loader = None, None

            test_set = ClimateData(args, vocab=vocab, split='test')
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            num_class = test_set.NUM_CLASSES

        return vocab, train_loader, val_loader, test_loader, num_class