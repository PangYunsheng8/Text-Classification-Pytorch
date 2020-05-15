import argparse
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Text Classification")

        # model
        parser.add_argument("--model", type=str, default="FastText", help="model name")
        parser.add_argument("--mode", type=str, default="train", help="train or inference")

        # dataset
        parser.add_argument("--dataset", type=str, default="climate", help="dataset name")
        parser.add_argument("--train_path", type=str, default="data/train.json", help="path of training set")
        parser.add_argument("--val_path", type=str, default="data/valid.json", help="path of val set")
        parser.add_argument("--test_path", type=str, default="", help="path of test set")
        parser.add_argument("--vocab_path", type=str, default="data/vocab.pkl", help="path of vocab")
        parser.add_argument("--max_vocab_size", type=int, default=50000, help="max vocab size")
        parser.add_argument("--max_len", type=int, default=1500, help="max length of sequence")
        parser.add_argument("--min_freq", type=int, default=3, help="min frequency of words")

        # training hyper params
        parser.add_argument("--num_epochs", type=int, default=30, metavar="N", help="number of epochs to train")
        parser.add_argument("--batch_size", type=int, default=32, metavar="N", help="input batch size for training")

        # optimizer params
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

        # cuda and seed
        parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
        parser.add_argument("--seed", type=int, default=1, help="random seed")
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids")

        # save
        parser.add_argument("--log_path", type=str, default="log/", help="path of log")
        parser.add_argument("--save_path", type=str, default="saved/", help="path of save")

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args