import torch
import numpy as np
import json
from config import Options
from dataloaders import make_data_loader
from train import train, init_network, evaluate, inference
from utils import generate_results
from config import DEVICE


def main():
    args = Options().parse()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")
    vocab, train_loader, val_loader, test_loader, num_class = make_data_loader(args)
    vocab_size = len(vocab)

    # train
    if args.model == "TextCNN": # 0.9846
        from models.TextCNN import TextCNN, Config
        config = Config()
        model = TextCNN(config, args.max_len, vocab_size, num_class)
    elif args.model == "FastText": # 0.9807(20 epoch), 0.9884(30 epoch)
        from models.FastText import FastText, Config
        config = Config()
        model = FastText(config, vocab_size, num_class)
    elif args.model == "DPCNN":
        from models.DPCNN import DPCNN, Config
        config = Config()
        model = DPCNN(config, vocab_size, num_class)
    elif args.model == "Transformer":
        from models.Transformer import Transformer, Config
        config = Config()
        model = Transformer(config, args.max_len, vocab_size, num_class)
    model.to(DEVICE)

    if args.mode == "train":
        init_network(model)
        train(args, model, train_loader, val_loader)
    elif args.mode == "inference":
        model.load_state_dict(torch.load(args.save_path + '/' + args.model + '.ckpt'))
        y_preds = inference(args, model, test_loader)
        generate_results(y_preds)

if __name__ == '__main__':
    main()