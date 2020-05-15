import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from tensorboardX import SummaryWriter


def init_network(model, method='kaiming', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)


def train(args, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_batch = 0
    dev_best_acc = 0
    writer = SummaryWriter(log_dir=args.log_path + '/' + args.model + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(args.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        for i, data in enumerate(train_iter):
            text = data["text"]
            label = data["label"]
            outputs = model(text)
            model.zero_grad()
            loss = F.cross_entropy(outputs, label)
            loss.backward()
            optimizer.step()

            if total_batch % 10 == 0:
                y_true = label.data.cpu()
                y_pred = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(y_true, y_pred)
                dev_acc, dev_loss = evaluate(model, dev_iter)

                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), args.save_path + '/' + args.model + '.ckpt')
                    print("saved model, best acc on dev: %.4f" % dev_acc)

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
    writer.close()


def evaluate(model, dev_iter):
    model.eval()
    loss_total = 0
    y_preds = np.array([], dtype=int)
    y_trues = np.array([], dtype=int)
    with torch.no_grad():
        for data in dev_iter:
            text = data["text"]
            label = data["label"]
            outputs = model(text)
            loss = F.cross_entropy(outputs, label)
            loss_total += loss
            y_true = label.data.cpu().numpy()
            y_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            y_trues = np.append(y_trues, y_true)
            y_preds = np.append(y_preds, y_pred)
    acc = metrics.accuracy_score(y_trues, y_preds)
    return acc, loss_total / len(dev_iter)


def inference(args, model, test_iter):
    model.eval()
    y_preds = np.array([], dtype=int)
    with torch.no_grad():
        for i, data in enumerate(test_iter):
            text = data["text"]
            outputs = model(text)

            y_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            y_preds = np.append(y_preds, y_pred)
    return y_preds