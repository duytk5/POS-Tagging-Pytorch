from __future__ import print_function
from data_loader import *
from static import *
from torch.utils.data import DataLoader
import argparse
import torch
from models.main_model import MainModel
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score


def get_y_invalid(y, b, mask):
    ans = []
    for ib in range(b):
        le = mask[ib]
        for il in range(len(le)):
            if le[il][0] == 1:
                ans.append(y[ib * len(le) + il])
    return ans


def main(args_):
    print(device)
    log = open(FILE_LOG, "a")
    log.truncate(0)

    train, test = SC_DATA("train"), SC_DATA("test")
    print('LOAD WORD2VEC VOCAB DONE !! =========================')
    train_dl = DataLoader(train, batch_size=args_.batch, shuffle=True)
    test_dl = DataLoader(test, batch_size=args_.batch, shuffle=False)

    model = MainModel(input_dim=6418, hidden_dim=128, output_dim=24).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_.lr)

    for epoch in range(args_.epoch):
        # Train
        train_loss = 0.0
        y = []
        output = []
        for feature, label, mask, list_chars, mask_chars in tqdm(train_dl, desc='Training', leave=False):
            label = label.view(-1)
            y += get_y_invalid(label.numpy().tolist(), len(mask), mask)

            feature, label = Variable(feature.long()).to(device), Variable(label.squeeze()).to(device)
            mask = Variable(mask.float()).to(device)
            list_chars = Variable(list_chars.long()).to(device)
            mask_chars = Variable(mask_chars.float()).to(device)

            y_hat = model(feature, mask, list_chars, mask_chars)
            loss = criterion(y_hat, label)
            train_loss += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_hat = np.argmax(y_hat.data.cpu(), axis=1)  # b x len
            output += get_y_invalid(y_hat.tolist(), len(mask), mask)

        train_acc = accuracy_score(y, output)
        # Test
        test_loss = 0.0
        y = []
        output = []
        for feature, label, mask, list_chars, mask_chars in tqdm(test_dl, desc='Testing', leave=False):
            label = label.view(-1)
            y += get_y_invalid(label.numpy().tolist(), len(mask), mask)

            feature, label = Variable(feature.long()).to(device), Variable(label.squeeze()).to(device)
            mask = Variable(mask.float()).to(device)
            list_chars = Variable(list_chars.long()).to(device)
            mask_chars = Variable(mask_chars.float()).to(device)

            y_hat = model(feature, mask, list_chars, mask_chars)
            loss = criterion(y_hat, label)
            test_loss += loss.data.item()
            y_hat = np.argmax(y_hat.data.cpu(), axis=1)  # b x len
            output += get_y_invalid(y_hat.tolist(), len(mask), mask)

        test_acc = accuracy_score(y, output)
        test_f1s = f1_score(y, output, average='macro')
        print('Epoch %2d: loss: %10.2f acc : %0.4f> Test loss: %5.2f acc: %0.4f f1s: %0.4f'
              % (epoch, train_loss, train_acc, test_loss, test_acc, test_f1s), file=log)
        print()
        print('Epoch %2d: loss: %10.2f acc : %0.4f> Test loss: %5.2f acc: %0.4f f1s: %0.4f'
              % (epoch, train_loss, train_acc, test_loss, test_acc, test_f1s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--model', type=int, default=0)
    args = parser.parse_args()
    main(args)
