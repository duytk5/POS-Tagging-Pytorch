from torch import nn
import numpy as np
from torch.autograd import Variable
from lstm_cell import LSTMCell
from static import *
import torch


class LSTMCharacter(nn.Module):
    def init_hidden(self):
        zero1 = Variable(torch.FloatTensor(np.zeros((self.batch_size, self.hidden_dim)))).to(device)
        zero2 = Variable(torch.FloatTensor(np.zeros((self.batch_size, self.hidden_dim)))).to(device)
        return zero1, zero2

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMCharacter, self).__init__()
        self.batch_size = 0
        self.length_word = 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = nn.Softmax(1)
        self.linear = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.cell_1 = nn.LSTMCell(64, self.hidden_dim, bias=True).to(device)
        self.cell_2 = nn.LSTMCell(64, self.hidden_dim, bias=True).to(device)
        self.embedding_char = nn.Embedding(self.input_dim, 64)

    def forward(self, x, mask):
        # B L D -> L B D
        self.batch_size = x.shape[0]
        self.length_word = x.shape[1]
        x = self.embedding_char(x).view(self.batch_size, self.length_word, 64).transpose(0, 1)
        mask = mask.transpose(0, 1)
        h_1, c_1 = self.init_hidden()
        h_2, c_2 = self.init_hidden()
        for i in range(x.shape[0]):
            xi_1 = x[i]
            xi_2 = x[x.shape[0] - i - 1]
            h_1, c_1 = self.cell_1(xi_1, (h_1, c_1))
            h_2, c_2 = self.cell_2(xi_2, (h_2, c_2))

        h_1 = torch.cat((h_1.transpose(0, 1), h_2.transpose(0, 1)))
        h_1 = h_1.view(self.hidden_dim*2, self.batch_size)\
            .transpose(0, 1)
        return self.activation(self.linear(h_1))

