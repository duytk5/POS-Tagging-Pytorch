from torch import nn
import numpy as np
from torch.autograd import Variable
from lstm_cell import LSTMCell
from static import *


class LSTM_FIRST(nn.Module):
    def init_hidden(self):
        zero1 = Variable(torch.FloatTensor(np.zeros((self.batch_size, self.hidden_dim)))).to(device)
        zero2 = Variable(torch.FloatTensor(np.zeros((self.batch_size, self.hidden_dim)))).to(device)
        return zero1, zero2

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM_FIRST, self).__init__()
        self.batch_size = 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = nn.Sigmoid()
        self.cell_1 = nn.LSTMCell(self.input_dim, self.hidden_dim, bias=True).to(device)
        self.cell_2 = nn.LSTMCell(self.input_dim, self.hidden_dim, bias=True).to(device)

    def forward(self, x):
        # B L D -> L B D
        self.batch_size = x.shape[0]

        x = x.transpose(0, 1)
        h_1, c_1 = self.init_hidden()
        h_2, c_2 = self.init_hidden()
        list_lstm_1 = Variable(torch.FloatTensor(np.empty(0))).to(device)
        list_lstm_2 = Variable(torch.FloatTensor(np.empty(0))).to(device)

        for i in range(x.shape[0]):
            xi_1 = x[i]
            h_1, c_1 = self.cell_1(xi_1, (h_1, c_1))
            xi_2 = x[x.shape[0] - i - 1]
            h_2, c_2 = self.cell_2(xi_2, (h_2, c_2))
            list_lstm_1 = torch.cat((list_lstm_1, h_1))
            list_lstm_2 = torch.cat((h_2, list_lstm_2))
        list_lstm = (list_lstm_1 + list_lstm_2)/2
        return list_lstm

