from torch import nn
import numpy as np
from torch.autograd import Variable
from lstm_cell import LSTMCell
from static import *


class LSTM_OUT(nn.Module):
    def init_hidden(self):
        zero1 = Variable(torch.FloatTensor(np.zeros((self.batch_size, self.hidden_dim)))).to(device)
        zero2 = Variable(torch.FloatTensor(np.zeros((self.batch_size, self.hidden_dim)))).to(device)
        return zero1, zero2

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM_OUT, self).__init__()
        self.batch_size = 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = nn.Sigmoid()
        self.cell = nn.LSTMCell(self.input_dim, self.hidden_dim, bias=True).to(device)

    def forward(self, x):
        # B L D -> L B D
        self.batch_size = x.shape[0]

        x = x.transpose(0, 1)
        h, c = self.init_hidden()
        list_lstm = Variable(torch.FloatTensor(np.empty(0))).to(device)
        for xi in x:
            h, c = self.cell(xi, (h, c))
            list_lstm = torch.cat((list_lstm, self.activation(h)))
        return list_lstm

