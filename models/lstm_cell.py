from torch import nn
from static import *


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wx = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        i = self.sigmoid(self.Wh(h) + self.Wx(x))
        f = self.sigmoid(self.Wh(h) + self.Wx(x))
        o = self.sigmoid(self.Wh(h) + self.Wx(x))
        g = self.tanh(self.Wh(h) + self.Wx(x))
        c = f * c + i * g
        h = o * self.tanh(c)
        return h, c
