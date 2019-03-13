from torch import nn
from lstm_output_layer import LSTM_OUT
from lstm_first_layer import LSTM_FIRST
from lstm_character import LSTMCharacter
from torch.autograd import Variable
import numpy as np
from static import *


class MainModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MainModel, self).__init__()
        self.lstm_layer_1 = LSTM_OUT
        self.batch_size = 0
        self.output_dim_char = 50
        self.hidden_dim_char = 64
        self.input_dim_char = 187
        self.length_sentence = 100
        self.input_dim = input_dim
        self.input_lstm_dim = 50 + self.output_dim_char
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM
        self.lstm_layer_1 = LSTM_FIRST(self.input_lstm_dim, self.hidden_dim, self.hidden_dim).to(device)
        self.lstm_layer_out = LSTM_OUT(self.hidden_dim, self.hidden_dim, self.output_dim).to(device)
        self.lstm_layer_character = LSTMCharacter(self.input_dim_char, self.hidden_dim_char, self.output_dim_char).to(device)
        self.linear_1 = nn.Linear(self.hidden_dim, 1024)
        self.linear_2 = nn.Linear(1024, self.output_dim)
        self.embedding = nn.Embedding(self.input_dim, 50)

    def decode(self, x):
        layer = self.lstm_layer_1(x)  # -> L B D
        layer = layer.view(self.length_sentence, self.batch_size, self.hidden_dim).transpose(0, 1)
        return layer

    def find_output(self, x, mask):
        layer = self.lstm_layer_out(x)
        layer = self.linear_1(layer)
        layer = self.linear_2(layer)
        output = layer.view(self.length_sentence, self.batch_size, self.output_dim).transpose(0, 1)
        output = output * mask
        output = output.contiguous().view(self.length_sentence * self.batch_size, self.output_dim)
        return output

    def forward(self, inputs, mask, list_chars, mask_chars):
        # B x L x D and B x L x Lc x Dc
        self.batch_size = inputs.shape[0]
        self.length_sentence = inputs.shape[1]

        # # L x B x D
        inputs = self.embedding(inputs).view(self.batch_size, self.length_sentence, 50).transpose(0, 1)
        list_chars = list_chars.transpose(0, 1)  # L x B x Lc x Dc
        mask_chars = mask_chars.transpose(0, 1)

        x = Variable(torch.FloatTensor(np.empty(0))).to(device)
        for i in range(self.length_sentence):
            ch = self.lstm_layer_character(list_chars[i], mask_chars[i])  # --> Bx D
            # print ch
            in_tmp = torch.cat((inputs[i].transpose(0, 1), ch.transpose(0, 1)))\
                .view(self.input_lstm_dim, self.batch_size)\
                .transpose(0, 1)
            x = torch.cat((x, in_tmp))
        # L , B , D -> B, L , D
        x = x.view(self.length_sentence, self.batch_size, self.input_lstm_dim)\
            .transpose(0, 1)
        # x = inputs
        x = self.decode(x)  # out --> BxL hidden_D
        x = self.find_output(x, mask)  # out --> BxL output_D
        return x
