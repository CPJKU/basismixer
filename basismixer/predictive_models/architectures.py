
import numpy
import torch
from torch import nn

from basismixer.predictive_models.base import NNModel, standardize, get_nonlinearity


class FeedForwardModel(NNModel):
    """Simple Dense FFNN
    """

    def __init__(self,
                 input_size, output_size,
                 hidden_size, dropout=0.0,
                 nonlinearity=nn.ReLU(),
                 input_names=None,
                 output_names=None,
                 input_type=None,
                 dtype=torch.float32,
                 device=None):
        super().__init__(input_names=input_names,
                         output_names=output_names,
                         input_type=input_type,
                         dtype=dtype,
                         device=device,
                         is_rnn=False)

        self.input_size = input_size
        if not isinstance(hidden_size, (list, tuple)):
            hidden_size = [hidden_size]
        self.hidden_size = hidden_size
        self.output_size = output_size

        if not isinstance(dropout, (list, tuple)):
            self.dropout = len(self.hidden_size) * [dropout]
        else:
            if len(dropout) != len(self.hidden_size):
                raise ValueError('`dropout` should be the same length '
                                 'as `hidden_size`.')

        if not isinstance(nonlinearity, (list, tuple)):
            self.nonlinearity = len(self.hidden_size) * [nonlinearity]
        else:
            if len(nonlinearity) != len(self.hidden_size):
                raise ValueError('`nonlinearity` should be the same length ',
                                 'as `hidden_size`.')

        self.nonlinearity = [get_nonlinearity(nl) for nl in self.nonlinearity]

        if self.output_names is None:
            self.output_names = [str(i) for i in range(self.output_size)]

        in_features = input_size
        hidden_layers = []
        for hs, p, nl in zip(self.hidden_size, self.dropout, self.nonlinearity):
            hidden_layers.append(nn.Linear(in_features, hs))
            in_features = hs
            hidden_layers.append(nl)

            if p != 0:
                hidden_layers.append(nn.Dropout(p))

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output = nn.Linear(in_features=self.hidden_size[-1],
                                out_features=self.output_size)

    @standardize
    def forward(self, x):
        h = self.hidden_layers(x)
        output = self.output(h)
        return output


class RecurrentModel(NNModel):
    """Simple RNN
    """

    def __init__(self,
                 input_size, output_size,
                 recurrent_size, hidden_size,
                 n_layers=1, dropout=0.0,
                 recurrent_unit='GRU',
                 dense_nl=nn.ReLU(),
                 bidirectional=True,
                 batch_first=True,
                 input_names=None,
                 output_names=None,
                 input_type=None,
                 dtype=torch.float32,
                 device=None):
        super().__init__(input_names=input_names,
                         output_names=output_names,
                         input_type=input_type,
                         dtype=dtype,
                         device=device,
                         is_rnn=True)
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.recurrent_unit = recurrent_unit
        if recurrent_unit == 'GRU':
            recurrent_unit = nn.GRU
        elif recurrent_unit == 'LSTM':
            recurrent_unit = nn.LSTM
        else:
            raise Exception(recurrent_unit + "is not supported as recurrent unit")

        self.rnn = recurrent_unit(input_size, self.recurrent_size,
                          self.n_layers,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=self.bidirectional)
        dense_in_features = (self.recurrent_size * 2 if
                             self.bidirectional else self.recurrent_size)
        self.dense = nn.Linear(in_features=dense_in_features,
                               out_features=self.hidden_size)
        self.dense_nl = get_nonlinearity(dense_nl)
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=output_size)

        if self.output_names is None:
            self.output_names = [str(i) for i in range(self.output_size)]

    def init_hidden(self, x):
        if self.bidirectional:
            n_layers = 2 * self.n_layers
        else:
            n_layers = self.n_layers
        if self.recurrent_unit == 'LSTM':
            return (torch.zeros(n_layers, x.size(0), self.recurrent_size).type(x.type()),
                    torch.zeros(n_layers, x.size(0), self.recurrent_size).type(x.type()))
        return torch.zeros(n_layers, x.size(0), self.recurrent_size).type(x.type())

    @standardize
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = self.init_hidden(x)
        # tensor of shape (batch_size, seq_len, hidden_size*2) if bidirectional, tuple of 2 tensors if LSTM
        output, h = self.rnn(x, h0)
        flatten_shape = (self.recurrent_size * 2
                         if self.bidirectional else self.recurrent_size)
        dense = self.dense_nl(self.dense(output.contiguous().view(-1, flatten_shape)))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)

        return y
