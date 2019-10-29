import numpy
import torch
from torch import nn

from basismixer.predictive_models.base import NNModel


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

    def save_model(self, filename):
        state = dict(
            arch=type(self).__name__,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            nonlinearity=self.nonlinearity,
            state_dict=self.state_dict(),
            input_names=self.input_names,
            output_names=self.output_names,
            is_rnn=self.is_rnn,
            input_type=self.input_type,
        )
        torch.save(state, filename)

    @classmethod
    def load_model(cls, filename):
        try:
            kwargs = torch.load(model_fn)
        except RuntimeError:
            kwargs = torch.load(model_fn,
                                map_location='cpu')
        state_dict = kwargs.pop('state_dict')
        kwargs.pop('arch')
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model

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
                 n_layers, dropout=0.0,
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
        self.rnn = nn.GRU(input_size, self.recurrent_size,
                          self.n_layers,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=self.bidirectional)
        dense_in_features = (self.recurrent_size * 2 if
                             self.bidirectional else self.recurrent_size)
        self.dense = nn.Linear(in_features=dense_in_features,
                               out_features=self.hidden_size)
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=output_size)

        if self.output_names is None:
            self.output_names = [str(i) for i in range(self.output_size)]

    def save_model(self, filename):
        state = dict(
            arch=type(self).__name__,
            input_size=self.input_size,
            output_size=self.output_size,
            recurrent_size=self.recurrent_size,
            n_layers=self.n_layers,
            batch_first=self.batch_first,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            nonlinearity=self.nonlinearity,
            state_dict=self.state_dict(),
            input_names=self.input_names,
            output_names=self.output_names,
            is_rnn=self.is_rnn,
            input_type=self.input_type,
            dtype=self.dtype,
            device=self.device
        )
        torch.save(state, filename)

    @classmethod
    def load_model(cls, filename):
        try:
            kwargs = torch.load(model_fn)
        except RuntimeError:
            kwargs = torch.load(model_fn,
                                map_location='cpu')
        state_dict = kwargs.pop('state_dict')
        kwargs.pop('arch')
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model

    def init_hidden(self, batch_size):
        if self.bidirectional:
            n_layers = 2 * self.n_layers
        else:
            n_layers = self.n_layers
        return torch.zeros(n_layers, batch_size, self.recurrent_size)

    def forward(self, x, original_len=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = self.init_hidden(batch_size).type(x.type())

        # pack_padded_sequence() prevents rnns to process padded data
        if original_len is not None:
            original_len = torch.as_tensor(
                original_len, dtype=torch.int64, device='cpu')
            x = nn.utils.rnn.pack_padded_sequence(
                x, original_len, batch_first=True, enforce_sorted=False)

        # tensor of shape (batch_size, seq_len, hidden_size*2) if bidirectional
        output, h = self.rnn(x, h0)

        if isinstance(output, nn.utils.rnn.PackedSequence):
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        flatten_shape = (self.recurrent_size * 2
                         if self.bidirectional else self.recurrent_size)
        dense = self.dense(output.contiguous().view(-1, flatten_shape))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)

        return y
