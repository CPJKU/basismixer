from abc import ABC, abstractmethod

import numpy as np
import torch

from torch import nn


class PredictiveModel(ABC):
    """
    Predictive Model
    """

    def __init__(self, input_names=None,
                 output_names=None,
                 is_rnn=False,
                 input_type=None):
        # name of input features
        self.input_names = input_names
        # name of output features
        self.output_names = output_names
        # if the model is
        self.is_rnn = is_rnn
        # if the input is onsetwise or notewise
        self.input_type = input_type

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """The predictive model is a callable itself that computes an
        output (i.e., predictions of expressive parameters) given some
        inputs. This method has to be implemented in each subclass
        """
        pass

    def predict(self, x):
        """
        Predict
        """
        # reshape for recurrent models
        do_reshape = self.is_rnn and len(x.shape) < 3
        if do_reshape:
            x = x[np.newaxis]

        # model predictions
        predictions = self(x)

        # predictions as numpy array
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.data.detach().cpu().numpy()

        if do_reshape:
            predictions = predictions[0]

        # Output as structured array
        if predictions.ndim < 3:
            output = np.zeros(len(predictions),
                              dtype=[(pn, 'f4') for pn in self.output_names])
            for i, pn in enumerate(self.output_names):
                output[pn] = predictions[:, i]
        else:
            output = []
            for p in predictions:
                out = np.zeros(len(p),
                               dtype=[(pn, 'f4') for pn in self.output_names])
                for i, pn in enumerate(self.output_names):
                    out[pn] = p[:, i]
                output.append(out)
            output = np.array(output)

        return output


class RecurrentModel(nn.Module, PredictiveModel):
    def __init__(self,
                 input_size, output_size,
                 recurrent_size, hidden_size,
                 n_layers, dropout=0.0,
                 batch_first=True,
                 input_names=None,
                 output_names=None,
                 input_type=None):
        nn.Module.__init__(self)
        PredictiveModel.__init__(self,
                                 input_names=input_names,
                                 output_names=output_names,
                                 is_rnn=True,
                                 input_type=input_names)

        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.rnn = nn.GRU(input_size, self.recurrent_size,
                          self.n_layers,
                          batch_first=batch_first, dropout=dropout)
        self.dense = nn.Linear(in_features=self.recurrent_size,
                               out_features=self.hidden_size)
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=output_size)

        if self.output_names is None:
            self.output_names = [str(i) for i in range(self.output_size)]

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.recurrent_size)

    def forward(self, x, origin_len=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = self.init_hidden(batch_size).type(x.type())

        # pack_padded_sequence() prevents rnns to process padded data
        if origin_len is not None:
            origin_len = torch.as_tensor(
                origin_len, dtype=torch.int64, device='cpu')
            x = nn.utils.rnn.pack_padded_sequence(
                x, origin_len, batch_first=True, enforce_sorted=False)

        output, h = self.rnn(x, h0)

        if isinstance(output, nn.utils.rnn.PackedSequence):
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        dense = self.dense(output.contiguous().view(-1, self.recurrent_size))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)

        return y
