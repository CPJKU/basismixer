from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
from torch import nn


class FullPredictiveModel(object):
    """Meta model for predicting an expressive performance
    """

    def __init__(self, models, input_names, output_names,
                 default_values={},
                 overlapping_output_strategy='FIFO'):

        self.models = models
        self.input_names = np.array(input_names)
        self.output_names = np.array(output_names)
        self.default_values = default_values
        self.overlapping_output_strategy = overlapping_output_strategy

        # check that there is a default value for each expressive parameter
        if len(set(default_values.keys()).difference(set(self.output_names))) != 0:
            raise KeyError('`default_values` must contain a value for each '
                           'parameter in `output_names`.')

        # indices of the basis functions for each model
        self.model_bf_idxs = []
        for model in self.models:
            bf_idxs = np.array([int(np.where(self.input_names == bf)[0])
                                for bf in model.input_names])
            self.model_bf_idxs.append(bf_idxs)

        # indices of the models for each parameter
        self.model_param_idxs = defaultdict(list)
        for pn in self.output_names:
            for i, model in enumerate(self.models):
                if pn in model.output_names:
                    self.model_param_idxs[pn].append(i)

    def predict(self, x, score_onsets):

        if x.ndim != 2:
            raise ValueError('The inputs should be a 2D array')

        _predictions = []

        for bf_idxs, model in zip(self.model_bf_idxs, self.models):
            # Get slice of the input corresponding to the bfs
            # used in the model
            model_input = x[:, bf_idxs]

            # aggregate bfs per onset
            if model.input_type == 'onsetwise':
                model_input, unique_onset_idxs = aggregate_onsetwise_bfs(model_input,
                                                                         score_onsets)
            # make predictions
            preds = model.predict(model_input)

            # expand predictions per each note
            if model.input_type == 'onsetwise':
                preds = expand_onsetwise_preds(preds, unique_onset_idxs)

            _predictions.append(preds)

        # structured array for holding expressive parameters
        predictions = np.zeros(len(score_onsets),
                               dtype=[(pn, 'f4') for pn in self.output_names])
        # assign predictions according to the overlapping strategy
        # or default value
        for pn in self.output_names:
            model_idxs = self.model_param_idxs[pn]
            if len(model_idxs) > 0:
                if self.overlapping_output_strategy == 'FIFO':
                    predictions[pn] = _predictions[model_idxs[0]][pn]

                elif self.overlapping_output_strategy == 'mean':
                    predictions[pn] = np.mean(
                        np.column_stack([_predictions[mix][pn] for mix in model_idxs]),
                        axis=1)
            else:
                predictions[pn] = np.ones(len(score_onsets)) * self.default_values[pn]
        return predictions


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

    @abstractmethod
    def save_model(self, filename):
        pass

    def predict(self, x):
        """
        Predict
        """
        # reshape for recurrent models
        do_reshape = self.is_rnn and len(x.shape) < 3
        if do_reshape:
            x = x[np.newaxis]

        if isinstance(self, nn.Module):
            mx = torch.tensor(x).type(self.dtype).to(self.device)
        else:
            mx = x
        # model predictions
        predictions = self(mx)

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


class FeedForwardModel(nn.Module, PredictiveModel):
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
        nn.Module.__init__(self)
        PredictiveModel.__init__(self,
                                 input_names=input_names,
                                 output_names=output_names,
                                 is_rnn=False,
                                 input_type=input_type)

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

        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')

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

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        self.type(dtype)

    def forward(self, x):
        h = self.hidden_layers(x)
        output = self.output(h)
        return output


class RecurrentModel(nn.Module, PredictiveModel):
    """Simple RNN
    """

    def __init__(self,
                 input_size, output_size,
                 recurrent_size, hidden_size,
                 n_layers, dropout=0.0,
                 batch_first=True,
                 input_names=None,
                 output_names=None,
                 input_type=None,
                 dtype=torch.float32,
                 device=None):
        nn.Module.__init__(self)
        PredictiveModel.__init__(self,
                                 input_names=input_names,
                                 output_names=output_names,
                                 is_rnn=True,
                                 input_type=input_type)

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

        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        self.type(dtype)

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
        return torch.zeros(self.n_layers, batch_size, self.recurrent_size)

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

        output, h = self.rnn(x, h0)

        if isinstance(output, nn.utils.rnn.PackedSequence):
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        dense = self.dense(output.contiguous().view(-1, self.recurrent_size))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)

        return y


def aggregate_onsetwise_bfs(notewise_inputs, score_onsets):
    """Agregate basis functions per onset
    """
    unique_score_onsets = np.unique(score_onsets)
    unique_onset_idxs = [np.where(score_onsets == u)[0]
                         for u in unique_score_onsets]

    onsetwise_notewise_inputs = np.zeros((len(unique_score_onsets),
                                          notewise_inputs.shape[1]))

    for i, uix in enumerate(unique_onset_idxs):
        onsetwise_notewise_inputs[i] = notewise_inputs[uix].mean(0)

    return onsetwise_notewise_inputs, unique_onset_idxs


def expand_onsetwise_preds(onsetwise_predictions, unique_onset_idxs):
    """Expand onsetwise predictions for each note
    """
    n_notes = sum([len(uix) for uix in unique_onset_idxs])
    notewise_predictions = np.zeros(n_notes, dtype=onsetwise_predictions.dtype)

    for i, uix in enumerate(unique_onset_idxs):
        notewise_predictions[uix] = onsetwise_predictions[[i]]

    return notewise_predictions
