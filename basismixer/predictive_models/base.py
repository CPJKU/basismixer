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


class NNModel(nn.Module, PredictiveModel):
    """
    Base model for Neural Network Models
    """

    def __init__(self, input_names=None,
                 output_names=None,
                 input_type=None,
                 dtype=torch.float32,
                 device=None,
                 is_rnn=False):
        nn.Module.__init__(self)
        PredictiveModel.__init__(self,
                                 input_names=input_names,
                                 output_names=output_names,
                                 is_rnn=is_rnn,
                                 input_type=input_type)
        self.dtype = dtype
        self.device = device if device is not None else torch.device('cpu')
        self.to(self.device)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        self.type(dtype)


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
