import numpy as np
import torch

from torch.utils.data import Dataset, Sampler

class SubsequenceSampler(Sampler):

    def __init__(self, seq_length, data_source, epoch_size='auto',
                 seq_indices=None, shuffle=True,
                 random_state=np.random.RandomState(1984),
                 valid_sequences=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.seq_length = seq_length
        self.epoch_size = epoch_size

        self.seq_indices = seq_indices

        if self.seq_indices is None:

            if hasattr(self.data_source, 'seq_indices'):
                self.seq_indices = self.data_source.seq_indices
            else:
                raise ValueError('Dataset must have a list of indices for each'
                                 'unique sequence')

        self.shuffle = shuffle

        self.rng = random_state

        self.seq_lengths = [len(l) for l in self.seq_indices]

        if isinstance(self.seq_length, int):
            if any([sl < self.seq_length for sl in self.seq_lengths]):
                self.seq_length = [min(self.seq_lengths)] * len(self.seq_lengths)

            else:
                self.seq_length = [self.seq_length] * len(self.seq_lengths)

        elif self.seq_length == 'full':
            self.seq_length = self.seq_lengths

        self.valid_sequences = valid_sequences

        if self.valid_sequences is None:
            self.valid_sequences = np.arange(len(self.seq_indices))

    def __len__(self):

        if self.epoch_size == 'auto':

            return sum([sl for i, sl in enumerate(self.seq_lengths)
                        if i in self.valid_sequences]) // min(self.seq_length)
        elif isinstance(self.epoch_size, int):
            return self.epoch_size

    def __iter__(self):

        # select sequence
        if self.shuffle:
            sequences = self.rng.randint(0, len(self.valid_sequences), size=len(self))
        else:
            sequences = np.arange(0, len(self.valid_sequences))

        indices = []

        for pi in sequences:
            # select subsequence
            si = self.valid_sequences[pi]
            sl = self.seq_lengths[si]
            max_sl = self.seq_length[si]
            if self.shuffle and self.seq_length:
                start_idx = int(self.rng.randint(0, sl - max_sl))
            else:
                start_idx = 0
            indices.append(self.seq_indices[si][start_idx] + torch.arange(max_sl))

        return iter(indices)

class SimpleBaseDataset(Dataset):
    """
    Base class to create vanilla datasets
    """

    def __init__(self, dtype=torch.float, device=None):

        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

    def __len__(self):
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError

    def cuda(self):
        """
        Set device to CUDA
        """
        self.device = torch.device('cuda')
        self.to(self.device)

    def cpu(self):
        """
        Set device to CPU
        """
        self.device = torch.device('cpu')
        self.to(self.device)

    def __len__(self):
        return self.inputs.shape[0]

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx == len(self):
            raise StopIteration
        res = self[self.iter_idx]
        self.iter_idx += 1
        return res

    def __getitem__(self, idx):
        return NotImplementedError


class BasisMixerDataset(SimpleBaseDataset):
    """
    Base class for building datasets
    """

    def __init__(self, X, Y,
                 is_piecewise=True,
                 dtype=torch.float, device=None):

        super().__init__(dtype=dtype, device=device)
        # Name of all pieces in the dataset
        self.pieces = list(all_performances.keys())
        # Boolean indicating if the Dataset iterates onsetwise or piece-wise
        self.is_piecewise = is_piecewise

        # Initialize list for containing beat periods
        # this attribute will be a concatenate all
        # beat periods of all pieces into a single 1D array
        self.beat_periods = []
        # List of score IOIs (will be concatenated into a 1D array)
        self.s_iois = []
        # List of the score features (depending on the type of score features
        # if any), will either be None, a single 1D or 2D array or a list of
        # arrays (all with the same length as self.beat_periods)
        self.s_features = []
        # Performed onsets (will be concatenated into a 1D array)
        self.p_onsets = []
        # Index of the pieces to which each row in self.beat_periods,
        # self.s_iois, self.s_features and self.p_onsets belong
        # to.
        self.piece_indices = []
        # A list of lists containing the indices of the onsets for each piece
        # (which indices in self.beat_periods et al. belong to which piece)
        self.onset_indices = []

        # Append information to the arrays
        k = 0
        for pi, pn in enumerate(all_performances):
            # unpack performance information
            po, sioi, bp, sf = all_performances[pn]

            # Check that the sizes are correct
            assert len(po) == len(bp)
            assert len(sioi) == len(po) - 1
            if sf is not None:
                if isinstance(sf, (list, tuple)):
                    for s in sf:
                        assert len(s) == len(po)
                else:
                    assert len(sf) == len(po)

            # Extend lists
            self.p_onsets.append(po)
            self.beat_periods.append(bp)
            # the score IOIs are padded with a 0 at the begining so that they
            # have the same size as beat_periods and p_onsets (so that it is
            # easy to have acces to the correct elements using the indices)
            self.s_iois.append(np.r_[0, sioi])
            self.s_features.append(sf)
            self.piece_indices.append(np.ones_like(bp) * pi)
            self.onset_indices.append(np.arange(k, k + len(bp)).astype(np.int))
            k += len(bp)

        # Get main dtypes
        if isinstance(self.dtype, (list, tuple)):
            # dtype for beat period, s_iois and p_onsets
            dt = self.dtype[0]
            # dtypes for the score features
            sf_dt = self.dtype[1:]
        else:
            # dtype for beat period, s_iois and p_onsets
            dt = self.dtype
            # Use the same dtype for all score features
            sf_dt = self.dtype

        # Concatenate arrays of indices
        self.piece_indices = np.hstack(self.piece_indices).astype(np.int)
        self.all_indices = np.hstack(self.onset_indices).astype(np.int)
        # For using with SubsequenceSampler
        self.seq_indices = self.onset_indices

        # Concatenate performance and score information and cast it as torch.Tensors
        self.p_onsets = torch.from_numpy(np.hstack(self.p_onsets)).type(dt)
        self.beat_periods = torch.from_numpy(np.hstack(self.beat_periods)).type(dt)
        self.s_iois = torch.from_numpy(np.hstack(self.s_iois)).type(dt)
        if all([sf is not None for sf in self.s_features]):
            if isinstance(sf, (list, tuple)):
                cat_fun = []
                for sf in self.s_features[0]:
                    if sf.ndim == 1:
                        cat_fun.append(np.hstack)
                    elif sf.ndim > 1:
                        cat_fun.append(np.vstack)

                if isinstance(sf_dt, (list, tuple)):
                    assert len(sf_dt) == len(cat_fun)
                else:
                    sf_dt = [sf_dt] * len(cat_fun)

                s_features = []
                for ci, cf in enumerate(cat_fun):
                    s_features.append(torch.from_numpy(
                        cf([f[ci] for f in self.s_features])).type(sf_dt[ci]))

            else:
                if self.s_features[0].ndim == 1:
                    self.s_features = torch.from_numpy(np.hstack(self.s_features)).type(sf_dt)
                else:
                    self.s_features = torch.from_numpy(np.vstack(self.s_features)).type(st_dt)

        else:
            self.s_features = None

    def __len__(self):
        if self.is_piecewise:
            return len(self.pieces)
        else:
            return len(self.all_indices)

    def to(self, device):
        self.device = device
        self.p_onsets = self.p_onsets.to(device)
        self.beat_periods = self.beat_periods.to(device)
        self.s_iois = self.s_iois.to(device)

        if self.s_features is not None:
            if isinstance(self.s_features, list):
                self.s_features = [sf.to(device) for sf in self.s_features]
            else:
                self.s_features = self.s_features.to(device)

    def __getitem__(self, idx):
        if self.is_piecewise:
            idxs = self.onset_indices[idx]
        else:
            idxs = idx

        output = (self.p_onsets[idxs],
                  self.s_iois[idxs],
                  self.beat_periods[idxs])
        if self.s_features is not None:
            if isinstance(self.s_features, (list, tuple)):
                output += (s[idxs] for s in self.s_features)
            else:
                output += (self.s_features[idxs],)

        return output
