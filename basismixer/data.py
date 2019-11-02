#!/usr/bin/env python

import logging

import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from partitura import load_musicxml, load_match
from partitura.score import expand_grace_notes
from basismixer.basisfunctions import make_basis
from basismixer.utils import (pair_files,
                              get_unique_onset_idxs,
                              notewise_to_onsetwise)

from basismixer.performance_codec import get_performance_codec

LOGGER = logging.getLogger(__name__)


def make_datasets(model_specs, mxml_folder, match_folder, pieces=None):
    """Create an dataset for each in a list of model specifications.

    A model specification is a dictionary with the keys 'onsetwise'
    (bool), 'basis_functions' (a list of basis function names),
    'parameter_names' (a list of parameter names) and 'seq_len' (an integer). For
    example:

    { 
      'onsetwise': False,
      'basis_functions: ['polynomial_pitch_basis', 'duration_basis'],
      'parameter_names': ['velocity_trend', 'beat_period'],
      'seq_len': 1
    }

    The datasets are created based on pairs of MusicXML files and
    match files found in `mxml_folder` and `match_folder`
    respectively.

    Parameters
    ----------
    model_specs : list
        A list of dictionaries
    mxml_folder : str
        Path to folder with MusicXML files
    match_folder : str
        Path to folder with Match files
    pieces : list or None, optional
        If not None only pieces with a piecename occurring in the list
        are included in the datasets

    Returns
    -------
    list
        A list of triplets (ConcatDataSet, input_names, output_names) 
        with the same length as `model_specs`.
    
    
    """

    all_targets = list(set([n for model_spec in model_specs
                            for n in model_spec['parameter_names']]))

    perf_codec = get_performance_codec(all_targets)

    # different subsets of basis functions may be returned for different
    # pieces. idx_map maintains a global mapping from basis names to
    # indices into the columns of the model inputs.
    bf_idx_map = {}
    # a list to gather the data from which the dataset will be built.
    data = []

    all_basis_functions = set([n for model_spec in model_specs
                               for n in model_spec['basis_functions']])
    folders = dict(mxml=mxml_folder, match=match_folder)

    for piece, files in pair_files(folders).items():
        if pieces is not None and piece not in pieces:
            continue
        # load the score
        xml_fn = files['mxml'].pop()
        LOGGER.info('Processing {}'.format(xml_fn))
        part = load_musicxml(xml_fn)

        # get indices of the unique onsets
        # expand grace note durations (necessary for correct computation of
        # targets)
        expand_grace_notes(part)

        # compute the basis functions
        basis, bf_names = make_basis(part, all_basis_functions)

        # map the basis names returned for this piece to their global
        # indices
        bf_idx = np.array([bf_idx_map.setdefault(name, len(bf_idx_map))
                           for i, name in enumerate(bf_names)])


        # a dictionary from note id to index. We need this to select the
        # subset of rows in the `basis` array that have a matching entry in
        # the targets.
        nid_dict = dict((n.id, i) for i, n in enumerate(part.notes_tied))

        for match in files['match']:

            # if not '_p01' in match:
            #     continue
            if 'Take' in match:
                continue
            LOGGER.info('Processing {}'.format(match))

            # load the performed part and the alignment from the match file
            ppart, alignment = load_match(match, first_note_at_zero=True)

            # For Magaloff dataset (remove repetititions)
            for n in alignment:
                if n['label'] == 'match':
                    n['score_id'] = n['score_id'].split('-')[0]

            # compute the targets
            targets, snote_ids = perf_codec.encode(part, ppart, alignment)

            matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])
            basis_matched = basis[matched_subset_idxs]

            score_onsets = part.note_array[matched_subset_idxs]['onset']
            unique_onset_idxs = get_unique_onset_idxs(score_onsets)

            data.append((basis_matched, bf_idx, targets, unique_onset_idxs))
    
    return piece_data_to_datasets(data, bf_idx_map, model_specs)


def piece_data_to_datasets(data, bf_idx_map, model_specs):
    # total number of basis functions in the dataset
    #n_basis = len(bf_idx_map)
    idx_bf_map = dict((v, k) for k, v in bf_idx_map.items())
    # print(bf_idx_map)
    # print(bf_idx_inv_map)
    input_names = np.array([idx_bf_map[i] for i in range(len(idx_bf_map))])

    input_basis = np.array([n.split('.', 1)[0] for n in input_names])

    dataset_per_model = []
    input_names_per_model = []
    output_names_per_model = []
    for m_spec in model_specs:
        # the global indices of the basis functions that this model needs
        model_idx = np.concatenate([np.where(input_basis==n)[0]
                                    for n in m_spec['basis_functions']])
        # trg_idx = np.array([perf_codec.parameter_names.index(n) for n in m_spec['targets']])
        n_basis = len(model_idx)

        input_names_per_model.append(input_names[model_idx])
        output_names_per_model.append(m_spec['parameter_names'])

        m_datasets = []
        m_input_names = []

        for bf, idx, targets, uox in data:
            # idx: the global indices that this piece has

            # the subset of basisfunctions that this model is interested in:
            useful = np.isin(idx, model_idx)
            # idx mapped to the subset of basisfunctions for this model
            model_idx_subset = np.array([np.where(model_idx==i)[0][0]
                                         for i in idx[useful]])

            # select only the required bfs
            bf = bf[:, useful]
            # select only the required targets
            targets = np.array([targets[n] for n in m_spec['parameter_names']]).T
            
            if m_spec['onsetwise']:

                bf = notewise_to_onsetwise(bf, uox)
                targets = notewise_to_onsetwise(targets, uox)
            
            ds = BasisMixerDataSet(bf, model_idx_subset, n_basis,
                                   targets, m_spec['seq_len'])
            
            m_datasets.append(ds)


        dataset_per_model.append(ConcatDataset(m_datasets))
        
    return zip(dataset_per_model, input_names_per_model, output_names_per_model)


class BasisMixerDataSet(Dataset):
    """A Dataset to train basis mixer models.

    The dataset corresponds to a single peformance of a piece, and
    holds both the basis function representation of the score, and the
    expressive parameters (targets) that encode the performance.

    Since the basis function representation may hold different subsets
    of the total set of basis functions for different pieces, the
    dataset holds the indices of the current basis functions into the
    global basis function representation (attribute `idx`). When an
    item is retrieved from the dataset the dense basis function
    representation for the current piece is "blown up" to the size of
    the global basis function representation (attribute `n_basis`),
    having zeros for all basis functions not present in this piece.

    The `seq_len` parameter determines the number of consecutive rows
    from `basis`/`targets` that make up a single data instance.

    Parameters
    ----------
    basis : ndarray, shape (N, K)
        A basis function representation of K basis for N notes
    idx : ndarray, shape (K,)
        An array of indices into the total number of basis for each
        basis in `basis`. The values in `idx` should be less than
        `n_basis`.
    n_basis : int
        The total number of basis functions
    targets : ndarray, shape (N, L)
        The representation of the performance as L expressive
        parameters for N notes
    seq_len : int, optional
        The sequence length. If `seq_len` > `N` the Dataset will have
        zero-length. Defaults to 1.

    Attributes
    ----------
    basis : ndarray, shape (N, K)
        See Parameters Section
    idx : ndarray, shape (K,)
        See Parameters Section
    n_basis : int
        See Parameters Section
    targets : ndarray, shape (N, L)
        See Parameters Section
    seq_len : int
        See Parameters Section.

    """
    def __init__(self, basis, idx, n_basis, targets, seq_len=1):
        self.basis = basis
        self.idx = idx
        self.n_basis = n_basis
        self.targets = targets
        self.seq_len = seq_len

    def __getitem__(self, i):

        if i + self.seq_len > len(self.basis):
            raise IndexError

        x = np.zeros((self.seq_len, self.n_basis))
        x[:, self.idx] = self.basis[i:i + self.seq_len]

        y = self.targets[i:i + self.seq_len, :]

        return x, y

    def __len__(self):
        return max(0, len(self.basis) - self.seq_len)

# def make_dataset(mxml_folder, match_folder, basis_functions, perf_codec, seq_len,
#                  aggregate_onsetwise=False, valid_pieces=None):
#     """Create a dataset from the MusicXML and Match files.

#     Parameters
#     ----------
#     mxml_folder : str
#         Path to folder with MusicXML files
#     match_folder : str
#         Path to folder with Match files
#     basis_functions : list
#         A list of elements that can be either functions or basis
#         function names as strings.
#     perf_codec : :class:`~basismixer.performance_codec.PerformanceCodec`
#         Performance codec to compute the expressive parameters.
#     seq_len : int
#         The sequence length for the dataset.

#     Returns
#     -------
#     ConcatDataSet
#         Datasets for each performance concatenated into a single
#         dataset

#     """

#     # different subsets of basis functions may be returned for different
#     # pieces. idx_map maintains a global mapping from basis names to
#     # indices into the columns of the model inputs.
#     bf_idx_map = {}
#     # a list to gather the data from which the dataset will be built.
#     data = []

#     folders = dict(mxml=mxml_folder, match=match_folder)
#     for files in pair_files(folders, valid_pieces=valid_pieces).values():
#         # files is a dictionary with keys 'mxml', and 'match'. The
#         # corresponding values are sets of filenames. In our case there
#         # will be a single musicxml file and multiple match files.

#         # load the score
#         xml_fn = files['mxml'].pop()
#         LOGGER.info('Processing {}'.format(xml_fn))
#         part = load_musicxml(xml_fn)

#         # get indices of the unique onsets
#         # expand grace note durations (necessary for correct computation of
#         # targets)
#         expand_grace_notes(part)

#         # compute the basis functions
#         basis, bf_names = make_basis(part, basis_functions)

#         # map the basis names returned for this piece to their global
#         # indices
#         bf_idx = np.array([bf_idx_map.setdefault(name, len(bf_idx_map))
#                            for i, name in enumerate(bf_names)])

#         # a dictionary from note id to index. We need this to select the
#         # subset of rows in the `basis` array that have a matching entry in
#         # the targets.
#         nid_dict = dict((n.id, i) for i, n in enumerate(part.notes_tied))

#         for match in files['match']:

#             LOGGER.info('Processing {}'.format(match))

#             # load the performed part and the alignment from the match file
#             ppart, alignment = load_match(match, first_note_at_zero=True)

#             # compute the targets
#             targets, snote_ids = perf_codec.encode(part, ppart, alignment)
#             # convert the targets to a regular array
#             targets = targets.view(np.float32).reshape((len(targets), -1))

#             # select the subset of rows in the `basis` array that have a
#             # matching entry in the targets

#             matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])
#             basis_matched = basis[matched_subset_idxs]

#             if aggregate_onsetwise:
#                 score_onsets = part.note_array[matched_subset_idxs]['onset']
#                 unique_onset_idxs = get_unique_onset_idxs(score_onsets)
#                 basis_matched = notewise_to_onsetwise(basis_matched, unique_onset_idxs)
#                 targets = notewise_to_onsetwise(targets, unique_onset_idxs)

#             data.append((basis_matched, bf_idx, targets))

#             # FOR DEVELOPMENT
#             # break

#     # total number of basis functions in the dataset
#     n_basis = len(bf_idx_map)

#     output_names = perf_codec.parameter_names
#     bf_idx_inv_map = dict((v, k) for k, v in bf_idx_map.items())
#     input_names = [bf_idx_inv_map[i] for i in range(n_basis)]

#     # create and concatenate the datasets for each performance
#     dataset = ConcatDataset([BasisMixerDataSet(basis, idx, n_basis,
#                                                targets, seq_len)
#                              for basis, idx, targets in data])

#     return dataset, input_names, output_names


# class BasisMixerDataSet(Dataset):
#     """A Dataset to train basis mixer models.

#     The dataset corresponds to a single peformance of a piece, and
#     holds both the basis function representation of the score, and the
#     expressive parameters (targets) that encode the performance.

#     Since the basis function representation may hold different subsets
#     of the total set of basis functions for different pieces, the
#     dataset holds the indices of the current basis functions into the
#     global basis function representation (attribute `idx`). When an
#     item is retrieved from the dataset the dense basis function
#     representation for the current piece is "blown up" to the size of
#     the global basis function representation (attribute `n_basis`),
#     having zeros for all basis functions not present in this piece.

#     The `seq_len` parameter determines the number of consecutive rows
#     from `basis`/`targets` that make up a single data instance.

#     Parameters
#     ----------
#     basis : ndarray, shape (N, K)
#         A basis function representation of K basis for N notes
#     idx : ndarray, shape (K,)
#         An array of indices into the total number of basis for each
#         basis in `basis`. The values in `idx` should be less than
#         `n_basis`.
#     n_basis : int
#         The total number of basis functions
#     targets : ndarray, shape (N, L)
#         The representation of the performance as L expressive
#         parameters for N notes
#     seq_len : int, optional
#         The sequence length. If `seq_len` > `N` the Dataset will have
#         zero-length. Defaults to 1.
#     targets_idx : ndarray, shape(M, ) or None (optional)
#         An array of indices of the targets to use for training.
#         The values in `targets_idx` should be less than L. If `None`,
#         all targets in the dataset will be used.

#     Attributes
#     ----------
#     basis : ndarray, shape (N, K)
#         See Parameters Section
#     idx : ndarray, shape (K,)
#         See Parameters Section
#     n_basis : int
#         See Parameters Section
#     targets : ndarray, shape (N, L)
#         See Parameters Section
#     seq_len : int
#         See Parameters Section.

#     """

#     def __init__(self, basis, idx, n_basis, targets, seq_len=1, targets_idx=None):

#         self.basis = basis
#         self.idx = idx
#         self.n_basis = n_basis
#         self.targets = targets
#         self.seq_len = seq_len
#         if targets_idx is None:
#             self.targets_idx = np.arange(targets.shape[1], dtype=np.int)
#         else:
#             self.targets_idx

#     def __getitem__(self, i):

#         if i + self.seq_len > len(self.basis):
#             raise IndexError

#         x = np.zeros((self.seq_len, self.n_basis))
#         x[:, self.idx] = self.basis[i:i + self.seq_len]

#         y = self.targets[i:i + self.seq_len, self.targets_idx]

#         return x, y

#     def __len__(self):
#         return max(0, len(self.basis) - self.seq_len)


# class BasisMixerDataSet(torch.utils.data.Dataset):
#     def __init__(self, basis, idx, n_basis, targets):
#         self.basis = basis
#         self.idx = idx
#         self.n_basis = n_basis
#         self.targets = targets
#     def __getitem__(self, i):
#         x = np.zeros(self.n_basis)
#         x[self.idx] = self.basis[i]
#         y = self.targets[i, :]
#         return x, y
#     def __len__(self):
#         return len(self.basis)
