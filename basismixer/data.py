#!/usr/bin/env python

import logging
import multiprocessing
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import partitura.musicanalysis
from partitura import load_musicxml, load_match
from partitura.score import expand_grace_notes
from torch.utils.data import Dataset

from basismixer.performance_codec import get_performance_codec
from basismixer.utils import (pair_files,
                              get_unique_onset_idxs,
                              notewise_to_onsetwise)
from .parse_tsv_alignment import load_alignment_from_ASAP

LOGGER = logging.getLogger(__name__)

from partitura.score import GraceNote, Note


def remove_grace_notes(part):
    """Remove all grace notes from a timeline.

    The specified timeline object will be modified in place.

    Parameters
    ----------
    timeline : Timeline
        The timeline from which to remove the grace notes

    """
    for gn in list(part.iter_all(GraceNote)):
        for n in list(part.iter_all(Note)):
            if n.tie_next == gn:
               n.tie_next = None
        part.remove(gn)


def process_piece(piece_performances, perf_codec, all_basis_functions, gracenotes, dataset_name):
    piece, performances = piece_performances
    data = []
    quirks = False
    if dataset_name == 'asap':
        name = '/'.join(str(piece).split('asap')[1].split('/')[1:-1])
    else:
        name = piece.split('/')[-1].split('.')[0]
        #quirks = True

    LOGGER.info('Processing {}'.format(piece))

    part = load_musicxml(piece)
    part = partitura.score.merge_parts(part)
    part = partitura.score.unfold_part_maximal(part, update_ids=dataset_name != '4x22')
    bm = part.beat_map

    # get indices of the unique onsets
    if gracenotes == 'remove':
        # Remove grace notes
        remove_grace_notes(part)
    else:
        # expand grace note durations (necessary for correct computation of
        # targets)
        expand_grace_notes(part)
    basis, bf_names = partitura.musicanalysis.make_note_feats(part, list(all_basis_functions))

    nid_dict = dict((n.id, i) for i, n in enumerate(part.notes_tied))

    for performance in performances:
        if dataset_name == 'asap':
            alignment = load_alignment_from_ASAP(performance)
            ppart = partitura.load_performance_midi(str(performance).split("_note_alignments/")[0] + ".mid")
        else:
            ppart, alignment = load_match(performance, first_note_at_zero=True)

            #if quirks: todo: check if quirks are really needed
            #    for n in alignment:
            #        if n['label'] == 'match':
            #            n['score_id'] = n['score_id'].split('-')[0]

        assert len(ppart.performedparts) == 1
        ppart = ppart.performedparts[0]

        # compute the targets
        targets, snote_ids = perf_codec.encode(part, ppart, alignment)

        matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])
        basis_matched = basis[matched_subset_idxs]

        score_onsets = bm([n.start.t for n in part.notes_tied])[matched_subset_idxs]
        unique_onset_idxs = get_unique_onset_idxs(score_onsets)

        i = -2 if dataset_name == 'asap' else -1

        performance_name = str(performance).split('/')[i]

        data.append((basis_matched, bf_names, targets, unique_onset_idxs, name, performance_name))
    return data


class ProcessPiece:
    def __init__(self, args):
        self.args = args

    def __call__(self, piece):
        return process_piece(piece, *self.args)


def filter_blocklist(pieces):
    blocklist = ['Liszt/Sonata', ]
    pieces_filtered = []
    for p in pieces:
        flag = True
        for b in blocklist:
            if b in str(p):
                flag = False
        if flag:
            pieces_filtered.append(p)
    print(f"filtered out {len(pieces) - len(pieces_filtered)} pieces!")
    return pieces_filtered


def make_datasets(model_specs, root_folder, dataset_name, gracenotes='remove', processes=0):
    assert dataset_name in ['4x22', 'magaloff', 'asap']

    quirks = dataset_name == 'magaloff'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_targets = list(set([n for model_spec in model_specs
                                for n in model_spec['parameter_names']]))

        perf_codec = get_performance_codec(all_targets)

        bf_idx_map = {}

        all_basis_functions = set([n for model_spec in model_specs
                                   for n in model_spec['basis_functions']])

        if dataset_name == 'asap':#todo: fix loading of Liszt/Sonata
            assert 'asap' in root_folder.split('/')[-1], 'Root folder name must contain "asap"'
            pieces = list(Path(root_folder).rglob("*/xml_score.musicxml"))
            pieces = filter_blocklist(pieces)
            performances = [list(Path(piece).parent.glob("*_note_alignments/note_alignment.tsv")) for piece in pieces]
            piece_performances = zip(pieces, performances)
        else:
            mxml_folder = root_folder + ('xml' if dataset_name == 'magaloff' else 'musicxml')
            match_folder = root_folder + 'match'
            folders = dict(mxml=mxml_folder, match=match_folder)
            paired_files = pair_files(folders, by_prefix=not quirks)
            piece_performances = []#[(pf['mxml'][0], list(pf['match'])) for pf in paired_files]
            for pf in paired_files.items():
                if 'chopin_op35_Mv3' in pf[0]:#todo: repair loading, do not filter...
                    continue
                piece_performances.append((list(pf[1]['mxml'])[0], list(pf[1]['match'])))

        if processes <= 0:
            processes = multiprocessing.cpu_count()

        if processes > 1:
            pool = Pool(processes)
            pieces = list(pool.map(ProcessPiece((perf_codec, all_basis_functions, gracenotes, dataset_name)), piece_performances))
        else:
            pieces = [process_piece(p, perf_codec, all_basis_functions, gracenotes, dataset_name) for p in piece_performances]
        pieces = [list(i) for sublist in pieces for i in sublist]

        for piece in pieces:
            bf_idx = np.array([bf_idx_map.setdefault(name, len(bf_idx_map))
                               for i, name in enumerate(piece[1])])
            piece[1] = bf_idx

        data = [tuple(l) for l in pieces]

        return piece_data_to_datasets(data, bf_idx_map, model_specs)


def piece_data_to_datasets(data, bf_idx_map, model_specs):
    # total number of basis functions in the dataset
    # n_basis = len(bf_idx_map)
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
        model_idx = np.concatenate([np.where(input_basis == n)[0]
                                    for n in m_spec['basis_functions']])
        # trg_idx = np.array([perf_codec.parameter_names.index(n) for n in m_spec['targets']])
        n_basis = len(model_idx)

        input_names_per_model.append(input_names[model_idx])
        output_names_per_model.append(m_spec['parameter_names'])

        m_datasets = []
        m_input_names = []

        for bf, idx, targets, uox, name, perf_name in data:
            # idx: the global indices that this piece has

            # the subset of basisfunctions that this model is interested in:
            useful = np.isin(idx, model_idx)
            # idx mapped to the subset of basisfunctions for this model
            model_idx_subset = np.array([np.where(model_idx == i)[0][0]
                                         for i in idx[useful]])

            # select only the required bfs
            bf = bf[:, useful]
            # select only the required targets
            targets = np.array([targets[n] for n in m_spec['parameter_names']]).T

            if m_spec['onsetwise']:
                bf = notewise_to_onsetwise(bf, uox)
                targets = notewise_to_onsetwise(targets, uox)

            ds = BasisMixerDataSet(bf, model_idx_subset, n_basis, targets,
                                   input_names_per_model[-1], output_names_per_model[-1],
                                   m_spec['seq_len'], name, perf_name)

            m_datasets.append(ds)

        dataset_per_model.append(m_datasets)

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
    name : str or None, optional
        A name for the dataset. Defaults to None

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
    name : str or None
        See Parameters Section.

    """

    def __init__(self, basis, idx, n_basis, targets, input_names, output_names, seq_len=1, name=None, perf_name=None):
        self.basis = basis
        self.idx = idx
        self.n_basis = n_basis
        self.targets = targets
        self.seq_len = seq_len
        self.name = name
        self.perf_name = perf_name
        self.input_names = input_names
        self.output_names = output_names

    @property
    def piecewise(self):
        return self.seq_len == -1

    def __getitem__(self, i):
        if self.piecewise:
            return self._get_item_piecewise(i)
        else:
            return self._get_item_sequencewise(i)

    def _get_item_piecewise(self, i):
        if i > 0:
            raise IndexError
        x = np.zeros((len(self.basis), self.n_basis))
        x[:, self.idx] = self.basis

        return x, self.targets

    def _get_item_sequencewise(self, i):
        if i + self.seq_len > len(self.basis):
            raise IndexError

        x = np.zeros((self.seq_len, self.n_basis))
        x[:, self.idx] = self.basis[i:i + self.seq_len]

        y = self.targets[i:i + self.seq_len, :]

        return x, y

    def __len__(self):
        if self.piecewise:
            return 1
        else:
            return max(0, len(self.basis) - self.seq_len)
