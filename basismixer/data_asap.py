#!/usr/bin/env python

import logging
import os

import warnings

import numpy as np
import partitura.musicanalysis
from torch.utils.data import Dataset, ConcatDataset

from partitura import load_musicxml, load_match
from partitura.score import expand_grace_notes
from basismixer.utils import (pair_files,
                              get_unique_onset_idxs,
                              notewise_to_onsetwise)
from .data import piece_data_to_datasets
from basismixer.performance_codec import get_performance_codec
from .parse_tsv_aligment import load_alignment_from_ASAP
from partitura.performance import PerformedPart
from pathlib import Path
from multiprocessing import Pool
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


def process_piece(piece, root_folder, perf_codec, all_basis_functions, gracenotes):
    data = []
    name = str(piece).split(root_folder + '/')[1].split('/xml_score.musicxml')[0]
    LOGGER.info('Processing {}'.format(piece))

    part = load_musicxml(piece)
    part = partitura.score.merge_parts(part)
    part = partitura.score.unfold_part_maximal(part)
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

    performances = list(Path(piece).parent.glob("*_note_alignments/note_alignment.tsv"))

    for performance in performances:
        alignment = load_alignment_from_ASAP(performance)
        ppart = partitura.load_performance_midi(str(performance).split("_note_alignments/")[0] + ".mid")

        # compute the targets
        targets, snote_ids = perf_codec.encode(part, ppart, alignment)

        matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])
        basis_matched = basis[matched_subset_idxs]

        score_onsets = bm([n.start.t for n in part.notes_tied])[matched_subset_idxs]
        unique_onset_idxs = get_unique_onset_idxs(score_onsets)

        performance_name = str(performance).split('/')[-2]

        data.append((basis_matched, bf_names, targets, unique_onset_idxs, name, performance_name))
    return data


class ProcessPiece:
    def __init__(self, args):
        self.args = args

    def __call__(self, piece):
        return process_piece(piece, *self.args)


def make_datasets(model_specs, root_folder, quirks=False, gracenotes='remove'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_targets = list(set([n for model_spec in model_specs
                                for n in model_spec['parameter_names']]))

        perf_codec = get_performance_codec(all_targets)

        bf_idx_map = {}

        all_basis_functions = set([n for model_spec in model_specs
                                   for n in model_spec['basis_functions']])


        pieces = list(Path(root_folder).rglob("*/xml_score.musicxml"))
        pool = Pool(40)

        pieces = list(pool.map(ProcessPiece((root_folder, perf_codec, all_basis_functions, gracenotes)), pieces))
        pieces = [list(i) for sublist in pieces for i in sublist]

        for piece in pieces:
            bf_idx = np.array([bf_idx_map.setdefault(name, len(bf_idx_map))
                               for i, name in enumerate(piece[1])])
            piece[1] = bf_idx

        data = [tuple(l) for l in pieces]

        return piece_data_to_datasets(data, bf_idx_map, model_specs)
