#!/usr/bin/env python

import logging
import os

import numpy as np
import partitura.musicanalysis
from torch.utils.data import Dataset, ConcatDataset

from partitura import load_musicxml, load_match
from partitura.score import expand_grace_notes
from basismixer.basisfunctions import make_basis
from basismixer.utils import (pair_files,
                              get_unique_onset_idxs,
                              notewise_to_onsetwise)
from .data import piece_data_to_datasets
from basismixer.performance_codec import get_performance_codec
from .parse_tsv_aligment import load_alignment_from_ASAP
from partitura.performance import PerformedPart
from pathlib import Path
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

def make_datasets(model_specs, root_folder, quirks=False, gracenotes='remove'):
    all_targets = list(set([n for model_spec in model_specs
                            for n in model_spec['parameter_names']]))

    perf_codec = get_performance_codec(all_targets)

    bf_idx_map = {}

    data = []

    all_basis_functions = set([n for model_spec in model_specs
                               for n in model_spec['basis_functions']])

    pieces = list(Path(root_folder).rglob("*/xml_score.musicxml"))

    i = 0
    for piece in pieces:

        print(f"{i}/{len(pieces)}")
        i += 1

        name = "/".join(str(piece).split("/")[-3:-1])
        LOGGER.info('Processing {}'.format(piece))

        part = load_musicxml(piece)
        part = partitura.score.merge_parts(part)

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


        bf_idx = np.array([bf_idx_map.setdefault(name, len(bf_idx_map))
                           for i, name in enumerate(bf_names)])

        nid_dict = dict((n.id, i) for i, n in enumerate(part.notes_tied))

        performances = list(Path(piece).parent.glob("*_note_alignments/note_alignment.tsv"))

        for performance in performances:
            alignment = load_alignment_from_ASAP(performance)
            ppart = partitura.load_performance_midi(str(performance).split("_note_alignments/")[0] + ".mid")#todo: can I load this from midi without loss?

            if quirks:
                for n in alignment:
                    if n['label'] == 'match':
                        n['score_id'] = n['score_id'].split('-')[0]

            # compute the targets
            targets, snote_ids = perf_codec.encode(part, ppart, alignment)

            matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])
            basis_matched = basis[matched_subset_idxs]

            score_onsets = bm([n.start.t for n in part.notes_tied])[matched_subset_idxs]
            unique_onset_idxs = get_unique_onset_idxs(score_onsets)

            data.append((basis_matched, bf_idx, targets, unique_onset_idxs, name))

    return piece_data_to_datasets(data, bf_idx_map, model_specs)


def performed_part_from_alignment(alignment, pedal_threshold=64, first_note_at_zero=False):#todo: delete?
    """Make PerformedPart from performance info in a MatchFile

    Parameters
    ----------
    mf : MatchFile
        A MatchFile instance
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : PerformedPart
        A performed part

    """
    from partitura.io import MatchFile
    mf = MatchFile()
    mf.lines = np.array(alignment)

    # Get midi time units
    mpq = mf.info("midiClockRate")  # 500000 -> microseconds per quarter
    ppq = mf.info("midiClockUnits")  # 500 -> parts per quarter

    # PerformedNote instances for all MatchNotes
    notes = []

    first_note = next(mf.iter_notes(), None)
    if first_note and first_note_at_zero:
        offset = first_note.Onset * mpq / (10 ** 6 * ppq)
    else:
        offset = 0

    for note in mf.iter_notes():

        sound_off = note.Offset if note.AdjOffset is None else note.AdjOffset

        notes.append(
            dict(
                id=note.Number,
                midi_pitch=note.MidiPitch,
                note_on=note.Onset * mpq / (10 ** 6 * ppq) - offset,
                note_off=note.Offset * mpq / (10 ** 6 * ppq) - offset,
                sound_off=sound_off * mpq / (10 ** 6 * ppq) - offset,
                velocity=note.Velocity,
            )
        )

    # SustainPedal instances for sustain pedal lines
    sustain_pedal = []
    for ped in mf.sustain_pedal:
        sustain_pedal.append(
            dict(
                number=64,  # type='sustain_pedal',
                time=ped.Time * mpq / (10 ** 6 * ppq),
                value=ped.Value,
            )
        )

    # Make performed part
    ppart = PerformedPart(
        id="P1",
        part_name=mf.info("piece"),
        notes=notes,
        controls=sustain_pedal,
        sustain_pedal_threshold=pedal_threshold,
    )
    return ppart