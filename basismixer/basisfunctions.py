#!/usr/bin/env python

import sys
import logging
import numpy as np
import partitura.utils
from scipy.interpolate import interp1d
import types

import partitura.score as score
from partitura.utils import ensure_notearray

LOGGER = logging.getLogger(__name__)


class InvalidBasisException(Exception):
    pass


def print_basis_functions():
    """Print a list of all basisfunction names defined in this module,
    with descriptions where available.

    """
    module = sys.modules[__name__]
    doc_indent = 4
    for name in list_basis_functions():
        print('* {}'.format(name))
        member = getattr(sys.modules[__name__], name)
        if member.__doc__:
            print(' ' * doc_indent + member.__doc__.replace('\n', ' ' * doc_indent + '\n'))


def list_basis_functions():
    """Return a list of all basisfunction names defined in this module.

    The basisfunction names listed here can be specified by name in
    the `make_basis` function. For example:

    >>> basis, names = make_basis(part, ['metrical_basis', 'articulation_basis'])

    Returns
    -------
    list
        A list of strings

    """
    module = sys.modules[__name__]
    bfs = []
    exclude = {'make_basis'}
    for name in dir(module):
        if name in exclude:
            continue
        member = getattr(sys.modules[__name__], name)
        if isinstance(member, types.FunctionType) and name.endswith('_basis'):
            bfs.append(name)
    return bfs


def make_basis(part, basis_functions):
    """Compute the specified basis functions for a part.

    The function returns the computed basis functions as a N x M
    array, where N equals `len(part.notes_tied)` and M equals the
    total number of descriptors of all basis functions that occur in
    part.

    Furthermore the function returns the names of the basis functions.
    A list of strings of size M. The names have the name of the
    function prepended to the name of the descriptor. For example if a
    function named `abc_basis` returns descriptors `a`, `b`, and `c`,
    then the list of names returned by `make_basis(part,
    ['abc_basis'])` will be ['abc_basis.a', 'abc_basis.b',
    'abc_basis.c'].

    Parameters
    ----------
    part : Part
        The score as a Part instance
    basis_functions : list
        A list of basis functions. Elements of the list can be either
        the functions themselves or the names of a basis function as
        strings (or a mix). The basis functions specified by name are
        looked up in the `basismixer.basisfunctions` module.

    Returns
    -------
    basis : ndarray
        The basis functions
    names : list
        The basis names
    
    """
    
    acc = []

    for bf in basis_functions:

        if isinstance(bf, str):
            # get function by name from module
            func = getattr(sys.modules[__name__], bf)
        elif isinstance(bf, types.FunctionType):
            func = bf
        else:
            LOGGER.warning('Ignoring unknown basis function {}'.format(bf))

        bf, bn = func(part)

        # check if the size and number of the basis function are correct
        if bf.shape[1] != len(bn):
            msg = ('number of basis names {} does not equal '
                   'number of basis {}'.format(len(bn), bf.shape[1]))
            raise InvalidBasisException(msg)
        n_notes = len(part.notes_tied)
        if len(bf) != n_notes:
            msg = ('length of basis {} does not equal '
                   'number of notes {}'.format(len(bf), n_notes))
            raise InvalidBasisException(msg)

        if np.any(np.logical_or(np.isnan(bf), np.isinf(bf))):
            problematic = np.unique(np.where(np.logical_or(np.isnan(bf), np.isinf(bf)))[1])
            msg = ('NaNs or Infs found in the following basis: {} '
                   .format(', '.join(np.array(bn)[problematic])))
            raise InvalidBasisException(msg)
        
        # prefix basis names by function name
        bn = ['{}.{}'.format(func.__name__, n) for n in bn]

        acc.append((bf, bn))

    _data, _names = zip(*acc)
    basis_data = np.column_stack(_data)
    basis_names = [n for ns in _names for n in ns]
    return basis_data, basis_names


def polynomial_pitch_basis(part):
    """Polynomial pitch basis.

    Returns:
    * pitch : the midi pitch of the note
    * pitch^2 : the square of the midi pitch
    * pitch^3 : the power of 3 of the midi pitch
    
    """

    basis_names = ['pitch', 'pitch^2', 'pitch^3']
    max_pitch = 127
    pitches = ensure_notearray(part)["pitch"].astype(np.float)
    W = np.column_stack((pitches / max_pitch,
                         pitches**2 / max_pitch**2,
                         pitches**3 / max_pitch**3))

    return W, basis_names


def duration_basis(part):
    """Duration basis.

    Returns:
    * duration : the duration of the note

    """

    basis_names = ['duration']

    nd = np.array([(n.start.t, n.end_tied.t) for n in part.notes_tied])
    bm = part.beat_map

    durations_beat = bm(nd[:, 1]) - bm(nd[:, 0])
    W = durations_beat
    W.shape = (-1, 1)
    return W, basis_names

def onset_basis(part):
    """Onset basis
    
    Returns:
    * onset : the onset of the note in beats
    * score_position : position of the note in the score between 0 (the beginning of the piece) and 1 (the end of the piece)
    
    TODO:
    * rel_position_repetition
    """
    basis_names = ['onset', 'score_position']

    onsets_beat = ensure_notearray(part)["onset_beat"]
    rel_position = normalize(onsets_beat, method='minmax')

    W = np.column_stack((onsets_beat, rel_position))

    return W, basis_names

def relative_score_position_basis(part):
    W, names = onset_basis(part)
    return W[:, 1:], names[1:]


def grace_basis(part):
    """Grace basis.

    Returns:
    * grace_note : 1 when the note is a grace note, 0 otherwise
    * n_grace : the length of the grace note sequence to which 
                this note belongs (0 for non-grace notes)
    * grace_pos : the (1-based) position of the grace note in 
                  the sequence (0 for non-grace notes)

    """

    basis_names = ['grace_note', 'n_grace', 'grace_pos']

    notes = ensure_notearray(part)
    W = np.zeros((len(notes), 3))
    for i, n in enumerate(notes):
        if n["duration_beat"] == 0:
            n_grace = np.where((notes["onset_beat"] == n["onset_beat"]) & (notes["duration_beat"]==0))[0].shape[0]
            # or
            # n_grace = np.nonzero(ensure_notearray(part, include_grace_notes=True)["is_grace"])
            W[i, 0] = 1
            W[i, 1] = n_grace
            # W[i, 2] = n_grace - sum(1 for _ in n.iter_grace_seq()) + 1
            W[i, 2] = n_grace - sum(1 for _ in range(n_grace)) + 1
    return W, basis_names


def loudness_direction_basis(part):
    """The loudness directions in part.

    This function returns a varying number of descriptors, depending
    on which directions are present. Some directions are grouped
    together. For example 'decrescendo' and 'diminuendo' are encoded
    together in a descriptor 'loudness_decr'. The descriptor names of
    textual directions such as 'adagio' are the verbatim directions.
    
    Some possible descriptors:
    * p : piano
    * f : forte
    * pp : pianissimo
    * loudness_incr : crescendo direction
    * loudness_decr : decrescendo or diminuendo direction
    
    """

    onsets = ensure_notearray(part)["onset_div"]
    N = len(onsets)

    directions = list(part.iter_all(
        score.LoudnessDirection, include_subclasses=True))

    def to_name(d):
        if isinstance(d, score.ConstantLoudnessDirection):
            return d.text
        elif isinstance(d, score.ImpulsiveLoudnessDirection):
            return d.text
        elif isinstance(d, score.IncreasingLoudnessDirection):
            return 'loudness_incr'
        elif isinstance(d, score.DecreasingLoudnessDirection):
            return 'loudness_decr'

    basis_by_name = {}
    for d in directions:
        j, bf = basis_by_name.setdefault(to_name(d),
                                         (len(basis_by_name), np.zeros(N)))
        bf += basis_function_activation(d)(onsets)

    W = np.empty((len(onsets), len(basis_by_name)))
    names = [None] * len(basis_by_name)
    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


def tempo_direction_basis(part):
    """The tempo directions in part.

    This function returns a varying number of descriptors, depending
    on which directions are present. Some directions are grouped
    together. For example 'adagio' and 'molto adagio' are encoded
    together in a descriptor 'adagio'.
    
    Some possible descriptors:
    * adagio : directions like 'adagio', 'molto adagio'

    """
    onsets = ensure_notearray(part)["onset_div"]
    N = len(onsets)

    directions = list(part.iter_all(
        score.TempoDirection, include_subclasses=True))

    def to_name(d):
        if isinstance(d, score.ResetTempoDirection):
            ref = d.reference_tempo
            if ref:
                return ref.text
            else:
                return d.text
        elif isinstance(d, score.ConstantTempoDirection):
            return d.text
        elif isinstance(d, score.IncreasingTempoDirection):
            return 'tempo_incr'
        elif isinstance(d, score.DecreasingTempoDirection):
            return 'tempo_decr'

    basis_by_name = {}
    for d in directions:
        j, bf = basis_by_name.setdefault(to_name(d),
                                         (len(basis_by_name), np.zeros(N)))
        bf += basis_function_activation(d)(onsets)

    W = np.empty((len(onsets), len(basis_by_name)))
    names = [None] * len(basis_by_name)
    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names


def articulation_direction_basis(part):
    """
    """
    onsets = ensure_notearray(part)["onset_div"]
    N = len(onsets)

    directions = list(part.iter_all(
        score.ArticulationDirection, include_subclasses=True))

    def to_name(d):
        return d.text

    basis_by_name = {}

    for d in directions:

        j, bf = basis_by_name.setdefault(to_name(d),
                                         (len(basis_by_name), np.zeros(N)))
        bf += basis_function_activation(d)(onsets)

    W = np.empty((len(onsets), len(basis_by_name)))
    names = [None] * len(basis_by_name)

    for name, (j, bf) in basis_by_name.items():

        W[:, j] = bf
        names[j] = name

    return W, names


def basis_function_activation(direction):
    epsilon = 1e-6

    if isinstance(direction, (score.DynamicLoudnessDirection,
                              score.DynamicTempoDirection)):
        # a dynamic direction will be encoded as a ramp from d.start.t to
        # d.end.t, and then a step from d.end.t to the start of the next
        # constant direction.

        # There are two potential issues:

        # Issue 1. d.end is None (e.g. just a ritardando without dashes). In this case
        if direction.end:
            direction_end = direction.end.t
        else:
            # assume the end of d is the end of the measure:
            measure = next(direction.start.iter_prev(score.Measure, eq=True), None)
            if measure:
                direction_end = measure.start.t
            else:
                # no measure, unlikely, but not impossible.
                direction_end = direction.start.t

        if isinstance(direction, score.TempoDirection):
            next_dir = next(direction.start.iter_next(
                score.ConstantTempoDirection), None)
        if isinstance(direction, score.ArticulationDirection):
            next_dir = next(direction.start.iter_next(
                score.ConstantArticulationDirection), None)
        else:
            next_dir = next(direction.start.iter_next(
                score.ConstantLoudnessDirection), None)

        if next_dir:
            # TODO: what do we do when next_dir is too far away?
            sustained_end = next_dir.start.t
        else:
            # Issue 2. there is no next constant direction. In that case the
            # basis function will be a ramp with a quarter note ramp
            sustained_end = direction_end + direction.start.quarter

        x = [direction.start.t,
             direction_end - epsilon,
             sustained_end - epsilon]
        y = [0, 1, 1]

    elif isinstance(direction, (score.ConstantLoudnessDirection,
                                score.ConstantArticulationDirection,
                                score.ConstantTempoDirection)):
        x = [direction.start.t - epsilon,
             direction.start.t,
             direction.end.t - epsilon,
             direction.end.t]
        y = [0, 1, 1, 0]

    else:  # impulsive
        x = [direction.start.t - epsilon,
             direction.start.t,
             direction.start.t + epsilon]
        y = [0, 1, 0]

    return interp1d(x, y, bounds_error=False, fill_value=0)


def slur_basis(part):
    """Slur basis.

    Returns:
    * slur_incr : a ramp function that increases from 0 
                  to 1 over the course of the slur
    * slur_decr : a ramp function that decreases from 1 
                  to 0 over the course of the slur

    """
    names = ['slur_incr', 'slur_decr']
    onsets = ensure_notearray(part)["onset_div"]
    slurs = part.iter_all(score.Slur)
    W = np.zeros((len(onsets), 2))

    for slur in slurs:
        if not slur.end:
            continue
        x = [slur.start.t, slur.end.t]
        y_inc = [0, 1]
        y_dec = [1, 0]
        W[:, 0] += interp1d(x, y_inc, bounds_error=False, fill_value=0)(onsets)
        W[:, 1] += interp1d(x, y_dec, bounds_error=False, fill_value=0)(onsets)

    return W, names


def articulation_basis(part):
    """Articulation basis.

    This basis returns articulation-related note annotations, such as accents, legato, and tenuto.

    Possible descriptors:
    * accent : 1 when the note has an annotated accent sign
    * legato : 1 when the note has an annotated legato sign
    * staccato : 1 when the note has an annotated staccato sign
    ...

    """
    names = ['accent', 'strong-accent', 'staccato', 'tenuto',
             'detached-legato', 'staccatissimo', 'spiccato',
             'scoop', 'plop', 'doit', 'falloff', 'breath-mark',
             'caesura', 'stress', 'unstress', 'soft-accent']
    basis_by_name = {}
    notes = part.notes_tied
    N = len(notes)
    for i, n in enumerate(notes):
        if n.articulations:
            for art in n.articulations:
                if art in names:
                    j, bf = basis_by_name.setdefault(
                        art,
                        (len(basis_by_name), np.zeros(N)))
                    bf[i] = 1

    M = len(basis_by_name)
    W = np.empty((N, M))
    names = [None] * M

    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names

# # for a subset of the articulations do e.g.
# def staccato_basis(part):
#     W, names = articulation_basis(part)
#     if 'staccato' in names:
#         i = names.index('staccato')
#         return W[:, i:i + 1], ['staccato']
#     else:
#         return np.empty(len(W)), []


def fermata_basis(part):
    """Fermata basis.

    Returns:
    * fermata : 1 when the note coincides with a fermata sign.

    """
    names = ['fermata']
    onsets = ensure_notearray(part)["onset_div"]
    W = np.zeros((len(onsets), 1))
    for ferm in part.iter_all(score.Fermata):
        W[onsets == ferm.start.t, 0] = 1
    return W, names


def metrical_basis(part):
    """Metrical basis

    This basis encodes the metrical position in the bar. For example
    the first beat in a 3/4 meter is encoded in a binary descriptor
    'metrical_3_4_0', the fifth beat in a 6/8 meter as
    'metrical_6_8_4', etc. Any positions that do not fall on a beat
    are encoded in a basis suffixed '_weak'. For example a note
    starting on the second 8th note in a bar of 4/4 meter will have a
    non-zero value in the 'metrical_4_4_weak' descriptor.
    
    """
    notes = part.notes_tied
    ts_map = part.time_signature_map
    bm = part.beat_map
    basis_by_name = {}
    eps = 10**-6

    for i, n in enumerate(notes):

        beats, beat_type = ts_map(n.start.t).astype(int)
        measure = next(n.start.iter_prev(score.Measure, eq=True), None)

        if measure:
            measure_start = measure.start.t
        else:
            measure_start = 0

        pos = bm(n.start.t) - bm(measure_start)

        if pos % 1 < eps:
            name = 'metrical_{}_{}_{}'.format(beats, beat_type, int(pos))
        else:
            name = 'metrical_{}_{}_weak'.format(beats, beat_type)

        j, bf = basis_by_name.setdefault(name,
                                         (len(basis_by_name), np.zeros(len(notes))))
        bf[i] = 1

    W = np.empty((len(notes), len(basis_by_name)))
    names = [None] * len(basis_by_name)
    for name, (j, bf) in basis_by_name.items():
        W[:, j] = bf
        names[j] = name

    return W, names

def metrical_strength_basis(part):
    """Metrical strength basis

    This basis encodes the beat phase (relative position of a note within
    the measure), as well as metrical strength of common time signatures.
    """
    notes = part.notes_tied
    ts_map = part.time_signature_map
    bm = part.beat_map

    names = ['beat_phase',
             'metrical_strength_downbeat',
             'metrical_strength_secondary',
             'metrical_strength_weak']
    
    W = np.zeros((len(notes), len(names)))
    for i, n in enumerate(notes):

        beats, beat_type = ts_map(n.start.t).astype(int)
        measure = next(n.start.iter_prev(score.Measure, eq=True), None)

        if beats == 4:
            # for 4/4
            sec_beat = 2
        elif beats == 6:
            # for 6/8
            sec_beat = 3
        elif beats == 12:
            # for 12/8
            sec_beat = 6
        else:
            sec_beat = None

        if measure:
            measure_start = measure.start.t
        else:
            measure_start = 0

        pos = bm(n.start.t) - bm(measure_start)

        m_pos = np.mod(pos, beats)

        W[i, 0] = m_pos / beats
        
        if m_pos == 0:
            W[i, 1] = 1
        elif m_pos == sec_beat:
            W[i, 2] = 1
        else:
            W[i, 3] = 1

    return W, names

def time_signature_basis(part):
    """TIme Signature basis
    This basis encodes the time signature of the note in two sets of one-hot vectors,
    a one hot encoding of number of beats and a one hot encoding of beat type
    """

    notes = ensure_notearray(part)
    ts_map = part.time_signature_map
    possible_beats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'other']
    possible_beat_types = [1, 2, 4, 8, 16, 'other']
    W_beats = np.zeros((len(notes), len(possible_beats)))
    W_types = np.zeros((len(notes), len(possible_beat_types)))

    names = (['time_signature_num_{0}'.format(b) for b in possible_beats] +
             ['time_signature_den_{0}'.format(b) for b in possible_beat_types])
    
    for i, n in enumerate(notes):
        beats, beat_type = ts_map(n["onset_div"]).astype(int)

        if beats in possible_beats:
            W_beats[i, beats - 1] = 1
        else:
            W_beats[i, -1] = 1

        if beat_type in possible_beat_types:
            W_types[i, possible_beat_types.index(beat_type)] = 1
        else:
            W_types[i, -1] = 1

    W = np.column_stack((W_beats, W_types))

    return W, names


def vertical_neighbor_basis(part):
    """Vertical neighbor basis.

    Describes various aspects of simultaneously starting notes.

    Returns:
    * n_total :
    * n_above :
    * n_below :
    * highest_pitch :
    * lowest_pitch :
    * pitch_range : 

    """
    # the list of descriptors
    names = ['n_total', 'n_above', 'n_below',
             'highest_pitch', 'lowest_pitch', 'pitch_range']
    # notes
    notes = ensure_notearray(part)

    W = np.empty((len(notes), len(names)))
    for i, n in enumerate(notes):
        neighbors = notes[np.where(notes["onset_beat"] == n["onset_beat"])]["pitch"]
        max_pitch = np.max(neighbors)
        min_pitch = np.min(neighbors)
        W[i, 0] = len(neighbors) - 1
        W[i, 1] = np.sum(neighbors > n["pitch"])
        W[i, 2] = np.sum(neighbors < n["pitch"])
        W[i, 3] = max_pitch
        W[i, 4] = min_pitch
        W[i, 5] = max_pitch - min_pitch
    return W, names


def normalize(data, method='minmax'):
    """
    Normalize data in one of several ways.

    The available normalization methods are:

    * minmax
      Rescale `data` to the range `[0, 1]` by subtracting the minimum
      and dividing by the range. If `data` is a 2d array, each column is
      rescaled to `[0, 1]`.

    * tanh
      Rescale `data` to the interval `(-1, 1)` using `tanh`. Note that
      if `data` is non-negative, the output interval will be `[0, 1)`.

    * tanh_unity
      Like "soft", but rather than rescaling strictly to the range (-1,
      1), following will hold:

      normalized = normalize(data, method="tanh_unity")
      np.where(data==1) == np.where(normalized==1)

      That is, the normalized data will equal one wherever the original data
      equals one. The target interval is `(-1/np.tanh(1), 1/np.tanh(1))`.

    Parameters
    ----------
    data: ndarray
        Data to be normalized
    method: {'minmax', 'tanh', 'tanh_unity'}, optional
        The normalization method. Defaults to 'minmax'.

    Returns
    -------
    ndarray
        Normalized copy of the data
    """

    """Normalize the data in `data`. There are several normalization
    
    """
    if method == 'minmax':
        vmin = np.min(data, 0)
        vmax = np.max(data, 0)

        if np.isclose(vmin, vmax):
            # Return all values as 0 or as 1?
            return np.zeros_like(data)
        else:
            return (data - vmin) / (vmax - vmin)
    elif method == 'tanh':
        return np.tanh(data)
    elif method == 'tanh_unity':
        return np.tanh(data) / np.tanh(1)


# from extra.utils.data_utils import smooth
# from extra.data_handling.sparse_feature_extraction import (
#     scorepart_to_notes,
#     notes_to_notecentered_pianoroll,
#     notes_to_pianoroll_note_slices,
# )
# from music_utils.key_id.key_identification import (
#     key_identification,
#     key_to_scaledegree,
#     fifths_to_key,
#     SCALE_DEGREES,
#     KEYS)

# # from extra.data_handling.annotation_tokenizer import tokenizer, TokenizeException

# class NoteCenteredPianoRollBasis(Basis):
#     # lowest_pitch = 21
#     # highest_pitch = 108
#     neighbour_pitches = 36
#     neighbour_beats = 8
#     beat_div = 8
#     names = ['{0}'.format(i) for i in
#              range((2 * neighbour_pitches + 1) * (2 * neighbour_beats * beat_div))]

#     @classmethod
#     def makeBasis(cls, score_part):
#         notes, idx = scorepart_to_notes(score_part)
#         W = notes_to_notecentered_pianoroll(
#             notes, onset_only=False,
#             neighbour_pitches=cls.neighbour_pitches,
#             neighbour_beats=cls.neighbour_beats,
#             beat_div=cls.beat_div)
#         # print('pitch span', r)
#         return FeatureBasis(W, cls.make_full_names())


# def scorepart_to_onsetwise_pianoroll(score_part, morphetic_pitch=False, return_ioi=False):
#     notes, _ = scorepart_to_notes(score_part, morphetic_pitch)
#     start, end = notes[0, 0], notes[-1, 0]
#     return notes_to_pianoroll_note_slices(notes, return_ioi=return_ioi)


# def zero_mean_pianoroll(X):
#     t, p = np.nonzero(X)
#     center = 64
#     Z = np.zeros_like(X)
#     for i, t_i in enumerate(unique_onset_idx(t)):
#         avg_pitch = int(np.round(np.mean(p[t_i])))
#         new_pitches = p[t_i] - avg_pitch + center
#         try:
#             Z[t[t_i], new_pitches] = 1
#         except IndexError:
#             new_pitches[new_pitches < 0 ] = 0
#             new_pitches[new_pitches >= Z.shape[1] ] = Z.shape[1] - 1
#             Z[t[t_i], new_pitches] = 1
#     return Z

# class SelfSimilarityBasis(Basis):
#     _filter_sizes_abs = (5, 10, 20, 50, 100)
#     _max_prop = .7
#     _filter_sizes_prop = (.005, .01, .05, .1, .2, .3, .5)
#     names = (['abs_{}'.format(x) for x in _filter_sizes_abs] +
#              ['prop_{}'.format(x) for x in _filter_sizes_prop] +
#              ['centered_abs_{}'.format(x) for x in _filter_sizes_abs] +
#              ['centered_prop_{}'.format(x) for x in _filter_sizes_prop])

#     @classmethod
#     def makeBasis(cls, score_part):
#         pr = scorepart_to_onsetwise_pianoroll(score_part, morphetic_pitch=True).toarray()
#         onsets = np.array([n.start.t for n in score_part.notes])
#         uox = unique_onset_idx(onsets)

#         N = len(onsets)
#         # N x 128
#         X_n = pr.T[np.array([x[0] for x in uox])]
#         pr = None
#         X = np.corrcoef(X_n)
#         X[np.isnan(X)] = 0

#         names = []
#         W = np.empty((N, 0))

#         W_abs, k_abs = self_sim_conv_absolute(X, K=cls._filter_sizes_abs, max_prop=cls._max_prop)
#         if len(k_abs) > 0:
#             names.extend(['abs_{}'.format(x) for x in k_abs])
#             W = np.column_stack((W, expand_array(W_abs, uox, N)))

#         W_prop, k_prop = self_sim_conv_proportional(X, K=cls._filter_sizes_prop)
#         if len(k_prop) > 0:
#             names.extend(['prop_{}'.format(x) for x in k_prop])
#             W = np.column_stack((W, expand_array(W_prop, uox, N)))

#         X_n = zero_mean_pianoroll(X_n)

#         X = np.corrcoef(X_n)
#         X[np.isnan(X)] = 0

#         W_abs, k_abs = self_sim_conv_absolute(X, K=cls._filter_sizes_abs, max_prop=cls._max_prop)
#         if len(k_abs) > 0:
#             names.extend(['centered_abs_{}'.format(x) for x in k_abs])
#             W = np.column_stack((W, expand_array(W_abs, uox, N)))

#         W_prop, k_prop = self_sim_conv_proportional(X, K=cls._filter_sizes_prop)
#         if len(k_prop) > 0:
#             names.extend(['centered_prop_{}'.format(x) for x in k_prop])
#             W = np.column_stack((W, expand_array(W_prop, uox, N)))

#         return FeatureBasis(normalize(W), cls.make_full_names(names))

# def expand_array(x, idx, N):
#     """
#     Given an array `x` and a list of grouped indices `idx`, return a new array `y`,
#     where the values of `x` are duplicated according to `idx`, such that:

#     y[idx[i]] = x[i], where idx[i] is an array of integers

#     This function is a convenience function to duplicate onsetwise features (`x`) to
#     obtain notewise features (`y`).

#     Argument `N` is the length of the output array.

#     Warning: there are no checks that `N` is consistent with `idx`, and that the
#     values in `idx` fill all of `y`.

#     For example: let x = [1, 2, 3] and idx = [[0, 1], [2], [3, 4]], (and N = 5,
#     redundantly), then y = [1, 1, 2, 3, 3]

#     Parameters
#     ----------
#     x: ndarray
#         Array with values (can be multidimensional)
#     idx: list
#         List of index-arrays
#     N: int
#         Size of the expanded array

#     Returns
#     -------
#     ndarray
#         Expanded array
#     """


#     s = tuple([N] + list(x.shape)[1:])
#     y = np.empty(s)
#     for v, i in zip(x, idx):
#         y[i] = v
#     return y

# class PianorollBasis(Basis):
#     names = ['{0}'.format(i) for i in
#              range(128)] + ['log2_duration']

#     @classmethod
#     def makeBasis(cls, score_part):
#         W, ioi = scorepart_to_onsetwise_pianoroll(score_part, return_ioi=True)
#         W = W.T.toarray()
#         # print(W.shape, ioi.shape)
#         # print(np.unique(np.sort(ioi)))
#         assert np.sum(np.sum(W, 1) > 0) == W.shape[0]
#         W = np.column_stack((W, np.log2(ioi)))
#         return FeatureBasis(soft_normalize(W), cls.make_full_names())

# class ExtremePitchBasis(Basis):

#     """
#     This basis computes the highest and lowest pitch at each score position
#     Each row in the resulting matrix corresponds to a note in the score and
#     contains the highest and lowest pitch of the score position to which
#     the note belongs (i.e. for the same extreme pitches will appear for all
#     notes that belong to the same score position.

#     highestpitch : highest pitch of each score position
#     lowestpitch : lowest pitch of each score position
#     """
#     names = ['highestpitch', 'lowestpitch']

#     @classmethod
#     def makeBasis(cls, scorePart):

#         Q = 127.0
#         # Pitches and onsets
#         p_o = np.array([(n.midi_pitch, n.start.t) for n in scorePart.notes])

#         unique_onsets = np.unique(p_o[:, 1])

#         unique_onset_idxs = [np.where(p_o[:, 1] == u)[0] for u in unique_onsets]

#         pitches = [p_o[ix, 0] for ix in unique_onset_idxs]

#         W = np.zeros((len(p_o), 2))

#         for u, p in zip(unique_onset_idxs, pitches):
#             W[u, 0] = p.max() / Q
#             W[u, 1] = p.min() / Q

#         return FeatureBasis(W, cls.make_full_names())


# class VerticalIntervalClassBasis(Basis):
#     """
#     Three features describing up to three vertical interval classes
#     above the bass, i.e. the intervals between the notes of a chord and
#     the lowest pitch excluding pitch class repetition and octaves

#     vertical_intervals_{1,2,3}
#     """

#     names = ['vertical_interval_class_1',
#              'vertical_interval_class_2',
#              'vertical_interval_class_3']

#     @classmethod
#     def makeBasis(cls, scorePart):

#         Q = 11.0
#         # Pitches and onsets
#         p_o = np.array([(n.midi_pitch, n.start.t) for n in scorePart.notes])

#         # get unique onsets
#         unique_onsets = np.unique(p_o[:, 1])

#         # get unique_onset_idxs
#         unique_onset_idxs = [np.where(p_o[:, 1] == u)[0] for u in unique_onsets]

#         pitches = [p_o[ix, 0] for ix in unique_onset_idxs]

#         W = np.zeros((len(p_o), 3))

#         for u, p in zip(unique_onset_idxs, pitches):
#             # Vertical interval class combination
#             pitch_classes = np.unique(np.mod(p, 12))
#             vertical_intervals = pitch_classes - pitch_classes.min()
#             vertical_intervals.sort()

#             # Normalize the vintc to lie between 0 and 1
#             W[u, :len(vertical_intervals[slice(1, 4)])] = (
#                 vertical_intervals[slice(1, 4)]) / Q

#         return FeatureBasis(W, cls.make_full_names())


# class VerticalNeighborBasis(Basis):

#     """
#     This basis has three members:

#     lower_neighbors: the number of simultaneously starting notes with lower pitches
#     upper_neighbors: the number of simultaneously starting notes with higher pitches

#     """

#     names = ['lower_neighbors', 'upper_neighbors', 'total_neighbors']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         t_dict = partition(lambda n: n.start.t, scorePart.notes)
#         n_dict = {}
#         for k, v in t_dict.items():
#             v.sort(key=attrgetter('midi_pitch'))
#             N = len(v) - 1
#             for i, n in enumerate(v):
#                 n_dict[n] = (i, N - i, N + 1)
#         W = np.array([n_dict[n] for n in scorePart.notes])
#         return FeatureBasis(soft_normalize(W), cls.make_full_names())


# class UIOIBasis(Basis):
#     """
#     This basis has two members:

#     ioi_prev: the time interval between the current onset (t) with the previous onset (t-1)
#     ioi_next: the time interval between (t-2) and (t-3)


#     """
#     names = ['u_ioi_prev', 'u_ioi_next']
#     @classmethod
#     def makeBasis(cls, scorePart):
#         onsets = np.array([n.start.t for n in scorePart.notes])
#         u_onset_idx = unique_onset_idx(onsets)
#         u_onsets = np.array([onsets[ii[0]] for ii in u_onset_idx])
#         # include offset of last note for computing last 'ioi'
#         u_ioi = np.diff(np.r_[u_onsets, scorePart.notes[-1].end.t])
#         u_W = np.column_stack((np.r_[0, u_ioi[:-1]], u_ioi)).astype(np.float)
#         W = np.empty((len(onsets), 2))
#         for i, ii in enumerate(u_onset_idx):
#             W[ii,:] = u_W[i,:]
#         return FeatureBasis(soft_normalize(W, preserve_unity=True), cls.make_full_names())

# class IOIBasis(Basis):

#     """
#     This basis has three members:

#     ioi_prev1: the time interval between the current onset (t) with the previous onset (t-1)
#     ioi_prev2: the time interval between (t-1) and (t-2)
#     ioi_prev3: the time interval between (t-2) and (t-3)

#     Each of these values is 0 in case there are no prior onsets

#     In this basis, the next onset is defined as the next onset that is

#     """

#     names = ['ioi_prev1', 'ioi_prev2', 'ioi_prev3',
#              'ioi_next1', 'ioi_next2', 'ioi_next3']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         t_dict = {}
#         for note in scorePart.notes:
#             pred1 = note.start.get_prev_of_type(Note)
#             if len(pred1) > 1:
#                 d1 = note.start.t - pred1[0].start.t
#                 pred2 = pred1[0].start.get_prev_of_type(Note)
#                 if len(pred2) > 1:
#                     d2 = pred1[0].start.t - pred2[0].start.t
#                     pred3 = pred2[0].start.get_prev_of_type(Note)
#                     if len(pred3) > 1:
#                         d3 = pred2[0].start.t - pred3[0].start.t
#                     else:
#                         d3 = 0
#                 else:
#                     d2 = 0
#                     d3 = 0
#             else:
#                 d1 = 0
#                 d2 = 0
#                 d3 = 0

#             succ1 = note.start.get_next_of_type(Note)
#             if len(succ1) > 1:
#                 d4 = succ1[0].start.t - note.start.t
#                 succ2 = succ1[0].start.get_next_of_type(Note)
#                 if len(succ2) > 1:
#                     d5 = succ2[0].start.t - succ1[0].start.t
#                     succ3 = succ2[0].start.get_next_of_type(Note)
#                     if len(succ3) > 1:
#                         d6 = succ3[0].start.t - succ2[0].start.t
#                     else:
#                         d6 = 0
#                 else:
#                     d5 = 0
#                     d6 = 0
#             else:
#                 d4 = 0
#                 d5 = 0
#                 d6 = 0

#             t_dict[note.start] = (d1, d2, d3, d4, d5, d6)
#         W = np.array([t_dict[n.start] for n in scorePart.notes])
#         return FeatureBasis(soft_normalize(W, preserve_unity=True), cls.make_full_names())

#     # @classmethod
#     # def makeBasis(cls, scorePart):
#     #     t_dict = {}
#     #     for tp in scorePart.timeline.points:
#     #         if tp.prev:
#     #             d1 = tp.t - tp.prev.t
#     #             if tp.prev.prev:
#     #                 d2 = tp.prev.t - tp.prev.prev.t
#     #                 if tp.prev.prev.prev:
#     #                     d3 = tp.prev.prev.t - tp.prev.prev.prev.t
#     #                 else:
#     #                     d3 = 0
#     #             else:
#     #                 d2 = 0
#     #         else:
#     #             d1 = 0
#     #             d2 = 0
#     #             d3 = 0

#     #         if tp.next:
#     #             d4 = tp.next.t - tp.t
#     #             if tp.next.next:
#     #                 d5 = tp.next.next.t - tp.next.t
#     #                 if tp.next.next.next:
#     #                     d6 = tp.next.next.next.t - tp.next.next.t
#     #                 else:
#     #                     d6 = 0
#     #             else:
#     #                 d5 = 0
#     #         else:
#     #             d4 = 0
#     #             d5 = 0
#     #             d6 = 0

#     #         t_dict[tp] = (d1, d2, d3, d4, d5, d6)
#     #     W = np.array([t_dict[n.start] for n in scorePart.notes])
#     #     return FeatureBasis(normalize(W), cls.make_full_names())


# class RitardandoBasis(Basis):
#     names = ['ritardando']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         end = scorePart.timeline.points[0].t
#         start = scorePart.timeline.points[-1].t
#         W = np.array([n.start.t for n in scorePart.notes], dtype=np.float)
#         W = np.exp(((W - start) / (end - start))**100) - 1
#         W.shape = (-1, 1)
#         return FeatureBasis(soft_normalize(W, preserve_unity=True), cls.make_full_names())


# class SlurBasis(Basis):
#     names = ['slur_step', 'slur_incr', 'slur_decr']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         slurs = scorePart.timeline.get_all_of_type(Slur)

#         W = np.zeros((len(scorePart.notes), 3), dtype=np.float32)

#         if len(slurs) > 0:
#             ss = np.array([(s.voice, s.start.t, s.end.t)
#                            for s in slurs
#                            if (s.start is not None and
#                                s.end is not None)])

#             if ss.shape[0] < len(slurs):
#                 LOGGER.info("Ignoring {0} of {1} slurs for missing start or end"
#                             .format(len(slurs) - ss.shape[0], len(slurs)))

#             # begin make arch
#             onsets = np.array([n.start.t for n in scorePart.notes])
#             first = np.min(onsets)
#             last = np.max(onsets)
#             eps = 10**-4

#             for v, start, end in ss:
#                 tmap = np.array([[min(first, start - eps),  0, 0],
#                                  [start - eps,              0, 0],
#                                  [start,                    0, 1],
#                                  [end,                      1, 0],
#                                  [end + eps,                0, 0],
#                                  [max(last, end + eps),     0, 0]])
#                 incr = interp1d(tmap[:, 0], tmap[:, 1])
#                 decr = interp1d(tmap[:, 0], tmap[:, 2])
#                 W[:, 1] += incr(onsets)
#                 W[:, 2] += decr(onsets)

#             start_idx = np.argsort(ss[:, 1])
#             end_idx = np.argsort(ss[:, 2])

#             ss_start = ss[start_idx,:]
#             ss_end = ss[end_idx,:]

#             idx = np.arange(ss.shape[0], dtype=np.int)

#             idx_start = idx[start_idx]
#             idx_end = idx[end_idx]

#             ndnv = np.array([(n.start.t, n.voice) for n in scorePart.notes])

#             start_before = np.searchsorted(
#                 ss_start[:, 1], ndnv[:, 0], side='right')
#             end_after = np.searchsorted(ss_end[:, 2], ndnv[:, 0], side='left')

#             for i in range(ndnv.shape[0]):
#                 spanning = tuple(
#                     set(idx_start[:start_before[i]]).intersection(set(idx_end[end_after[i]:])))
#                 W[i, 0] = 1 if ndnv[i, 1] in ss[spanning, 0] else 0

#         return FeatureBasis(W, cls.make_full_names())


# class ScoreTimeBasis(Basis):
#     names = ['beat']
#     @classmethod
#     def makeBasis(cls, scorePart):
#         nd = np.array([n.start.t for n in scorePart.notes])
#         bm = scorePart.beat_map
#         W = bm(nd)
#         W.shape = (-1, 1)
#         return FeatureBasis(W, cls.make_full_names())

# class RestBasis(Basis):
#     names = ['precedes_rest', 'precedes_rest_narrow', 'precedes_rest_mid', 'precedes_rest_wide']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         smooth_k = 2
#         smooth_k_mid = 6
#         smooth_k_wide = 10

#         t_rest = dict((n.start.t, 1 if len(n.end.get_starting_objects_of_type(Note)) == 0 else 0)
#                       for n in scorePart.notes)

#         t_sorted = sorted(t_rest.keys())

#         smoothed = np.column_stack(([t_rest[k] for k in t_sorted],
#                                     smooth([t_rest[k] for k in t_sorted], smooth_k),
#                                     smooth([t_rest[k] for k in t_sorted], smooth_k_mid),
#                                     smooth([t_rest[k] for k in t_sorted], smooth_k_wide)))
#         rest_smooth = dict((k, x) for k, x in zip(t_sorted, smoothed))

#         W = np.array([rest_smooth[n.start.t] for n in scorePart.notes])
#         return FeatureBasis(normalize(W), cls.make_full_names())


# class RepeatBasis(Basis):
#     names = ['repeat_end', 'repeat_end_short_ramp', 'repeat_end_med_ramp', 'repeat_end_wide_ramp']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         smooth_k = 2
#         smooth_k_mid = 6
#         smooth_k_wide = 10

#         on_repeat = dict((tp.t, 0 if len(tp.get_ending_objects_of_type(Repeat)) == 0 else 1)
#                          for tp in scorePart.timeline.points)
#         on_repeat[scorePart.timeline.points[-1].t] = 1
#         t_sorted = sorted(on_repeat.keys())

#         smoothed = np.column_stack((
#             [on_repeat[k] for k in t_sorted],
#             smooth([on_repeat[k] for k in t_sorted], smooth_k),
#             smooth([on_repeat[k] for k in t_sorted], smooth_k_mid),
#             smooth([on_repeat[k] for k in t_sorted], smooth_k_wide)))

#         repeat_smooth = dict((k, x) for k, x in zip(t_sorted, smoothed))

#         W = np.array([repeat_smooth[n.end.t] for n in scorePart.notes])
#         return FeatureBasis(normalize(W), cls.make_full_names())


# class HarmonicBasis(Basis):
#     names = KEYS + SCALE_DEGREES

#     @classmethod
#     def makeBasis(cls, scorePart):
#         tl = scorePart.timeline
#         key_sigs = tl.get_all_of_type(KeySignature)
#         measures = tl.get_all_of_type(Measure)

#         note_info = np.array([(n.midi_pitch, n.start.t, n.end.t)
#                               for n in scorePart.notes])

#         bar_onsets = np.array([m.start.t for m in measures])

#         key_info = [(ks.fifths, ks.mode, ks.start.t)
#                     for ks in key_sigs]

#         idx = np.searchsorted(note_info[:, 1], bar_onsets)
#         idx_key = np.searchsorted(note_info[:, 1], [x[2] for x in key_info])

#         key_segments = []
#         for key_notes in np.split(note_info, idx_key)[1:]:
#             key_segments.append(key_notes)

#         segments = []
#         for bar_notes in np.split(note_info, idx)[1:]:
#             if len(bar_notes) > 0:
#                 segments.append(bar_notes)

#         key_gt = []
#         for ks, seg in zip(key_info, key_segments):
#             key_gt.append((fifths_to_key(ks[0], ks[1], seg), ks[2]))

#         # for segment in segments:
#             # print segment
#         viterbi_path = key_identification(segments, key_gt, 'temperley')
#         # print viterbi path
#         key_seq = []
#         scale_degree_sect = []

#         for ky, segment in zip(viterbi_path, segments):
#             # print ky,segment
#             for kych in key_gt:

#                 try:
#                     if segment[0, 1] >= kych[1]:
#                         kyc = kych[0]

#                 except:
#                     pass

#             scale_degree_sect += [key_to_scaledegree(ky, kyc)] * len(segment)
#             key_seq += [ky] * len(segment)

#         W_key = np.zeros((len(note_info[:, 0]), 24))
#         W_sd = np.zeros((len(note_info[:, 0]), len(SCALE_DEGREES)))
#         for ii, ky in enumerate(zip(key_seq, scale_degree_sect)):
#             W_key[ii, KEYS.index(ky[0])] = 1
#             W_sd[ii, SCALE_DEGREES.index(ky[1])] = 1

#         W = np.hstack((W_key, W_sd))

#         return FeatureBasis(W, cls.make_full_names())

# class PredictivePianorollBasis(Basis):

#     # If this class attribute is defined, it will be set by
#     # `lbm.utils.basisUtilities.set_derived_data_folder`, which should be called
#     # before the makeBasis method is called
#     data_folder = None

#     @classmethod
#     def makeBasis(cls, scorePart):
#         onsets = np.array([n.start.t for n in scorePart.notes])
#         # Initialize matrix of basis functions
#         uox = unique_onset_idx(onsets)
#         N = len(uox)
#         W = None
#         names = None
#         if cls.data_folder is not None:
#             fn = os.path.join(cls.data_folder,
#                               '{}_hidden.npy'.format(scorePart.piece_name))
#             try:
#                 W_onset = np.load(fn)
#                 # print(W_onset.shape, fn, len(uox))
#                 W = expand_array(W_onset, uox, len(onsets))
#                 if len(W) != len(onsets):
#                     LOGGER.warning(('Data shape from {} does not coincide with '
#                                     'the number of onsets in the score: {} vs {}')
#                                    .format(fn, W.shape, len(onsets)))
#                     W = None
#                 else:
#                     names = ['feature{0:04}'.format(i) for i in range(W.shape[1])]
#             except:
#                 LOGGER.warning('Could not load data from {}'.format(fn))


#         else:
#             LOGGER.warning('Cannot create PredictivePianorollBasis, because no derived data folder has been specified')

#         if W is None:
#             names = []
#             W = np.zeros((len(onsets), 0))

#         return FeatureBasis(soft_normalize(W), cls.make_full_names(names))

# class HarmonicTensionBasis(Basis):
#     names = ['key', 'diameter', 'centroid']

#     # this should be set from outside before makeBasis is called
#     data_folder = None

#     @classmethod
#     def makeBasis(cls, scorePart):

#         onsets = np.array([n.start.t for n in scorePart.notes])
#         bars = scorePart.timeline.get_all_of_type(Measure)

#         # compute how many outputs Dorien code generates for this piece:

#         # default value used in Dorien's code
#         nvis = 4
#         ndivs = (bars[-1].end.t - bars[0].start.t)
#         ws = (ndivs / len(bars) ) / nvis
#         # print('expected nr of nlines in files', ndivs / ws)

#         start = bars[0].start.t
#         end = bars[-1].end.t

#         # Initialize matrix of basis functions
#         W = np.zeros((len(onsets), len(cls.names)))

#         if cls.data_folder is not None:
#             # Load harmonic tension information from Doriens
#             # XmlTensionVisualiser.jar output files
#             for i, b_name in enumerate(cls.names):
#                 fn = os.path.join(cls.data_folder,
#                                   '{}_{}.data'.format(scorePart.piece_name, b_name))
#                 try:
#                     data = np.loadtxt(fn)[:, 1]
#                 except:
#                     LOGGER.warning('Could not load data from {}'.format(fn))
#                     continue

#                 data = np.r_[0, data, 0]
#                 times = np.arange(start, end, ws)

#                 if len(times) == len(data) - 2:
#                     times = np.r_[start, times + ws / 2., end]
#                 elif len(times) - 1 == len(data):
#                     times = np.r_[start, times[1:], end]
#                 else:
#                     LOGGER.info('HarmonicTensionBasis expected {} data points from {}, got {}'
#                                 .format(len(times), fn, len(data)))
#                     times = np.linspace(start, end, len(data))
#                 W[:, i] = interp1d(times, data)(onsets)
#         else:
#             LOGGER.warning('Cannot create HarmonicTensionBasis, because no derived data folder has been specified')

#         return FeatureBasis(soft_normalize(W), cls.make_full_names())
