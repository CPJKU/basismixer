#!/usr/bin/env python

import sys
from collections import defaultdict
import numpy as np

def make_basis(part, names):
    acc = []
    cs = 0
    for name in names:
        # get function by name from module
        func = getattr(sys.modules[__name__], name)
        bf, bn = func(part)
        # TODO: check that size of bf matches part.notes_tied
        acc.append((bf, bn, cs))
        cs += len(bn)
        
    _data, _names, _offsets = zip(*acc)
    basis_data = np.column_stack(_data)
    basis_names = [n for ns in _names for n in ns]

    return basis_data, basis_names

# OBSOLETE
# def make_basis_old(part, basis_config):
#     # targets = basis_config.keys()
#     names = list(set(basis for basis_list in basis_config.values()
#                      for basis in basis_list))
#     acc = []
#     cs = 0
#     for name in names:
#         func = getattr(sys.modules[__name__], name)
#         bf, bn = func(part)
#         acc.append((bf, bn, cs))
#         cs += len(bn)
        
#     _data, _names, _offsets = zip(*acc)
#     basis_data = np.column_stack(_data)
#     basis_names = [n for ns in _names for n in ns]

#     target_basis_idx = defaultdict(list)
#     for target, t_names in basis_config.items():
#         for bf, bn, offset in zip(names, _names, _offsets):
#             if bf in t_names:
#                 target_basis_idx[target].extend(range(offset, offset+len(bn)))

#     return basis_data, basis_names, target_basis_idx # 


def odd_even_basis(part):
    N = len(part.notes)
    W = np.ones(N)
    if N % 2 == 0:
        basis_names = ['even']
    else:
        basis_names = ['odd']
    return W, basis_names

def polynomial_pitch_basis(part):

    basis_names = ['pitch', 'pitch^2', 'pitch^3']

    pitches = np.array([n.midi_pitch for n in part.notes]).astype(np.float)
    W = np.column_stack((pitches/127,
                         pitches**2/127**2,
                         pitches**3/127**3))
    
    return normalize(W), basis_names

def duration_basis(part):

    basis_names = ['duration']

    nd = np.array([(n.start.t, n.end.t) for n in part.notes])
    bm = part.beat_map

    durations_beat = bm(nd[:, 1]) - bm(nd[:, 0])
    W = durations_beat
    W.shape = (-1, 1)
    return normalize(W, 'tanh_unity'), basis_names



def normalize(data, method='minmax'):
    """
    Normalize the data in `data`. There are several normalization methods:

    * minmax

      Rescale `data` to the range `[0, 1]` by subtracting the minimum and dividing by
      the range. If `data` is a 2d array, each column is rescaled to `[0, 1]`.

    * tanh

      Rescale `data` to the interval `(-1, 1)` using `tanh`. Note that if `data`
      is non-negative, the output interval will be `[0, 1)`.

    * tanh_unity

      Like "soft", but rather than rescaling strictly to the range (-1, 1),
      following will hold:

          normalized = normalize(data, method="tanh_unity")
          np.where(data==1) == np.where(normalized==1)

      That is, that the target interval is `(-1/np.tanh(1), 1/np.tanh(1))`.
    """
    if method == 'minmax':
        vmin = np.min(data, 0)
        vmax = np.max(data, 0)
        return (data-vmin)/(vmax-vmin)
    elif method == 'tanh':
        return np.tanh(data)
    elif method == 'tanh_unity':
        return np.tanh(data)/np.tanh(1)


# import logging
# from operator import attrgetter
# from collections import defaultdict

# import numpy as np
# # from scipy import interpolate
# import scipy.stats
# from scipy.interpolate import interp1d

# from utils.basisUtilities import (normalize, soft_normalize,
#                                   FeatureBasis,
#                                   self_sim_conv_absolute,
#                                   self_sim_conv_proportional,
#                                   interpolate_feature)
# from lbm.extra.utils.container_utils import partition
# from lbm.timecodec import unique_onset_idx
# from extra.data_handling.scoreontology import (
#     Note, TimeSignature, Measure, Slur, Repeat,
#     KeySignature,
#     TempoDirection,
#     ConstantTempoDirection,
#     DynamicTempoDirection,
#     ResetTempoDirection,
#     LoudnessDirection,
#     DynamicLoudnessDirection,
#     ConstantLoudnessDirection,
#     ImpulsiveLoudnessDirection)


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

# # import IR_parser as ir
# LOGGER = logging.getLogger(__name__)


# class Basis(object):
#     names = []

#     @classmethod
#     def make_full_names(cls, names=None):
#         """
#         Return a list of basis function names prepended with the name of the Basis
#         class producing the functions. For Basis classes where the number of
#         basis fuctions is not known in advance (because they depend on the
#         information in the score), the list of basis function names to be
#         produced can be provided through `names`.

#         Parameters
#         ----------
#         names: list, optional
#             List of basis function names

#         Returns
#         -------
#         list
#             List of basis function names, with Basis class name prepended
#         """

#         if names is None:
#             names = cls.names
#         return ['.'.join((cls.__name__, name)) for name in names]


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


# # def makeLocalDirectionFeature(onsets, direction):
# #     epsilon = 1e-6
# #     if isinstance(direction, DynamicLoudnessDirection):
# #         d = np.array([(direction.start.t, 0),
# #                       (direction.end.t, 1),
# #                       (direction.end.t + epsilon, 0.0)])
# #     elif isinstance(direction, ConstantLoudnessDirection):
# #         end = onsets[-1]
# #         if direction.start.next != None:
# #             nextd = direction.start.get_next_of_type(ConstantLoudnessDirection)
# #             if nextd != None and len(nextd) > 0:
# #                 end = nextd[0].start.t

# #         d = np.array([(direction.start.t - epsilon, 0),
# #                       (direction.start.t, 1),
# #                       (end, 1),
# #                       (end + epsilon, 0)])
# #         d2 = np.array([(direction.start.t - 8 - epsilon, 0),
# #                        (direction.start.t, 1),
# #                        (direction.start.t + 2, 0),
# #                        ])
# #         return np.column_stack((interpolate_feature(onsets, d),
# # interpolate_feature(onsets, d1),
# #                                 interpolate_feature(onsets, d2)))
# #     else:
# #         d = np.array([(direction.start.t - epsilon, 0),
# #                       (direction.start.t, 1),
# #                       (direction.start.t + epsilon, 0)])
# #     return interpolate_feature(onsets, d)

# def make_local_loudness_direction_feature(onsets, direction):
#     # TODO: make sure that for dynamic, there is a step to follow the ramp
#     constant_type = ConstantLoudnessDirection
#     dynamic_type = DynamicLoudnessDirection

#     epsilon = 1e-6
#     last_time = onsets[-1]

#     if direction.end is None:
#         if isinstance(direction, dynamic_type):
#             nextd = direction.start.get_next_of_type(constant_type)
#             if len(nextd) > 0:
#                 direction_end_t = max(direction.start.t + epsilon,
#                                       nextd[0].start.t - epsilon)
#             else:
#                 direction_end_t = last_time
#         else:
#             direction_end_t = last_time
#     else:
#         direction_end_t = direction.end.t

#     if isinstance(direction, dynamic_type):
#         d = np.array([(direction.start.t, 0),
#                       (direction_end_t, 1),
#                       (direction_end_t + epsilon, 0.0)])
#     elif isinstance(direction, constant_type):
#         end = last_time
#         # if direction.start.next != None:
#         nextd = direction.start.get_next_of_type(constant_type)
#         if len(nextd) > 0:
#             end = nextd[0].start.t

#         d = np.array([(direction.start.t - epsilon, 0),
#                       (direction.start.t, 1),
#                       (end - epsilon, 1),
#                       (end, 0)])
#         return interpolate_feature(onsets, d)
#     else:
#         d = np.array([(direction.start.t - epsilon, 0),
#                       (direction.start.t, 1),
#                       (direction.start.t + epsilon, 0)])
#     return interpolate_feature(onsets, d)

# def make_local_tempo_direction_feature(onsets, direction):
#     constant_type = ConstantTempoDirection
#     dynamic_type = DynamicTempoDirection

#     epsilon = 1e-6

#     last_time = onsets[-1]
#     # assert last_time == np.max(onsets)
#     # if direction.end is None:
#     #     if isinstance(direction, dynamic_type):
#     #         nextd = direction.start.get_next_of_type(constant_type)
#     #         if len(nextd) > 0:
#     #             direction_end_t = max(direction.start.t + epsilon,
#     #                                   nextd[0].start.t - epsilon)
#     #         else:
#     #             direction_end_t = last_time + 1
#     # elif isinstance(direction, constant_type):
#     # nextd = direction.start.get_next_of_type(constant_type)
#     # if len(nextd) > 0 and nextd[0].text != u'a_tempo':
#     # direction_end_t = max(direction.start.t + epsilon,
#     # nextd[0].start.t - epsilon)
#     # else:
#     # direction_end_t = last_time + 1
#     # else:
#     # direction_end_t = last_time + 1
#     # else:
#     #     direction_end_t = direction.end.t

#     if isinstance(direction, dynamic_type):
#         if direction.end is None:
#             end = last_time + 1
#             nextd = direction.start.get_next_of_type(constant_type)
#             if len(nextd) > 0:
#                 end = nextd[0].start.t
#         else:
#             end = direction.end.t

#         d = np.array([(direction.start.t, 0),
#                       (end - epsilon, 1),
#                       (end, 0.0)])

#     elif isinstance(direction, constant_type):
#         end = last_time + 1
#         nextd = direction.start.get_next_of_type(constant_type)
#         if len(nextd) > 0 and nextd[0].text not in (u'a_tempo', u'in_tempo'):
#             end = nextd[0].start.t

#         d = np.array([(direction.start.t - epsilon, 0),
#                       (direction.start.t, 1),
#                       (end - epsilon, 1),
#                       (end, 0)])
#         return interpolate_feature(onsets, d)
#     else:
#         d = np.array([(direction.start.t - epsilon, 0),
#                       (direction.start.t, 1),
#                       (direction.start.t + epsilon, 0)])
#     return interpolate_feature(onsets, d)


# class TempoAnnotationBasis(Basis):
#     @classmethod
#     def makeBasis(cls, scorePart):
#         return cls._makeBasis(scorePart, indicator=False)

#     @classmethod
#     def _makeBasis(cls, scorePart, indicator):
#         """
#         Make basis functions with temporary effects, one basis
#         function for each type of mark
#         """
#         epsilon = 1e-4
#         onsets = np.array([n.start.t for n in scorePart.notes])
#         directions = [x for x in scorePart.timeline.get_all_of_type(TempoDirection, include_subclasses=True)
#                       if not isinstance(x, ResetTempoDirection) or x.text != 'a_tempo']
#         # d_directions = scorePart.timeline.get_all_of_type(DynamicTempoDirection)
#         # for d in directions:
#         #     print(d)
#         #     print(d.start.t, d.end.t if d.end else None)

#         # directions = scorePart.get_tempo_directions()
#         first_tempo = next((d for d in directions if
#                             isinstance(d, ConstantTempoDirection)),
#                            None)

#         direction_names = [x.text for x in directions]
#         dir_by_end_times = dict((x.end and x.end.t, x) for x in directions)
#         # print(direction_names)
#         # for n in direction_names:
#         #     try:
#         #         print(tokenizer.tokenize(n)[0].value)
#         #     except TokenizeException:
#         #         pass

#         tempo_1 = 'tempo_1'
#         tempo_1_referent = tempo_1

#         for d in directions:
#             if d.text == tempo_1:
#                 if first_tempo and first_tempo.text != tempo_1:
#                     print('setting {} at {} to {}'.format(d.text, d.start.t, first_tempo.text))
#                     d.text = first_tempo.text
#             elif d.text == 'a_tempo':
#                 print('a tempo', dir_by_end_times.get(d.start.t, None))

#         # exclude a_tempo:
#         # names = sorted(set(d.text for d in directions
#         #                    if d.text != 'a_tempo'))
#         names = sorted(set(d.text for d in directions))

#         name_to_idx = dict((name, idx) for idx, name in
#                            enumerate(names))
#         W = np.zeros((len(onsets), len(names)))
#         for i, d in enumerate(directions):
#             if d.text not in name_to_idx:
#                 continue
#             j = name_to_idx[d.text]

#             if indicator:
#                 W[:, j] += make_indicator_direction(onsets, d)
#             else:
#                 W[:, j] += make_local_tempo_direction_feature(onsets, d)

#         assert W.shape[1] == len(names)
#         # assert np.all(np.sum(W, 1) > 0)
#         # return FeatureBasis(soft_normalize(W, preserve_unity=True), cls.make_full_names(names))
#         # print(scorePart.pprint())
#         return FeatureBasis(W, cls.make_full_names(names))

# class IndicatorTempoAnnotationBasis(TempoAnnotationBasis):
#     @classmethod
#     def makeBasis(cls, scorePart):
#         return cls._makeBasis(scorePart, indicator=True)


# def make_indicator_direction(onsets, d):
#     v = np.zeros(len(onsets))
#     i = np.searchsorted(onsets, d.start.t)
#     if i < len(v):
#         v[i] = 1
#     return v


# def ensure_constant_at_start_end(dd, start, end, default_direction=u'mf'):
#     """
#     Make sure that there is a ConstantLoudnessDirection at the start and end
#     time. An "mf" Direction is insterted as a default when a
#     ConstantLoudnessDirection is missing.

#     Parameters
#     ----------

#     Returns
#     -------
#     """

#     constant = [d for d in dd
#                 if isinstance(d, ConstantLoudnessDirection)]

#     if not constant:
#         cld = ConstantLoudnessDirection(default_direction)
#         start.add_starting_object(cld)
#         dd.insert(0, cld)
#         end.add_ending_object(cld)
#     else:
#         if constant[0].start.t > start:
#             cld = ConstantLoudnessDirection(default_direction)
#             start.add_starting_object(cld)
#             dd.insert(0, cld)
#             constant[0].start.add_ending_object(cld)
#         if constant[-1].end.t < end:
#             cld = ConstantLoudnessDirection(default_direction)
#             constant[-1].end.add_starting_object(cld)
#             dd.append(cld)
#             end.add_ending_object(cld)
            
#     return dd


# def fill_out_loudness_direction_times(directions, max_time):
#     return fill_out_direction_times(directions, max_time,
#                                     ConstantLoudnessDirection,
#                                     DynamicLoudnessDirection,
#                                     ImpulsiveLoudnessDirection)

# def fill_out_tempo_direction_times(directions, max_time):
#     return fill_out_direction_times(directions, max_time,
#                                     ConstantTempoDirection,
#                                     DynamicTempoDirection)

# def fill_out_direction_times(directions, max_time, constant_type, dynamic_type, impulsive_type=None):
#     # for all dynamic: if unset end_time, end_time=timeOfNextConstant or lastTime
#     # for all impulsive: end_time = begin_time+epsilon
#     # for all constant: end_time = max(begin_time+epsilon,timeOfNextConstant)
#     # for last constant: end_time = lastTime
#     for i, d in enumerate(directions):
#         if isinstance(d, dynamic_type):
#             # print(d.text, d.start.t, d.end.t)
#             d.intermediate.append(d.end)
#             d.end = None

#     for i, d in enumerate(directions):

#         if impulsive_type and isinstance(d, impulsive_type):
#             directions[i].end = directions[i].start
#         else:

#             if d.end == None:

#                 # if type(d) in [ReferenceTempoDirection,ConstantTempoDirection]:
#                 #     next_list = [ReferenceTempoDirection,ConstantTempoDirection]
#                 # else:
#                 #     next_list = [ReferenceTempoDirection,ConstantLoudnessDirection,ConstantTempoDirection]
#                 # n = d.get_next_of_types(next_list)
#                 n = d.start.get_next_of_type(constant_type)
#                 if n:
#                     directions[i].end = max(d.start, n[0].start)
#                 else:
#                     directions[i].end = max_time

#     return directions


# class LoudnessAnnotationBasis(Basis):
#     # names = LoudnessDirection.get_labels()
#     directions = [LoudnessDirection]

#     @classmethod
#     def makeBasis(cls, scorePart):
#         """
#         Make basis functions with temporary effects, one basis
#         function for each type of mark
#         """
#         epsilon = 1e-4
#         onsets = np.array([n.start.t for n in scorePart.notes])

#         directions = scorePart.get_loudness_directions()
#         directions = fill_out_loudness_direction_times(
#             directions, scorePart.timeline.points[-1])

#         directions = ensure_constant_at_start_end(
#             directions,
#             scorePart.timeline.points[0],
#             scorePart.timeline.points[-1])

#         name2idx = {}
#         c = 0
#         names = []
#         for i, d in enumerate(directions):
#             n = d.text
#             if name2idx.has_key(n):
#                 continue
#             else:
#                 name2idx[n] = c
#                 c += 1
#                 names.append(n)

#         # W = np.zeros((len(onsets),len(u_names)))
#         W = np.zeros((len(onsets), c))
#         for i, d in enumerate(directions):
#             j = name2idx[d.text]
#             W[:, j] += make_local_loudness_direction_feature(onsets, d)
#         assert W.shape[1] == len(names)
#         return FeatureBasis(soft_normalize(W, preserve_unity=True), cls.make_full_names(names))


# class PolynomialPitchBasis(Basis):
#     names = ['pitch', 'pitch^2', 'pitch^3']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         Q = 127.0
#         pitches = np.array([n.midi_pitch for n in scorePart.notes])
#         W = np.zeros((len(pitches), 3))
#         W[:, 0] = pitches / Q
#         W[:, 1] = pitches**2 / Q**2
#         W[:, 2] = pitches**3 / Q**3
#         return FeatureBasis(normalize(W), cls.make_full_names())

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

#     @classmethod
#     def makeBasis_old(cls, scorePart):
#         basis_level = .25
#         slurs = scorePart.timeline.get_all_of_type(Slur)
#         if len(slurs) == 0:
#             W = np.zeros((len(scorePart.notes), 2), dtype=np.float)
#         else:
#             W = np.zeros((len(scorePart.notes), 2), dtype=np.float)

#             ss = np.array([(s.voice, s.start.t, s.end.t) for s in slurs])

#             # begin make arch
#             onsets = np.array([n.start.t for n in scorePart.notes])
#             first = np.min(onsets)
#             last = np.max(onsets)
#             eps = 10**-4

#             for v, start, end in ss:
#                 tmap = np.array([[min(first, start - eps),  basis_level],
#                                  [start - eps,              basis_level],
#                                  [start,                  -.5],
#                                  [end,                     .5],
#                                  [end + eps,                basis_level],
#                                  [max(last, end + eps),     basis_level]])
#                 m = interp1d(tmap[:, 0], tmap[:, 1])
#                 W[:, 1] -= m(onsets)**2

#             correct = len(ss) * basis_level**2

#             W[:, 1] += correct
#             # end make arch

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


# class DurationBasis(Basis):
#     names = ['duration']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         nd = np.array([(n.start.t, n.end.t) for n in scorePart.notes])

#         bm = scorePart.beat_map

#         note_times_beat = bm(nd[:, 1]) - bm(nd[:, 0])
#         W = note_times_beat
#         W.shape = (-1, 1)
#         return FeatureBasis(soft_normalize(W, preserve_unity=True), cls.make_full_names())

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



# class MetricalBasis(Basis):
#     meters = [(4, 4), (2, 2), (3, 4), (6, 8), (6, 4), (2, 8),
#               (3, 8), (12, 8), (5, 4), (2, 4), (9, 8), (12, 16)]
#     names = ['{0}_{1}_{2}'.format(beats, beat_type, position)
#              for beats, beat_type in meters for position in range(beats)] + \
#         ['{0}_{1}_weak'.format(beats, beat_type)
#          for beats, beat_type in meters]

#     @classmethod
#     def makeBasis(cls, scorePart):
#         nt = np.array([n.start.t for n in scorePart.notes])

#         bm = scorePart.beat_map

#         note_times_beat = bm(nt)

#         ts = scorePart.timeline.get_all_of_type(TimeSignature)

#         ms = scorePart.timeline.get_all_of_type(Measure)
#         assert len(ts) > 0

#         ts = np.array([(t.start.t, t.beats, t.beat_type) for t in ts])
#         tsi = np.searchsorted(ts[:, 0], nt, side='right') - 1

#         ms = bm(np.array([m.start.t for m in ms]))
#         msi = np.searchsorted(ms, note_times_beat, side='right') - 1
#         assert np.all(tsi >= 0)
#         assert np.all(msi >= 0)
#         eps = 10**-6
#         N = len(scorePart.notes)
#         basis_dict = defaultdict(lambda *x: np.zeros(N, dtype=np.float))

#         names_set = set(cls.names)

#         for i, n in enumerate(scorePart.notes):

#             m_position = note_times_beat[i] - ms[msi[i]]
#             if m_position % 1 > eps:
#                 m_position_label = 'weak'
#             else:
#                 m_position_label = '{0:d}'.format(int(m_position))
#             label = '{0:d}_{1:d}_{2}'.format(
#                 ts[tsi[i], 1], ts[tsi[i], 2], m_position_label)

#             if label in names_set:
#                 basis_dict[label][i] = 1
#             else:
#                 LOGGER.warning(('Produced an unsupported metrical label {0}. '
#                                 'Please check that time signatures are correct'
#                             ).format(label))

#         bases = tuple(basis_dict.items())
#         W = np.column_stack([b[1] for b in bases])
#         names = [b[0] for b in bases]
#         return FeatureBasis(W, cls.make_full_names(names))


# class BeatPhaseBasis(Basis):
#     """
#     This Basis function computes the relative location of an onset within the bar
#     """
#     names = ['beat_phase']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         nt = np.array([n.start.t for n in scorePart.notes])

#         bm = scorePart.beat_map

#         note_times_beat = bm(nt)

#         ts = scorePart.timeline.get_all_of_type(TimeSignature)

#         ms = scorePart.timeline.get_all_of_type(Measure)
#         assert len(ts) > 0

#         ts = np.array([(t.start.t, t.beats, t.beat_type) for t in ts])
#         tsi = np.searchsorted(ts[:, 0], nt, side='right') - 1

#         ms = bm(np.array([m.start.t for m in ms]))
#         msi = np.searchsorted(ms, note_times_beat, side='right') - 1
#         assert np.all(tsi >= 0)
#         assert np.all(msi >= 0)
#         eps = 10**-6
#         N = len(scorePart.notes)

#         W = np.zeros((N, 1))

#         for i, n in enumerate(scorePart.notes):

#             if note_times_beat[i] < eps:
#                 m_position = note_times_beat[i]
#             else:
#                 m_position = note_times_beat[i] - ms[msi[i]]
#             ts_num = ts[tsi[i], 1]
#             # b_phase = np.mod(m_position, ts_num)
#             W[i, 0] = np.mod(m_position, ts_num) / float(ts_num)

#         return FeatureBasis(W, cls.make_full_names(cls.names))
            

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


# class AccentBasis(Basis):
#     names = ['accent']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         W = np.array([n.accent for n in scorePart.notes], dtype=np.int)
#         W.shape = (-1, 1)
#         return FeatureBasis(W, cls.make_full_names())


# class StaccatoBasis(Basis):
#     names = ['staccato']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         W = np.array([n.staccato for n in scorePart.notes], dtype=np.int)
#         W.shape = (-1, 1)
#         return FeatureBasis(W, cls.make_full_names())


# class GraceBasis(Basis):
#     # names = ['grace', 'after_grace']
#     names = ['appoggiatura', 'after_appoggiatura',
#              'acciaccatura', 'before_acciaccatura']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         app_notes = [(i, n) for i, n in enumerate(scorePart.notes)
#                      if hasattr(n, 'appoggiatura_group_id')]
#         acc_notes = [(i, n) for i, n in enumerate(scorePart.notes)
#                      if hasattr(n, 'acciaccatura_group_id')]

#         W = np.zeros((len(scorePart.notes), len(cls.names)))

#         for i, n in app_notes:
#             if n.grace_type is None:
#                 W[i, 1] = n.appoggiatura_duration
#             else:
#                 W[i, 0] = n.appoggiatura_idx + 1

#         for i, n in acc_notes:
#             if n.grace_type is None:
#                 W[i, 3] = n.acciaccatura_duration
#             else:
#                 W[i, 2] = n.acciaccatura_idx + 1

#         # main_notes = np.where(np.diff(W) < 0)[0] + 1
#         # W = np.column_stack((W, np.zeros(W.shape[0])))
#         # W[main_notes, 1] = 1
#         return FeatureBasis(soft_normalize(W, preserve_unity=True), cls.make_full_names())


# class FermataBasis(Basis):
#     names = ['fermata']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         W = np.array([n.fermata for n in scorePart.notes], dtype=np.int)
#         W.shape = (-1, 1)
#         return FeatureBasis(W, cls.make_full_names())


# class ConstantBasis(Basis):
#     names = ['constant']

#     @classmethod
#     def makeBasis(cls, scorePart):
#         return FeatureBasis(np.ones((len(scorePart.notes), 1)), cls.make_full_names())


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


# class GroupBasis(Basis):
#     # the number of size classes for the groups
#     n_size = 10
#     names = (['whole_piece_asc', 'whole_piece_des'] +
#              [j.format(i) for i in range(n_size)
#               for j in ('group_size{0}_asc', 'group_size{0}_asc')])

#     # this should be set from outside before makeBasis is called
#     data_folder = None

#     @classmethod
#     def makeBasis(cls, scorePart):

#         # print scorePart.part_id, scorePart.part_name
#         note_info = np.array([(n.midi_pitch, n.start.t,
#                                n.end.t, int(n.id))
#                               for n in scorePart.notes])

#         # Initialize matrix of basis functions
#         W = np.zeros((len(note_info), cls.n_size * 2 + 2 ))

#         # Basis whole piece ascending and descending basis functions
#         unique_onsets = np.unique(note_info[:, 1])
#         ascending = np.linspace(0., 1., len(unique_onsets))
#         descending = np.linspace(1., 0., len(unique_onsets))

#         for onset, asc, desc in zip(
#                 unique_onsets, ascending, descending):
#             W[np.where(note_info[:, 1] == onset), 0] = asc
#             W[np.where(note_info[:, 1] == onset), 1] = desc

#         # print np.sort(note_info[:, 3])
#         if cls.data_folder is not None:
#             # Load grouping result file from Olivier's MiningSuite
#             fn = os.path.join(cls.data_folder,
#                               '{0}.txt'.format(scorePart.piece_name))
#             if not os.path.exists(fn):
#                 LOGGER.warning('Cannot find patterns for piece {0}'
#                                .format(scorePart.piece_name))
#             else:
#                 groups = np.loadtxt(fn, delimiter=',')
#                 # print(groups)
#                 # print(scorePart.piece_name)

#                 # check if the piece has more than one group
#                 # (otherwise, the only group is the whole piece)
#                 if len(groups) > 1:

#                     # length of groups
#                     group_length = groups[:, 1] - groups[:, 0]
#                     # mean group length
#                     g_mean = np.mean(group_length[1:])
#                     # standard deviation of group lengths
#                     g_std = np.std(group_length[1:])
#                     # set the maximal group length
#                     max_group_length = g_mean + 3 * g_std

#                     # Use groups longer than one note and
#                     # groups shorter than max allowed group length
#                     valid_idx = np.logical_and(
#                         group_length > 1,
#                         group_length <= max_group_length)

#                     # oldbins = np.linspace(1, max_group_length, cls.n_size + 1)
#                     percentiles = np.arange(1, cls.n_size + 1) * 100. / cls.n_size
#                     bins = np.percentile(group_length[valid_idx], percentiles)
#                     # bins[-1] += 1
#                     # import matplotlib.pyplot as plt
#                     # plt.hist(group_length[valid_idx], 40)
#                     # plt.show()

#                     # classify the groups according to their length
#                     # (into equal sized bins)
#                     group_length_digitized = np.digitize(
#                         group_length[valid_idx], bins, True)

#                     for (start, end), gl in zip(groups[valid_idx],
#                                                 group_length_digitized):
#                         # Get the indices of each group
#                         group_idx = np.logical_and(
#                             note_info[:, -1] >= start,
#                             note_info[:, -1] <= end)

#                         # Onsets of a group
#                         group_onsets = np.unique(
#                             note_info[np.where(group_idx), 1])

#                         # generate basis
#                         ascending = np.linspace(0, 1, len(group_onsets))
#                         descending = np.linspace(1, 0, len(group_onsets))

#                         # Assign ascending and descending indices
#                         # according to length of the group
#                         a_idx = 2 + gl * 2
#                         d_idx = 3 + gl * 2

#                         # Write values of the matrix of basis functions
#                         for onset, asc, desc in zip(
#                                 group_onsets, ascending, descending):
#                             W[np.where(note_info[:, 1] == onset), a_idx] = asc
#                             W[np.where(note_info[:, 1] == onset), d_idx] = desc

#         return FeatureBasis(W, cls.make_full_names())

# if __name__ == '__main__':
#     pass
