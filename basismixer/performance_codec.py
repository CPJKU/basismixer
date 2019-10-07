#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Codec
"""
import os
import sys
import numpy as np
import logging

from scipy.interpolate import interp1d
from scipy.misc import derivative


LOGGER = logging.getLogger(__name__)


def restrict_to_uint8(values, name=''):
    too_low_values = np.where(values < 1)[0]
    if len(too_low_values) > 0:
        LOGGER.warning('Setting {0} values < 1 to 1'.format(name))
        values[too_low_values] = 1
    too_high_values = np.where(values > 127)[0]
    if len(too_high_values) > 0:
        LOGGER.warning('Setting {0} values > 127 to 127'.format(name))
        values[too_high_values] = 127


class PerformanceCodec(object):
    def __init__(self, time_codec, dynamics_codec, codec_version,
                 default_values={}):

        self.codec_version = codec_version

        self.time_codec = time_codec

        self.dynamics_codec = dynamics_codec

        self.parameter_names = (self.dynamics_codec.parameter_names
                                + self.time_codec.parameter_names)

        self.default_values = default_values

    def encode(self, matched_score, return_u_onset_idx=False):

        (time_params, mean_beat_period,
         unique_onset_idxs) = self.time_codec.encode(
            score_onsets=matched_score['onset'],
            performed_onsets=matched_score['p_onset'],
            score_durations=matched_score['duration'],
            performed_durations=matched_score['p_duration'],
            return_u_onset_idx=True)

        dynamics_params = self.dynamics_codec.encode(
            pitches=matched_score['pitch'],
            velocities=matched_score['velocity'],
            score_onsets=matched_score['onset'],
            score_durations=matched_score['duration'],
            unique_onset_idxs=unique_onset_idxs)

        parameters = np.column_stack((dynamics_params, time_params))

        output = [parameters, mean_beat_period]

        if return_u_onset_idx:
            output.append(unique_onset_idxs)

        return output

    def decode(self, score, parameters, mean_beat_period, midi_out_fn=None,
               time_div=500, tempo=500000, sanitize=True, *args, **kwargs):

        pitches = score['pitch']

        restrict_to_uint8(pitches, 'pitch')

        time_params = parameters[:, len(self.dynamics_codec.parameter_names):]

        dynamics_params = parameters[:, :len(self.dynamics_codec.parameter_names)]

        onsets_durations = self.time_codec.decode(
            score_onsets=score['onset'],
            score_durations=score['duration'],
            parameters=time_params,
            mean_beat_period=mean_beat_period, *args, **kwargs)

        velocities = self.dynamics_codec.decode(
            pitches=score['pitch'],
            score_onsets=score['onset'],
            score_durations=score['duration'],
            parameters=dynamics_params)

        restrict_to_uint8(velocities, 'velocity')

        matched_performance = np.array(
            [(n['pitch'], n['onset'], n['duration'], v, od[0], od[1])
             for n, v, od in zip(score, velocities, onsets_durations)],
            dtype=[('pitch', 'i4'),
                   ('onset', 'f4'),
                   ('duration', 'f4'),
                   ('velocity', 'i4'),
                   ('p_onset', 'f4'),
                   ('p_duration', 'f4')])

        # TODO
        # * Sanitize
        # * Export MIDI

        return matched_performance


class NotewiseDynamicsCodec(object):
    """
    Performance Codec for encoding and decoding dynamics related
    expressive dymensions
    """

    def __init__(self, parameter_names='velocity'):

        self.parameter_names = (parameter_names, )

    def encode(self, pitches, velocities, *args, **kwargs):

        return velocities / 127.0

    def decode(self, pitches, parameters, *args, **kwargs):

        if np.max(parameters) <= 1:
            return np.round(parameters * 127.0)
        else:
            return np.round(parameters)


class TimeCodec(object):
    """
    Performance Codec for encoding and decoding time related expressive
    dimensions (tempo, timing and articulation)
    """

    def __init__(self,
                 parameter_names='auto',
                 tempo_fun=tempo_by_average,
                 normalization=None):

        if normalization is None:
            normalization = TempoNormalization()

        self.normalization = normalization
        if parameter_names == 'auto':
            self.parameter_names = (self.normalization.name,
                                    'timing',
                                    'log_articulation')
        elif isinstance(parameter_names, (list, tuple)):
            self.parameter_names = parameter_names

        self.tempo_fun = tempo_fun

    def encode(self, score_onsets, performed_onsets,
               score_durations, performed_durations,
               return_u_onset_idx=False):

        if score_onsets.shape != performed_onsets.shape:
            raise ValueError('The performance and the score should be of '
                             'the same size')

        # use float64, float32 led to problems that x == x + eps
        # evaluated to True
        score_onsets = score_onsets.astype(np.float64, copy=False)
        performed_onsets = performed_onsets.astype(np.float64, copy=False)
        score_durations = score_durations.astype(np.float64, copy=False)
        performed_durations = performed_durations.astype(np.float64, copy=False)
        score = np.column_stack((score_onsets, score_durations))
        performance = np.column_stack((performed_onsets, performed_durations))

        idx, br = segment_score_perf_times(np.column_stack((score_onsets, performed_onsets)))

        score_segs = np.split(score[idx], br)
        perf_segs = np.split(performance[idx], br)
        idx_segs = np.split(idx, br)

        res = []
        segment_lengths = []
        k = 0

        # beat period, timing and articulation
        parameters = np.empty((len(score), 3))

        for ii, s, p in zip(idx_segs, score_segs, perf_segs):
            segment_results = list(self._encode(s, p, return_u_onset_idx))
            parameters[ii] = segment_results[0]
            res.append(segment_results)
            if return_u_onset_idx:
                res[-1][2] = [np.sort(idx[x + k]) for x in res[-1][2]]

            l = len(res[-1][0])
            segment_lengths.append(l)
            k += l

        LOGGER.debug('Segment lengths {0}'.format(segment_lengths))

        mbps = np.vstack([(l, x[1]) for l, x in zip(segment_lengths, res)])
        mbps[:, 0] /= np.sum(mbps[:, 0])
        mbp = np.sum(np.prod(mbps, axis=1))

        if return_u_onset_idx:
            # funny way to append the u_onset_idx lists of all segments:
            u_onset_idx = sum((x[2] for x in res), [])
            return parameters, mbp, u_onset_idx
        else:
            return parameters, mbp

    def _encode(self, score, performance,
                return_u_onset_idx=False, segment_idx=0):
        # Compute beat period
        beat_period, s_onsets, unique_onset_idxs = self.tempo_fun(
            score_onsets=score[:, 0],
            performed_onsets=performance[:, 0],
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            return_onset_idxs=True)

        # Compute equivalent onsets
        eq_onsets = (np.cumsum(np.r_[0, beat_period[:-1] * np.diff(s_onsets)])
                     + performance[unique_onset_idxs[0], 0].mean())

        # Compute tempo parameter
        tempo_param = self.normalization.normalize_tempo(beat_period)

        # Compute articulation parameter
        articulation_param = encode_articulation(
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            unique_onset_idxs=unique_onset_idxs,
            beat_period=beat_period)

        # Initialize array of parameters
        parameters = np.zeros((score.shape[0], 3))
        parameters[:, 2] = articulation_param
        for i, jj in enumerate(unique_onset_idxs):
            parameters[jj, 0] = tempo_param[i]
            # Defined as in Eq. (3.9) in Thesis (pp. 34)
            parameters[jj, 1] = eq_onsets[i] - performance[jj, 0]

        if return_u_onset_idx:
            return parameters, beat_period.mean(), unique_onset_idxs
        else:
            return parameters, beat_period.mean()

    def decode(self, score_onsets, score_durations, parameters,
               mean_beat_period, *args, **kwargs):

        score_onsets = score_onsets.astype(np.float64, copy=False)
        score_durations = score_durations.astype(np.float64, copy=False)

        score_info = get_unique_seq(onsets=score_onsets,
                                    offsets=score_onsets + score_durations,
                                    unique_onset_idxs=None,
                                    return_diff=True)
        unique_s_onsets = score_info['u_onset']
        unique_onset_idxs = score_info['unique_onset_idxs']
        total_dur = score_info['total_dur']
        diff_u_onset_score = score_info['diff_u_onset']

        time_param = np.array([np.mean(parameters[uix, 0])
                               for uix in unique_onset_idxs])

        beat_period = self.normalization.rescale_tempo(time_param,
                                                       mean_beat_period,
                                                       *args, **kwargs)

        ioi_perf = diff_u_onset_score * beat_period

        eq_onset = np.cumsum(np.r_[0, ioi_perf])

        performance = np.zeros((score_onsets.shape[0], 2))

        for i, jj in enumerate(unique_onset_idxs):
            # decode onset
            performance[jj, 0] = eq_onset[i] - parameters[jj, 1]
            # decode duration
            performance[jj, 1] = decode_articulation(
                score_durations=score_durations[jj],
                articulation_parameter=parameters[jj, 2],
                beat_period=beat_period[i])

        performance[:, 0] -= np.min(performance[:, 0])

        return performance


def encode_articulation(score_durations, performed_durations,
                        unique_onset_idxs, beat_period):
    articulation = np.zeros_like(score_durations)
    for idx, bp in zip(unique_onset_idxs, beat_period):

        sd = score_durations[idx]
        pd = performed_durations[idx]

        # indices of notes with duration 0 (grace notes)
        grace_mask = sd == 0

        # Grace notes have an articulation ratio of 1
        sd[grace_mask] = 1
        pd[grace_mask] = bp
        # Compute log articulation ratio
        articulation[idx] = np.log2(pd / (bp * sd))

    return articulation


def decode_articulation(score_durations, articulation_parameter,
                        beat_period):

    art_ratio = (2 ** articulation_parameter)
    dur = art_ratio * score_durations * beat_period

    return dur


def tempo_by_average(score_onsets, performed_onsets,
                     score_durations, performed_durations,
                     unique_onset_idxs=None,
                     input_onsets=None,
                     return_onset_idxs=False):
    """
    Computes a tempo curve using the average of the onset times of all
    notes belonging to the same score onset.

    Parameters
    ----------
    score_onsets : np.ndarray
        Onset in beats of each note in the score.
    performed_onsets : np.ndarray
        Performed onsets in seconds of each note in the score.
    score_durations : np.ndarray
        Duration in beats of each note in the score.
    performed_durations : np.ndarray
        Performed duration in seconds of each note in the score.
    unique_onset_idxs : np.ndarray or None (optional)
        Indices of the notes with the same score onset. (By default is None,
        and is therefore, inferred from `score_onsets`).
    input_onsets : np.ndarray or None
        Input onset times in beats at which the tempo curve is to be
        sampled (by default is None, which means that the tempo curve
        is returned for each unique score onset)
    return_onset_idxs : bool
        Return the indices of the unique score onsets (Default is False)

    Returns
    -------
    tempo_curve : np.ndarray
        Tempo curve in seconds per beat (spb). If `input_onsets` was provided,
        this array contains the value of the tempo in spb for each onset
        in `input_onsets`. Otherwise, this array contains the value of the 
        tempo in spb for each unique score onset.
    input_onsets : np.ndarray
        The score onsets corresponding to each value of the tempo curve.
    unique_onset_idxs: list
        Each element of the list is an array of the indices of the score
        corresponding to the elements in `tempo_curve`. Only returned if
        `return_onset_idxs` is True.
    """
    # use float64, float32 led to problems that x == x + eps evaluated
    # to True
    score_onsets = np.array(score_onsets).astype(np.float64, copy=False)
    performed_onsets = np.array(performed_onsets).astype(np.float64, copy=False)
    score_durations = np.array(score_durations).astype(np.float64, copy=False)

    performed_durations = np.array(performed_durations).astype(np.float64, copy=False)

    # Get unique onsets if no provided
    if unique_onset_idxs is None:
        # Get indices of the unique onsets (quantize score onsets)
        unique_onset_idxs = get_unique_onset_idxs((1e4 * score_onsets).astype(np.int))

    # Get score information
    score_info = get_unique_seq(onsets=score_onsets,
                                offsets=score_onsets + score_durations,
                                unique_onset_idxs=unique_onset_idxs,
                                return_diff=False)
    # Get performance information
    perf_info = get_unique_seq(onsets=performed_onsets,
                               offsets=performed_onsets + performed_durations,
                               unique_onset_idxs=unique_onset_idxs,
                               return_diff=False)

    # unique score onsets
    unique_s_onsets = score_info['u_onset']
    # equivalent onsets
    eq_onsets = perf_info['u_onset']

    # Monotonize times
    eq_onset_mt, unique_s_onsets_mt = monotonize_times(eq_onsets,
                                                       deltas=unique_s_onsets)

    # Estimate Beat Period
    perf_iois = np.diff(eq_onset_mt)
    s_iois = np.diff(unique_s_onsets_mt)
    beat_period = perf_iois / s_iois

    # if len(unique_s_onsets_mt) - 1 != len(beat_period):
    #     import pdb
    #     pdb.set_trace()
    tempo_fun = interp1d(unique_s_onsets_mt[:-1], beat_period, kind='linear',
                         bounds_error=False,
                         fill_value=(beat_period[0], beat_period[-1]))

    if input_onsets is None:
        input_onsets = unique_s_onsets[:-1]

    tempo_curve = tempo_fun(input_onsets)

    output = [tempo_curve, input_onsets]

    if return_onset_idxs:
        output.append(unique_onset_idxs)

    return output


def tempo_by_derivative(score_onsets, performed_onsets,
                        score_durations, performed_durations,
                        unique_onset_idxs=None,
                        input_onsets=None,
                        return_onset_idxs=False):
    """
    Computes a tempo curve using the derivative of the average performed 
    onset times of all notes belonging to the same score onset with respect
    to that score onset. This results in a curve that is smoother than the
    tempo estimated using `tempo_by_average`.

    Parameters
    ----------
    score_onsets : np.ndarray
        Onset in beats of each note in the score.
    performed_onsets : np.ndarray
        Performed onsets in seconds of each note in the score.
    score_durations : np.ndarray
        Duration in beats of each note in the score.
    performed_durations : np.ndarray
        Performed duration in seconds of each note in the score.
    unique_onset_idxs : np.ndarray or None (optional)
        Indices of the notes with the same score onset. (By default is None,
        and is therefore, inferred from `score_onsets`).
    input_onsets : np.ndarray or None
        Input onset times in beats at which the tempo curve is to be
        sampled (by default is None, which means that the tempo curve
        is returned for each unique score onset)
    return_onset_idxs : bool
        Return the indices of the unique score onsets (Default is False)

    Returns
    -------
    tempo_curve : np.ndarray
        Tempo curve in seconds per beat (spb). If `input_onsets` was provided,
        this array contains the value of the tempo in spb for each onset
        in `input_onsets`. Otherwise, this array contains the value of the 
        tempo in spb for each unique score onset.
    input_onsets : np.ndarray
        The score onsets corresponding to each value of the tempo curve.
    unique_onset_idxs: list
        Each element of the list is an array of the indices of the score
        corresponding to the elements in `tempo_curve`. Only returned if
        `return_onset_idxs` is True.
    """
    # use float64, float32 led to problems that x == x + eps evaluated
    # to True
    score_onsets = np.array(score_onsets).astype(np.float64, copy=False)
    performed_onsets = np.array(performed_onsets).astype(np.float64, copy=False)
    score_durations = np.array(score_durations).astype(np.float64, copy=False)

    performed_durations = np.array(performed_durations).astype(np.float64, copy=False)

    # Get unique onsets if no provided
    if unique_onset_idxs is None:
        # Get indices of the unique onsets (quantize score onsets)
        unique_onset_idxs = get_unique_onset_idxs((1e4 * score_onsets).astype(np.int))

    # Get score information
    score_info = get_unique_seq(onsets=score_onsets,
                                offsets=score_onsets + score_durations,
                                unique_onset_idxs=unique_onset_idxs,
                                return_diff=False)
    # Get performance information
    perf_info = get_unique_seq(onsets=performed_onsets,
                               offsets=performed_onsets + performed_durations,
                               unique_onset_idxs=unique_onset_idxs,
                               return_diff=False)

    # unique score onsets
    unique_s_onsets = score_info['u_onset']
    # equivalent onsets
    eq_onsets = perf_info['u_onset']

    # Monotonize times
    eq_onset_mt, unique_s_onsets_mt = monotonize_times(eq_onsets,
                                                       deltas=unique_s_onsets)
    # Function that that interpolates the equivalent performed onsets
    # as a function of the score onset.
    onset_fun = interp1d(unique_s_onsets_mt, eq_onset_mt, kind='linear',
                         fill_value='extrapolate')

    if input_onsets is None:
        input_onsets = unique_s_onsets[:-1]

    tempo_curve = derivative(onset_fun, input_onsets, dx=0.5)

    output = [tempo_curve, input_onsets]

    if return_onset_idxs:
        output.append(unique_onset_idxs)

    return output


class TempoNormalization(object):
    def __init__(self, name='beat_period'):
        self.name = name

    def normalize_tempo(self, tempo_curve):
        return tempo_curve

    def rescale_tempo(self, tempo_param, mean_beat_period,
                      *args, **kwargs):
        return tempo_param


class LogTempoNormalization(TempoNormalization):
    def __init__(self):
        super().__init__('log_bp')

    def normalize_tempo(self, tempo_curve):
        return np.log2(tempo_curve)

    def rescale_tempo(self, tempo_param, mean_beat_period):
        return 2 ** tempo_param


class StandardTempoNormalization(TempoNormalization):
    def __init__(self):
        super().__init__('standardized_bp')

    def normalize_tempo(self, tempo_curve):
        return (tempo_curve - np.mean(tempo_curve)) / np.std(tempo_curve)

    def rescale_tempo(self, tempo_param, mean_beat_period, std_bp):
        return tempo_param * std_bp + mean_beat_period


class MeanTempoNormalization(TempoNormalization):
    def __init__(self):
        super().__init__('bpr')

    def normalize_tempo(self, tempo_curve):
        return tempo_curve / tempo_curve.mean()

    def rescale_tempo(self, tempo_param, mean_beat_period):
        return tempo_param * mean_beat_period


class LogRelTempoNormalization(TempoNormalization):
    def __init__(self):
        super().__init__('log_bpr')

    def normalize_tempo(self, tempo_curve):
        return np.log2(tempo_curve / tempo_curve.mean())

    def rescale_tempo(self, tempo_param, mean_beat_period):
        return (2**tempo_param) * mean_beat_period


def get_unique_onset_idxs(onsets, eps=1e-6, return_unique_onsets=False):
    """
    Get unique onsets and their indices.

    Parameters
    ----------
    onsets : np.ndarray
        Score onsets in beats.
    eps : float
        Small epsilon (for dealing with quantization in symbolic scores).
        This is particularly useful for dealing with triplets and other
        similar rhytmical structures that do not have a finite decimal 
        representation.
    return_unique_onsets : bool (optional)
        If `True`, returns the unique score onsets.

    Returns
    -------
    unique_onset_idxs : np.ndarray
        Indices of the unique onsets in the score.
    unique_onsets : np.ndarray
        Unique score onsets
    """
    # Do not assume that the onsets are sorted
    # (use a stable sorting algorithm for preserving the order
    # of elements with the same onset, which is useful e.g. if the
    # notes within a same onset are ordered by pitch)
    sort_idx = np.argsort(onsets, kind='mergesort')
    split_idx = np.where(np.diff(onsets[sort_idx]) > eps)[0] + 1
    unique_onset_idxs = np.split(sort_idx, split_idx)

    if return_unique_onsets:
        # Instead of np.unique(onsets)
        unique_onsets = np.array([onsets[uix].mean()
                                  for uix in unique_onset_idxs])

        return unique_onset_idxs, unique_onsets
    else:
        return unique_onset_idxs


def get_unique_seq(onsets, offsets, unique_onset_idxs=None,
                   return_diff=False):
    """
    Get unique onsets of a sequence of notes
    """
    eps = np.finfo(np.float32).eps

    first_time = np.min(onsets)

    # ensure last score time is later than last onset
    last_time = max(np.max(onsets) + eps, np.max(offsets))

    total_dur = last_time - first_time

    if unique_onset_idxs is None:
        # unique_onset_idxs = unique_onset_idx(score[:, 0])
        unique_onset_idxs = get_unique_onset_idxs(onsets)

    u_onset = np.array([np.mean(onsets[uix]) for uix in unique_onset_idxs])
    # add last offset, so we have as many IOIs as notes
    u_onset = np.r_[u_onset, last_time]

    output_dict = dict(
        u_onset=u_onset,
        total_dur=total_dur,
        unique_onset_idxs=unique_onset_idxs)

    if return_diff:
        output_dict['diff_u_onset'] = np.diff(u_onset)

    return output_dict


def segment_score_perf_times(score_perf_onsets):
    """
    Segment a list of pairs (score onset, performance onset), based on the
    euclidean distance between (chronological) subsequent pairs.

    Parameters
    ----------

    score_perf_onsets : ndarray
        ndarray of shape N x 2, conveying N pairs of score and performance
        onsets, respectively

    Returns
    -------

    sort_idx : ndarray
        Index that sorts `score_performance_onsets` correctly for segmentation

    breakpoints: list
        Indices of segment start elements in
        `score_performance_onsets[sort_idx]`. The index of the first segment
        (0) is not included.
    """

    assert len(score_perf_onsets) > 0
    y = score_perf_onsets[:, :]

    # y is sorted according to increasing score onsets; within score onsets, y
    # is sorted according to increasing performance onsets:

    # secondary sort: performance time
    i1 = np.argsort(y[:, 1])
    # primary sort: score time
    i2 = np.argsort(y[i1, 0], kind='mergesort')
    sort_idx = i1[i2]
    y = y[sort_idx, :]

    # quantize score onsets to determine unique score onsets
    uox = get_unique_onset_idxs((y[:, 0] * 10000).astype(np.int))

    u_onset_score = np.array([np.mean(y[x, 0]) for x in uox])
    u_onset_perf = np.array([np.mean(y[x, 1]) for x in uox])

    # version of y that contains unique score/perf onsets
    y_u = np.column_stack((u_onset_score, u_onset_perf))

    # rescale score and performance times to [0,1], to compute
    # euclidean distance
    y_u -= np.min(y_u, axis=0)
    y_u /= np.max(y_u, axis=0)

    diff_u = np.diff(y_u, axis=0)
    dist_u = np.sum(diff_u ** 2, axis=1) ** .5

    med_d = np.median(dist_u)
    max_d = np.max(dist_u)
    max_med_ratio = max_d / med_d

    # segment the performance if max distance between adjacent points is more
    # than max_ratio times larger than the median:
    max_ratio = 10
    do_segment = max_med_ratio > max_ratio

    # minimal size of a segment. If the segmentation yields segments smaller
    # than min_seg_length, no segmentation will be applied
    min_seg_length = 3

    if do_segment:
        breakpoints_u = find_segment_boundaries(dist_u)
        if np.any(np.diff(np.r_[0, breakpoints_u, len(dist_u)]) < min_seg_length):
            breakpoints = []
        else:
            breakpoints = np.searchsorted(y[:, 0], u_onset_score[breakpoints_u], side='right')
    else:
        breakpoints = []

    return sort_idx, breakpoints


def find_segment_boundaries(dist, max_from_best=10):
    """
    Determine segment boundaries by segmenting using a greedy search
    (segmenting at the largest gaps first). The loss of the segmentation
    is measured by the ratio (max dist within segments) / (min dist at 
    segment boundaries).

    Parameters
    ----------

    dist : ndarray
        Array of euclidean distance values

    max_from_best : int
        Stop the search if the best loss value was observed more than
        `max_from_best` candidates ago (and return the best segmentation
        up to that point)

    Returns
    -------

    ndarray
        Array of segmentation boundaries (start index of each segment, 
        excluding 0)

    """
    sidx = np.argsort(dist)[::-1]
    bp = []
    best_loss = eval_segments(dist, bp)
    best_i = 0
    since_best = 0
    for i in range(len(sidx)):
        loss = eval_segments(dist, sorted(sidx[:i + 1]))
        if loss < best_loss:
            best_loss = loss
            best_i = i
        else:
            since_best += 1
        if since_best > max_from_best:
            break
    return sorted(sidx[:best_i + 1])


def eval_segments(dist, bp):
    """
    Compute the ratio (max dist within segments) / (min dist at segment
    boundaries). If there is only one segment, return the maximal within segment
    distance.

    Parameters
    ----------

    dist : ndarray
        Array of euclidean distance values

    bp : ndarray
        Array of segmentation boundaries (start index of each segment,
        excluding 0).

    Returns
    -------

    float
        Loss value (lower means better segmentation)
    """

    N = len(dist)
    idx = np.zeros(N, np.bool)
    idx[bp] = True
    if len(bp) == 0:
        within_dist = np.max(dist[~idx])
        return within_dist
    else:
        within_dist = np.max(dist[~idx])
        between_dist = np.min(dist[idx])
        return within_dist / between_dist
