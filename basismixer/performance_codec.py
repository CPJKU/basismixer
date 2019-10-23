#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Codec
"""
import os
import sys
import numpy as np
import logging

import numpy.lib.recfunctions as rfn

from scipy.interpolate import interp1d
from scipy.misc import derivative

from basismixer.utils import clip

LOGGER = logging.getLogger(__name__)


class PerformanceCodec(object):
    """
    A Codec for decomposing a performance into expressive parameters
    """

    def __init__(self, time_codec, dynamics_codec,
                 default_values={}):
        super().__init__()
        self.time_codec = time_codec

        self.dynamics_codec = dynamics_codec

        self.parameter_names = (self.dynamics_codec.parameter_names +
                                self.time_codec.parameter_names)
        self.default_values = default_values

    def encode(self, matched_score, return_u_onset_idx=False):
        """
        Encode expressive parameters from a matched performance

        Parameters
        ----------
        matched_score : structured array
            Performance matched to its score.
        return_u_onset_idx : bool
            Return the indices of the unique score onsets
        """
        m_score, resort_idx = sort_matched_score(matched_score)
        # Get time-related parameters
        (time_params,
         unique_onset_idxs) = self.time_codec.encode(
            score_onsets=m_score['onset'],
            performed_onsets=m_score['p_onset'],
            score_durations=m_score['duration'],
            performed_durations=m_score['p_duration'],
            return_u_onset_idx=True)

        # Get dynamics-related parameters
        dynamics_params = self.dynamics_codec.encode(
            pitches=m_score['pitch'],
            velocities=m_score['velocity'],
            score_onsets=m_score['onset'],
            score_durations=m_score['duration'],
            unique_onset_idxs=unique_onset_idxs)

        # Concatenate parameters into a single matrix
        parameters = rfn.merge_arrays([time_params, dynamics_params],
                                      flatten=True, usemask=False)
        # parameters = np.array(
        #     [tuple(d) + tuple(t)
        #      for d, t in zip(dynamics_params, time_params)],
        #     dtype=[(pn, 'f4') for pn in self.parameter_names])
        # resort parameters to have the same order as the score
        parameters = parameters[resort_idx]

        unique_onset_idxs = [resort_idx[uix] for uix in unique_onset_idxs]

        if return_u_onset_idx:
            return parameters, unique_onset_idxs
        else:
            return parameters

    def decode(self, score, parameters,
               sanitize=True, *args, **kwargs):

        m_score, resort_idx, sort_idx = sort_matched_score(score,
                                                           return_sort_idx=True)
        pitches = m_score['pitch']

        clip(pitches, 1, 127)

        dynamics_params = parameters[list(self.dynamics_codec.parameter_names)][sort_idx]
        time_params = parameters[list(self.time_codec.parameter_names)][sort_idx]

        onsets_durations = self.time_codec.decode(
            score_onsets=m_score['onset'],
            score_durations=m_score['duration'],
            parameters=time_params, *args, **kwargs)

        velocities = self.dynamics_codec.decode(
            pitches=m_score['pitch'],
            score_onsets=m_score['onset'],
            score_durations=m_score['duration'],
            parameters=dynamics_params)

        clip(velocities, 1, 127)

        matched_performance = np.array(
            [(n['pitch'], n['onset'], n['duration'], v, od[0], od[1])
             for n, v, od in zip(m_score, velocities, onsets_durations)],
            dtype=[('pitch', 'i4'),
                   ('onset', 'f4'),
                   ('duration', 'f4'),
                   ('velocity', 'i4'),
                   ('p_onset', 'f4'),
                   ('p_duration', 'f4')])

        # TODO (outside of this function)
        # * Sanitize
        # * rescale according to default values
        return matched_performance[resort_idx]

#### Methods for Time-related parameters ####


def encode_articulation(score_durations, performed_durations,
                        unique_onset_idxs, beat_period):
    """
    Encode articulation
    """
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
    """
    Decode articulation
    """

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


#### Temo Parameter Normalizations ####

def bp_scale(beat_period):
    return [beat_period]


def bp_rescale(tempo_params):
    return tempo_params['beat_period']


def log_bp_scale(beat_period):
    return [np.log2(beat_period)]


def log_bp_rescale(tempo_params):
    return 2 ** tempo_params['log_bp']


def standardized_bp_scale(beat_period):
    std_bp = np.std(beat_period) * np.ones_like(beat_period)
    mean_beat_period = np.mean(beat_period) * np.ones_like(beat_period)
    standardized_bp = (beat_period - mean_beat_period) / std_bp
    return [standardized_bp, mean_beat_period, std_bp]


def standardized_bp_rescale(tempo_params):
    return (tempo_params['standardized_bp'] * tempo_params['std_bp'] +
            tempo_params['mean_beat_period'])


def bpr_scale(beat_period):
    mean_beat_period = np.mean(beat_period) * np.ones_like(beat_period)
    bpr = beat_period / mean_beat_period
    return [bpr, mean_beat_period]


def bpr_rescale(tempo_params):
    return tempo_params['bpr'] * tempo_params['mean_beat_period']


def log_bpr_scale(beat_period):
    bpr, mean_beat_period = bpr_scale(beat_period)
    return [np.log2(bpr), mean_beat_period]


def log_bpr_rescale(tempo_params):
    return (2 ** tempo_params['log_bpr'] * tempo_params['mean_beat_period'])


TEMPO_NORMALIZATION = dict(
    beat_period=dict(scale=bp_scale,
                     rescale=bp_rescale,
                     param_names=('beat_period',)),
    log_bp=dict(scale=log_bp_scale,
                rescale=log_bp_rescale,
                param_names=('log_bp',)),
    bpr=dict(scale=bpr_scale,
             rescale=bpr_rescale,
             param_names=('bpr', 'mean_beat_period')),
    log_bpr=dict(scale=log_bpr_scale,
                 rescale=log_bpr_rescale,
                 param_names=('log_bpr', 'mean_beat_period')),
    standardized_bp=dict(scale=standardized_bp_scale,
                         rescale=standardized_bp_rescale,
                         param_names=('standardized_bp', 'mean_beat_period', 'std_bp'))
)

#### Codecs for Dynamics ####


class NotewiseDynamicsCodec(object):
    """
    Performance Codec for encoding and decoding dynamics related
    expressive dymensions
    """
    parameter_names = ('velocity', )

    def encode(self, pitches, velocities, *args, **kwargs):
        return np.array(velocities, dtype=[('velocity', 'f4')])
        # return (velocities / 127.0).reshape(-1, 1)

    def decode(self, pitches, parameters, *args, **kwargs):

        velocity = parameters['velocity']
        if np.max(velocity) <= 1:
            return np.round(velocity * 127.0)
        else:
            return np.round(velocity)


class OnsetwiseDecompositionDynamicsCodec(object):
    """
    Performance Codec for encoding and decoding dynamics related
    expressive dimensions
    """
    parameter_names = ('velocity_trend', 'velocity_dev')

    def encode(self, pitches, velocities, score_onsets,
               score_durations, unique_onset_idxs=None,
               return_u_onset_idx=False,
               *args, **kwargs):

        if unique_onset_idxs is None:
            unique_onset_idxs = get_unique_onset_idxs(score_onsets)

        velocity_trend = np.array([np.max(velocities[uix]) for uix in unique_onset_idxs])

        parameters = np.zeros(len(pitches), dtype=[(pn, 'f4') for pn in self.parameter_names])
        for i, jj in enumerate(unique_onset_idxs):
            parameters['velocity_trend'][jj] = velocity_trend[i] / 127.0
            parameters['velocity_dev'][jj] = (velocity_trend[i] - velocities[jj]) / 127.0

        if return_u_onset_idx:
            return parameters, unique_onset_idxs
        else:
            return parameters

    def decode(self, parameters, *args, **kwargs):

        velocity = parameters['velocity_trend'] - parameters['velocity_dev']

        if np.max(velocity) <= 1:
            return np.round(velocity * 127.0)
        else:
            return np.round(velocity)

#### Codecs for time-related parameters ####


class TimeCodec(object):
    """
    Performance Codec for encoding and decoding time related expressive
    dimensions (tempo, timing and articulation)
    """

    def __init__(self,
                 tempo_fun=tempo_by_average,
                 normalization='beat_period'):
        # Get tempo normalization
        try:
            self.normalization = TEMPO_NORMALIZATION[normalization]
        except KeyError:
            raise KeyError('Unknown normalization specification: '
                           '{}'.format(normalization))

        tempo_param_name = self.normalization['param_names'][0]

        self.parameter_names = ((tempo_param_name, 'timing', 'log_articulation') +
                                self.normalization['param_names'][1:])

        self.tempo_fun = tempo_fun

    def encode(self, score_onsets, performed_onsets,
               score_durations, performed_durations,
               return_u_onset_idx=False):
        """
        Compute time-related performance parameters from a performance
        """

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

        # Compute beat period
        beat_period, s_onsets, unique_onset_idxs = self.tempo_fun(
            score_onsets=score[:, 0],
            performed_onsets=performance[:, 0],
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            return_onset_idxs=True)

        # Compute equivalent onsets
        eq_onsets = (np.cumsum(np.r_[0, beat_period[:-1] * np.diff(s_onsets)]) +
                     performance[unique_onset_idxs[0], 0].mean())

        # Compute tempo parameter
        tempo_params = self.normalization['scale'](beat_period)

        # Compute articulation parameter
        articulation_param = encode_articulation(
            score_durations=score[:, 1],
            performed_durations=performance[:, 1],
            unique_onset_idxs=unique_onset_idxs,
            beat_period=beat_period)

        # Initialize array of parameters
        parameters = np.zeros(len(score),
                              dtype=[(pn, 'f4') for pn in self.parameter_names])
        parameters['log_articulation'] = articulation_param
        for i, jj in enumerate(unique_onset_idxs):
            parameters[self.normalization['param_names'][0]][jj] = tempo_params[0][i]
            # Defined as in Eq. (3.9) in Thesis (pp. 34)
            parameters['timing'][jj] = eq_onsets[i] - performance[jj, 0]

            for pn, tp in zip(self.normalization['param_names'][1:], tempo_params[1:]):
                parameters[pn][jj] = tp[i]

        if return_u_onset_idx:
            return parameters, unique_onset_idxs
        else:
            return parameters

    def decode(self, score_onsets, score_durations,
               parameters, *args, **kwargs):
        """
        Decode a performance into onsets and durations in seconds
        for each note in the score.
        """
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

        time_param = np.array(
            [tuple([np.mean(parameters[pn][uix])
                    for pn in self.normalization['param_names']])
             for uix in unique_onset_idxs],
            dtype=[(pn, 'f4') for pn in self.normalization['param_names']])

        beat_period = self.normalization['rescale'](time_param)

        ioi_perf = diff_u_onset_score * beat_period

        eq_onset = np.cumsum(np.r_[0, ioi_perf])

        performance = np.zeros((len(score_onsets), 2))

        for i, jj in enumerate(unique_onset_idxs):
            # decode onset
            performance[jj, 0] = eq_onset[i] - parameters['timing'][jj]
            # decode duration
            performance[jj, 1] = decode_articulation(
                score_durations=score_durations[jj],
                articulation_parameter=parameters['log_articulation'][jj],
                beat_period=beat_period[i])

        performance[:, 0] -= np.min(performance[:, 0])

        return performance

#### Miscellaneous utilities ####


def sort_matched_score(matched_score, return_sort_idx=False):
    """
    Sort score by score onset and then by pitch

    Parameters
    ----------
    matched_score : structured array
        Matched score
    return_sort_idx : bool
        Return indices for sorting the matched_score

    Returns
    -------
    m_score : structured array
        Sorted Matched score
    resort_idx : numpy array
        Indices to resort m_score to the original order
    sort_idx : numpy array
        Indices to sort `matched_score` into `m_score`.
        only returned if `return_sort_idx` is True.
    """

    pitch_sort_idx = np.argsort(matched_score['pitch'], kind='mergesort')
    onset_sort_idx = np.argsort(matched_score['onset'][pitch_sort_idx],
                                kind='mergesort')

    sort_idx = np.arange(len(matched_score),
                         dtype=np.int)[pitch_sort_idx][onset_sort_idx]
    resort_idx = np.argsort(sort_idx)
    m_score = matched_score[sort_idx]

    if return_sort_idx:
        return m_score, resort_idx, sort_idx
    else:
        return m_score, resort_idx


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


def _is_monotonic(s, mask, strict=True, deltas=None):
    """Return True when `s[mask]` is monotonic. When `strict` is True,
    check for strict monotonicity.

    Parameters
    ----------
    s : ndarray
        a sequence of numbers

    mask : ndarray(bool)
        a boolean mask to be applied to `s`

    strict : bool
        when True, return a strictly monotonic sequence (default: True)

    """

    if s[mask].shape[0] < 4:
        return True
    else:
        if strict:
            return np.all(np.diff(s[mask]) / np.diff(deltas[mask]) > 0.0)
        else:
            return np.all(np.diff(s[mask]) / np.diff(deltas[mask]) >= 0.0)


def monotonize_times(s, strict=True, deltas=None, return_interp_seq=False):
    """Interpolate linearly over as many points in `s` as necessary to
    obtain a monotonic sequence. The minimum and maximum of `s` are
    prepended and appended, respectively, to ensure monotonicity at
    the bounds of `s`.

    Parameters
    ----------
    s : ndarray
        a sequence of numbers
    strict : bool
        when True, return a strictly monotonic sequence (default: True)

    Returns
    -------
    ndarray
       a monotonic sequence that has been linearly interpolated using a subset of s

    """
    if strict:
        eps = np.finfo(np.float32).eps
    else:
        eps = 1e-3

    _s = np.r_[np.min(s) - eps, s, np.max(s) + eps]
    if deltas is not None:
        _deltas = np.r_[np.min(deltas) - eps, deltas, np.max(deltas) + eps]
    else:
        _deltas = None
    mask = np.ones(_s.shape[0], dtype=np.bool)
    monotonic_mask = _monotonize_times(_s, mask, strict, _deltas)[1:-1]  # is the mask always changed in place?
    mask[0] = mask[-1] = False
    # s_mono = interp1d(idx[mask], _s[mask])(idx[1:-1])
    if return_interp_seq:
        idx = np.arange(_s.shape[0])
        s_mono = interp1d(idx[mask], _s[mask])(idx[1:-1])
        return s_mono
    else:
        return _s[mask], _deltas[mask]


def _select_problematic(s, mask, deltas=None):
    """Return the index i of an element of s, such that removing s[i]
    reduces the non-monotonicity of s[mask]. This method assumes the
    first and last element of `s` are the the minimum and maximum,
    respectively.

    Parameters
    ----------
    s : ndarray
        a sequence of numbers

    mask : ndarray(bool)
        a boolean mask to be applied to s

    Returns
    -------
    int
       an index i such that removing s[i] from s reduces
       non-monotonicity

    """
    if deltas is not None:
        diffs = np.diff(s[mask]) / np.diff(deltas[mask])
    else:
        diffs = np.diff(s[mask])
    # ignore first and last elements
    problematic = np.argmin(diffs[1:] * diffs[:-1]) + 1
    return np.where(mask)[0][problematic]


def _monotonize_times(s, mask, strict, deltas=None):
    """
    Return a mask such that s[mask] is monotonic.

    """

    # print('is monotonic?')
    if _is_monotonic(s, mask, strict, deltas):
        # print('yes')
        return mask
    else:
        # print('no')
        p = _select_problematic(s, mask, deltas)
        # print('select problematic', p)
        mask[p] = False

        return _monotonize_times(s, mask, strict, deltas)
