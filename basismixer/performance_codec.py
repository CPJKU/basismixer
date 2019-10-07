#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Codec
"""
import os
import sys
import numpy as np
import logging

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
