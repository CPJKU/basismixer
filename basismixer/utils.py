#!/usr/bin/env python

import numpy as np


def pair_files(dir_dict, remove_incomplete=True):
    """Pair files in directories; dir_dict is of form (label: directory)

    TODO: complete docs below:

    Parameters
    ----------
    dir_dict : type
        Description of `dir_dict`
    remove_incomplete : type, optional
        Description of `remove_incomplete`

    Returns
    -------
    type
        Description of return value

    """
    result = defaultdict(dict)
    for label, directory in dir_dict.items():
        for f in os.listdir(directory):
            name = os.path.splitext(f)[0]
            result[name][label] = f

    if remove_incomplete:
        labels = dir_dict.keys()
        for k in result.keys():
            if not all([y in result[k] for y in labels]):
                del result[k]

    return result


def clip(v, low=0, high=127):
    """Clip values in `v` to the range `[low, high]`. The array is
    modified in-place.

    Parameters
    ----------
    v : ndarray
        Array of numbers
    low : number, optional
        Low bound. Defaults to 0.
    high : number, optional
        High bound. Defaults to 127.

    """
    too_low = np.where(v < low)[0]

    if len(too_low) > 0:
        # LOGGER.warning('Clipping {} low values'.format(too_low))
        v[too_low] = low

    too_high = np.where(v > high)[0]

    if len(too_high) > 0:
        # LOGGER.warning('Clipping {} high values'.format(too_high))
        v[too_high] = high


def extract_matched_performance(score_part, performed_part,
                                snote_idx, note_idx):

    score_array = score_part.note_array

    perf_array = performed_part.note_array

    score_info = score_array[snote_idx]
    perf_info = perf_array[note_idx]
