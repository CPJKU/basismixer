#!/usr/bin/env python

import os
from collections import defaultdict

import numpy as np

def pair_files(dir_dict, remove_incomplete=True, full_path=True):
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
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                name = os.path.splitext(f)[0]
                if full_path:
                    result[name][label] = path
                else:
                    result[name][label] = f

    if remove_incomplete:
        labels = set(dir_dict.keys())
        todo_delete = [k for k, k_labels in result.items()
                       if not set(k_labels) == labels]
        for k in todo_delete:
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

