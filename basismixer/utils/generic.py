#!/usr/bin/env python
import pickle
import bz2
import os
import tempfile
from collections import defaultdict

import numpy as np


def load_pyc_bz(fn):
    return pickle.load(bz2.BZ2File(fn, 'r'))


def save_pyc_bz(d, fn):
    pickle.dump(d, bz2.BZ2File(fn, 'w'), pickle.HIGHEST_PROTOCOL)


def to_memmap(a, folder=None):

    if folder:
        os.makedirs(folder, exist_ok=True)

    f = tempfile.NamedTemporaryFile(suffix='.npy', dir=folder)
    np.save(f.name, a)
    a_memmap = np.load(f.name, mmap_mode='r')
    return a_memmap


def pair_files(folder_dict, full_path=True, by_prefix=True):
    """Pair files in different directories by their filenames.

    The function returns a dictionary where the keys are the matched
    part of the filenames and the values are dictonaries. The keys of
    each of these dictionaries coincide with the keys in
    `folder_dict`. The value for a given key is a set of paired files
    in the corresponding folder.

    Parameters
    ----------
    folder_dict : dict
        Dictionary with arbitrary labels as keys and directory paths
        as values.
    full_path : bool, optional
        When True, return the full paths of the files. Otherwise only
        filenames are returned, omitting the directories.Defaults to
        True.
    by_prefix : bool, optional
        When True two files in different directories are paired
        whenever one filename (excluding the extension) is a prefix of
        the other. Otherwise files are only paired when the filenames
        excluding the extensions are equal. Defaults to True.

    Returns
    -------
    dict
        A dictionary with the paired files..

    """
    result = defaultdict(lambda: defaultdict(set))

    for label, directory in folder_dict.items():
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                name = os.path.splitext(f)[0]
                if full_path:
                    result[name][label].add(path)
                else:
                    result[name][label].add(f)

    if by_prefix and result:

        # sort by length
        snames = sorted(result.keys(), key=lambda x: len(x))
        # sort lexicographically
        snames.sort()
        cur = snames.pop(0)
        merged = set()
        while snames:
            nxt = snames.pop(0)
            if nxt.startswith(cur):
                for k, v in result[nxt].items():
                    if k in result[cur]:
                        result[cur][k].update(v)
                    else:
                        result[cur][k] = v
                merged.add(nxt)
            else:
                cur = nxt

        for n in merged:
            del result[n]

    # remove_incomplete items
    labels = set(folder_dict.keys())
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
