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


def pair_files(folder_dict, full_path=True,
               by_prefix=True):
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


def notewise_to_onsetwise(notewise_inputs, unique_onset_idxs):
    """Agregate basis functions per onset
    """
    onsetwise_inputs = np.zeros((len(unique_onset_idxs),
                                 notewise_inputs.shape[1]),
                                dtype=notewise_inputs.dtype)

    for i, uix in enumerate(unique_onset_idxs):
        onsetwise_inputs[i] = notewise_inputs[uix].mean(0)
    return onsetwise_inputs


def onsetwise_to_notewise(onsetwise_input, unique_onset_idxs):
    """Expand onsetwise predictions for each note
    """
    n_notes = sum([len(uix) for uix in unique_onset_idxs])
    notewise_inputs = np.zeros(n_notes, dtype=onsetwise_input.dtype)
    for i, uix in enumerate(unique_onset_idxs):
        notewise_inputs[uix] = onsetwise_input[[i]]
    return notewise_inputs
