"""
Music related utilities
"""
import numpy as np

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
