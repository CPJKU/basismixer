"""
Music related utilities
"""
import os

from typing import Union, List
import numpy as np
import torch
from partitura import load_score as load_score_partitura


from partitura.score import Score

from partitura.musicanalysis.performance_codec import get_unique_onset_idxs

from basismixer.utils import load_pyc_bz

from partitura.utils.misc import PathLike, deprecated_alias


@deprecated_alias(score_fn="filename")
def load_score(filename: PathLike) -> Score:
    """
    Load a score format supported by partitura, or a partitura
    ScoreLike object compressed in pickle format. The only difference
    between this method and the one in partitura is that this
    method also allows for loading a score from a compressed
    pickled file (useful when working with large datasets).

    Parameters
    ----------
    filename : PathLike
        Filename of the score to parse, or a file-like object.

    Returns
    -------
    scr: :class:`partitura.score.Score`
        A score instance.
    """

    # Get extension of the file
    extension = os.path.splitext(filename)[-1]

    if extension in (".pyc.bz", ".bz", ".pyc"):
        score = load_pyc_bz(filename)
    else:
        score = load_score_partitura(filename=filename)

    return score


def notewise_to_onsetwise(
    notewise_inputs: Union[np.ndarray, torch.Tensor],
    unique_onset_idxs: List[np.ndarray],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Agregate basis functions per onset

    Parameters
    ----------
    notewise_inputs : np.ndarray or torch.Tensor
        An array of notewise inputs (basis functions). This array
        has a shape (number_of_notes, number_of_basis_functions)
    unique_onset_idxs: list of np.ndarrays
        A list of the indices of the unique onsets. The lenght
        of the list is number_of_unique_onsets

    Returns
    -------
    onsetwise_inputs : np.ndarray or torch.Tensor
        The inputs aggregated by onset. The size of this array
        is (number_of_unique_onsets, number_of_basis_functions),
        and each of the rows correspond to the mean value of all
        of the notes that occur at that onset. This array will be
        a numpy array if `notewise_inputs` is also a numpy array,
        or a torch.Tensor if `notewise_inputs` is a torch.Tensor.
    """
    if isinstance(notewise_inputs, np.ndarray):
        if notewise_inputs.ndim == 1:
            shape = len(unique_onset_idxs)
        else:
            shape = (len(unique_onset_idxs),) + notewise_inputs.shape[1:]
        onsetwise_inputs = np.zeros(shape, dtype=notewise_inputs.dtype)
    elif isinstance(notewise_inputs, torch.Tensor):
        onsetwise_inputs = torch.zeros(
            (len(unique_onset_idxs), notewise_inputs.shape[1]),
            dtype=notewise_inputs.dtype,
        )

    for i, uix in enumerate(unique_onset_idxs):
        try:
            onsetwise_inputs[i] = notewise_inputs[uix].mean(0)
        except TypeError:
            for tn in notewise_inputs.dtype.names:
                onsetwise_inputs[i][tn] = notewise_inputs[uix][tn].mean()
    return onsetwise_inputs


def onsetwise_to_notewise(
    onsetwise_input: Union[np.ndarray, torch.Tensor],
    unique_onset_idxs: List[np.ndarray],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Expand onsetwise predictions for each note

    Parameters
    ----------
    notewise_inputs : np.ndarray or torch.Tensor
        An array of notewise inputs (basis functions). This array
        has a shape (number_of_notes, number_of_basis_functions)
    unique_onset_idxs: list of np.ndarray
        A list of the indices of the unique onsets. The lenght
        of the list is number_of_unique_onsets.

    Returns
    -------
    onsetwise_inputs : np.ndarray or torch.Tensor
        The inputs aggregated by onset. The size of this array
        is (number_of_unique_onsets, number_of_basis_functions),
        and each of the rows correspond to the mean value of all
        of the notes that occur at that onset. This array will be
        a numpy array if `notewise_inputs` is also a numpy array,
        or a torch.Tensor if `notewise_inputs` is a torch.Tensor.
    """
    n_notes = sum([len(uix) for uix in unique_onset_idxs])
    if isinstance(onsetwise_input, np.ndarray):
        if onsetwise_input.ndim == 1:
            shape = n_notes
        else:
            shape = (n_notes,) + onsetwise_input.shape[1:]
        notewise_inputs = np.zeros(shape, dtype=onsetwise_input.dtype)
    elif isinstance(onsetwise_input, torch.Tensor):
        notewise_inputs = torch.zeros(n_notes, dtype=onsetwise_input.dtype)
    for i, uix in enumerate(unique_onset_idxs):
        notewise_inputs[uix] = onsetwise_input[[i]]
    return notewise_inputs
