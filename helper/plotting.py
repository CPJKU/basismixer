#!/usr/bin/env python

import numpy as np
# import argparse
import matplotlib.pyplot as plt
from helper.visualization import make_plot

def plot_basis(basis, names, onsets=None, title=None):
    n_basis = basis.shape[1]

    if onsets is None:
        x = np.arange(len(basis))
    else:
        x = onsets

    w = len(x)/30
    h = n_basis

    fig, axs = plt.subplots(n_basis, sharex=True,
                            gridspec_kw={'hspace': 0})
    if n_basis == 1:
        axs = [axs]
    
    fig.set_size_inches(w, h)

    if title:
        fig.suptitle(title)
        
    for i, name in enumerate(names):
        axs[i].fill_between(x, 0, basis[:, i], label=name)
        axs[i].legend(frameon=False, loc='upper left')

    fig.tight_layout()

    if title:
        fig.subplots_adjust(top=0.95)

    # fig.savefig(out_fn)

def plot_predictions_and_targets(predictions, targets):
    param_names = predictions.dtype.names
    n_params = len(param_names)
    fig, axs = plt.subplots(n_params, sharex=True)

    fig.set_size_inches(len(predictions) / 30, n_params)
    for i, pn in enumerate(param_names):
        axs[i].plot(predictions[pn], color='firebrick',
                    label='predictions')
        if targets is not None:
            axs[i].plot(targets[:, i], color='blue', label='targets')
        axs[i].set_title(pn)
        axs[i].legend(frameon=False, loc='upper left')

    fig.tight_layout()


def plot_predictions(predictions, onsets=None,
                     param_names=None):

    if param_names is None:
        param_names = predictions.dtype.names
    fig, axs = plt.subplots(len(param_names),
                            sharex=True,
                            gridspec_kw={'hspace': 0.15})
    plt.subplots_adjust(left=0.07, right=0.99, top=.99, bottom=0.1)

    make_plot(fig, axs, predictions[param_names], onsets)
    
   


# def main():
#     parser = argparse.ArgumentParser(description="Do something")
#     parser.add_argument("file", help="some file")
#     args = parser.parse_args()


# if __name__ == '__main__':
#     main()
