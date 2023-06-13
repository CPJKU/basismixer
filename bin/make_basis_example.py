#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt
import numpy as np
import partitura
import partitura.musicanalysis as ma
import partitura.score


def main():
    parser = argparse.ArgumentParser(
        description="Create basis functions for a MusicXML file"
    )
    parser.add_argument("musicxml", help="MusicXML file")
    parser.add_argument(
        "--basis",
        type=str,
        nargs="+",
        help="names of one or more basis features",
    )
    # parser.add_argument("--cachefolder", type=str, help='Cache folder')
    # parser.add_argument("--basisconfig", type=str,
    #                     help=("JSON file specifying a set of basis functions for each expressive target. "
    #                           "If not specified a default configuration will be used."))
    args = parser.parse_args()
    # bcfg = args.basisconfig or basismixer.BASIS_CONFIG_EXAMPLE
    # basis_config = json.load(open(bcfg))
    # basis_names = list(set(i for ii in basis_config.values() for i in ii))

    part = partitura.load_musicxml(args.musicxml)
    part = partitura.score.merge_parts(part)
    part = partitura.score.unfold_part_maximal(part, update_ids=False)
    print(part.pretty())
    basis, names = ma.make_note_feats(part, args.basis)
    # plot
    onsets = None  # np.array([n.start.t for n in part.notes_tied])
    plot_basis(basis, names, "/tmp/out.png", onsets, title=part.part_name)


def plot_basis(basis, names, out_fn, onsets=None, title=None):
    n_basis = basis.shape[1]

    if onsets is None:
        x = np.arange(len(basis))
    else:
        x = onsets

    w = len(x) / 30
    h = n_basis

    fig, axs = plt.subplots(
        n_basis, sharex=True, sharey=True, gridspec_kw={"hspace": 0}
    )
    if n_basis == 1:
        axs = [axs]

    fig.set_size_inches(w, h)

    if title:
        fig.suptitle(title)

    for i, name in enumerate(names):
        axs[i].fill_between(x, 0, basis[:, i], label=name)
        axs[i].legend(frameon=False)

    fig.tight_layout()

    if title:
        fig.subplots_adjust(top=0.95)

    fig.savefig(out_fn)


#     idx_map = {}
#     basis_data = []
#     print(basis_config)

#     for musicxml in args.musicxml:

#         part = partitura.load_musicxml(musicxml)

#         if not isinstance(part, partitura.score.Part):
#             print('No score parts found for {}'.format(musicxml))

#         basis, names = bf.make_basis(part, basis_names)
#         # to move the array to disk and load as a memmap, do:
#         # basis = to_memmap(basis, args.cachefolder)

#         # plot
#         onsets = np.array([n.start.t for n in part.notes_tied])
#         plot_basis(basis, names, '/tmp/out.png', onsets, title=part.part_name)

#         idx = np.array([idx_map.setdefault(name, len(idx_map))
#                         for i, name in enumerate(names)])

#         basis_data.append((basis, idx))


#     N = len(idx_map)

#     _datasets = []
#     for basis, idx in basis_data:
#         _datasets.append(BasisMixerDataSet(basis, idx, N))

#     dataset = torch.utils.data.ConcatDataset(_datasets)

#     # map names to index:
#     # inv_idx_map = dict((v,k) for k, v in idx_map.items())


# class BasisMixerDataSet(torch.utils.data.Dataset):
#     def __init__(self, basis, idx, n_basis):
#         self.basis = basis
#         self.idx = idx
#         self.n_basis = n_basis

#     def __getitem__(self, i):
#         v = np.zeros(self.n_basis)
#         v[self.idx] = self.basis[i]
#         return v

#     def __len__(self):
#         return len(self.basis)


if __name__ == "__main__":
    main()
