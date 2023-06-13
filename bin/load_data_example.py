#!/usr/bin/env python

import argparse
import json
import logging

import numpy as np

# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset, ConcatDataset

logging.basicConfig(level=logging.INFO)

# from partitura import load_musicxml, load_match
# from partitura.score import expand_grace_notes
# from basismixer.basisfunctions import make_basis
from basismixer.data import make_datasets

# from basismixer.performance_codec import get_performance_codec
# from basismixer.utils import clip, get_unique_onset_idxs
# from basismixer.utils import (pair_files,
#                               get_unique_onset_idxs,
#                               notewise_to_onsetwise,
#                               onsetwise_to_notewise)

from basismixer.utils import save_pyc_bz
from basismixer.utils import pair_files
import sys
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Construct a dataset from files in the specified folders")
    parser.add_argument("xmlfolder", help="Folder with MusicXML files")
    parser.add_argument("matchfolder", help="Folder with match files")
    args = parser.parse_args()

    model_specs = [dict(onsetwise=False,
                        basis_functions=['polynomial_pitch_basis', 'duration_basis'],
                        parameter_names=['velocity_trend', 'beat_period_std'],
                        seq_len=100,
                        ),
                   dict(onsetwise=True,
                        basis_functions=['duration_basis', 'articulation_basis', 'loudness_direction_basis'],
                        parameter_names=['velocity_dev', 'timing'],
                        seq_len=100,
                        )
                   ]

    datasets = make_datasets(model_specs, args.xmlfolder, args.matchfolder)
    for dataset, in_n, out_n in datasets:
        x, y = dataset[0]
        print(in_n, len(in_n), x.shape)
        print(out_n, len(out_n), y.shape)
        print()

if __name__ == '__main__':
    main()
