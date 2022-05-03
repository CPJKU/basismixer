import numpy as np
import os
#from basismixer import make_datasets
import torch.cuda
from torch.utils.data import ConcatDataset

from basismixer.data_asap import make_datasets
from helper import init_dataset, data
from basismixer.utils import load_pyc_bz, save_pyc_bz
from helper.predictions import (construct_model, setup_output_directory, train_model, split_datasets_by_piece, split_datasets)
import torch.nn as nn

out_dir = setup_output_directory("runs")

basis_features = ['polynomial_pitch_feature',
                  'duration_feature',
                          #'loudness_direction_feature',
                          #'tempo_direction_feature',
                          #'articulation_feature',
                          #'slur_feature',
                          #'fermata_feature',
                          #'grace_feature',
                          #'metrical_feature'
                  ]

basis_functions = ['polynomial_pitch_basis',
                   'duration_basis',
                          #'loudness_direction_basis',
                          #'tempo_direction_basis',
                          #'articulation_basis',
                          #'slur_basis',
                          #'fermata_basis',
                          #'grace_basis',
                          #'metrical_basis'
                   ]

model_config = [
    dict(onsetwise=True,
         basis_functions=basis_features,
         parameter_names=['velocity_trend', 'beat_period_standardized'],
         model=dict(constructor=['basismixer.predictive_models', 'RecurrentModel'],
                    args=dict(recurrent_size=128,
                              n_layers=1,
                              recurrent_unit='LSTM',
                              hidden_size=128)),
         seq_len=50,
         train_args=dict(
             optimizer=['Adam', dict(lr=1e-3)],
             epochs=100,
             save_freq=1,
             early_stopping=100,
             batch_size=100
         )
    )
]

DATASET_DIR = '/home/plassma/Desktop/JKU/CP/basismixer/asap-dataset' if os.getlogin() == 'plassma' else '../asap-dataset'


def my_split_datasets(datasets, train_idx):

    np.random.shuffle(datasets)

    return (ConcatDataset(datasets[0:train_idx]),
            ConcatDataset(datasets[train_idx:]),
            ConcatDataset(datasets[train_idx:]))#todo: validation set!

def MSE_bound(dataset, log=False):
    from partitura.utils import partition
    by_piece = partition(lambda d:d.name, dataset)
    MSEs = []
    for name, pieces in by_piece.items():
        l = min([p.targets.shape[0] for p in pieces])
        targets = np.stack([p.targets[:l] for p in pieces])
        opt = np.expand_dims(targets.mean(axis=0), 0)
        mse = ((opt - targets) ** 2)  # no mean, as we want to keep len and #pieces as weight here
        MSEs.append(mse.flatten())

        if log:
            print(f"MSE for {name.ljust(50)}: {'{:1.4f}'.format(mse.mean())} \t pieces: {len(pieces)}")

    MSEs = np.concatenate(MSEs).mean()

    if log:
        print(f"inherent MSE: {MSEs}")
    return MSEs


dataset_fn = os.path.join(DATASET_DIR, 'asap.pyc.bz')

IGNORE_CACHE = False

if dataset_fn is not None and os.path.exists(dataset_fn) and not IGNORE_CACHE:
    datasets = load_pyc_bz(dataset_fn)
    train_idx = 1151 - 88
else:
    DIR_4x22 = os.path.expanduser('~/.cache/basismixer/OFAI-vienna4x22_rematched-7b60448')
    xmlfolder = os.path.join(DIR_4x22, 'musicxml')
    matchfolder = os.path.join(DIR_4x22, 'match')
    valid_set_params = (model_config, xmlfolder, matchfolder)
    datasets, train_idx = make_datasets(model_config,
                             DATASET_DIR, quirks=True, valid_set_params=valid_set_params)
    if dataset_fn is not None:
        save_pyc_bz(datasets, dataset_fn)

models = []
test_sets = []
for (dataset, in_names, out_names), config in zip(datasets, model_config):

    dataset = dataset[:train_idx]  # cut off 4x22
    train_idx = int(train_idx * 0.8)
    MSE_bound(dataset, True)
    # Build model
    model, model_out_dir = construct_model(config, in_names, out_names, out_dir)
    # Split datasets
    train_set, valid_set, test_set = split_datasets_by_piece(dataset)#my_split_datasets(dataset, train_idx)
    # Train Model
    train_model(model, train_set, valid_set, config, model_out_dir)

    models.append(model)
    test_sets.append(test_set)

from helper.plotting import plot_predictions_and_targets

for model, test_set in zip(models, test_sets):
    basis = test_set.datasets[0].basis
    idx = test_set.datasets[0].idx
    n_basis = test_set.datasets[0].n_basis
    inputs = np.zeros((len(basis), n_basis))
    inputs[:, idx] = basis
    targets = test_set.datasets[0].targets
    preds = model.predict(inputs)

    plot_predictions_and_targets(preds, targets)
