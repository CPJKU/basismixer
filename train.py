import numpy as np
import os
from basismixer import make_datasets
from helper import init_dataset, data
from basismixer.utils import load_pyc_bz, save_pyc_bz
from helper.predictions import (construct_model, setup_output_directory, train_model,
                                split_datasets, split_datasets_by_piece)
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
                              hidden_size=64)),
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

init_dataset() # download the corpus if necessary; set some variables

# path to the MusicXML and Match files
xmlfolder = os.path.join(data.DATASET_DIR, 'musicxml')
matchfolder = os.path.join(data.DATASET_DIR, 'match')

dataset_fn = os.path.join(data.DATASET_DIR, 'vienna_4x22.pyc.bz')

IGNORE_CACHE = True

if dataset_fn is not None and os.path.exists(dataset_fn) and not IGNORE_CACHE:
    datasets = load_pyc_bz(dataset_fn)
else:
    datasets = make_datasets(model_config,
                             xmlfolder,
                             matchfolder)
    if dataset_fn is not None:
        save_pyc_bz(datasets, dataset_fn)

models = []
test_sets = []
for (dataset, in_names, out_names), config in zip(datasets, model_config):
    # Build model
    model, model_out_dir = construct_model(config, in_names, out_names, out_dir)
    # Split datasets
    train_set, valid_set, test_set = split_datasets_by_piece(dataset)
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
