#!/usr/bin/env python

import argparse
import json
import logging
import os
from functools import partialmethod

import numpy as np
import torch
from partitura import save_performance_midi, save_match
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from basismixer.helper.rendering import compute_basis_from_xml
from basismixer.performance_codec import get_performance_codec
from basismixer.predictive_models.train import MultiMSELoss

logging.basicConfig(level=logging.INFO)

from basismixer.predictive_models import (construct_model,
                                          SupervisedTrainer,
                                          FullPredictiveModel)
from basismixer.utils import load_pyc_bz, save_pyc_bz, split_datasets_by_piece, prepare_datasets_for_model, \
    post_process_predictions
from basismixer import make_datasets

LOGGER = logging.getLogger(__name__)

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# def my_basis(part):
#     W = np.array([n.midi_pitch for n in part.notes_tied]).astype(np.float)
#     return W.reshape((-1, 1)), ['my']

basis_features = ['polynomial_pitch_feature', 'duration_feature', 'metrical_strength_feature']

CONFIG = [
    dict(onsetwise=False,
         basis_functions=basis_features,
         parameter_names=['velocity_dev', 'timing', 'articulation_log', 'velocity_trend', 'beat_period_standardized',
                          'beat_period_mean', 'beat_period_std'],# 'velocity_dev','timing', 'articulation_log', 'velocity_trend', 'beat_period_standardized', 'beat_period_mean', 'beat_period_std'],  #['velocity_dev', 'timing', 'articulation_log', 'velocity_trend', 'beat_period_standardized', 'beat_period_mean', 'beat_period_std']
         seq_len=50,
         model=dict(constructor=['basismixer.predictive_models', 'RecurrentModel'],
                    args=dict(recurrent_size=128,
                              n_layers=1,
                              hidden_size=64)),
         train_args=dict(
             optimizer_params=['Adam', dict(lr=1e-4)],
             epochs=20,
             save_freq=1,
             early_stopping=100,
             batch_size=50,
         )
    )
]

def jsonize_dict(input_dict):
    out_dict = dict()
    for k, v in input_dict.items():
        if isinstance(v, np.ndarray):
            out_dict[k] = v.tolist()
        elif isinstance(v, dict):
            out_dict[k] = jsonize_dict(v)
        else:
            out_dict[k] = v
    return out_dict


def render_fold_match(model, pieces, fold):
    import warnings
    predicter = FullPredictiveModel([model], in_names, out_names)
    perf_codec = get_performance_codec(predicter.output_names)
    for piece in pieces:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xml_fn = args.dataset_root_folder + 'xml/' + piece + '.xml' if args.dataset_name == 'magaloff' else args.dataset_root_folder + f'{piece}/xml_score.musicxml'
                basis, part = compute_basis_from_xml(xml_fn, model.input_names)
                onsets = np.array([n.start.t for n in part.notes_tied])
                preds = predicter.predict(basis, onsets)
                post_process_predictions(preds)
                predicted_ppart = perf_codec.decode(part, preds)
                out_folder = args.out_dir + f'/CV_fold_{fold}/'
                os.makedirs(out_folder, exist_ok=True)
                piece = piece.replace('/', '-')
                save_performance_midi(predicted_ppart, out_folder + f'{piece}.mid')
                alignment = [{'label': 'match', 'score_id': sn.id, 'performance_id': pn['id']} for sn, pn in zip(part.notes_tied, predicted_ppart.notes)]
                save_match(alignment, predicted_ppart, part, out_folder + f'{piece}.match')
        except Exception as e:
            print(f"could not render {piece}")
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a Model given a dataset")
    parser.add_argument("dataset_name", choices=["asap", "4x22", "magaloff"], help="Folder with MusicXML files")
    parser.add_argument("dataset_root_folder", help="Root folder of the dataset")
    parser.add_argument("--folds", help="number of folds in CV", default=10)
    parser.add_argument("--cache", help=(
        'Path to pickled datasets file. If specified and the file exists, '
        'and the cached data matches the model specs' #<---todo
        'the `dataset_root_folder` option will be ignored'))
    parser.add_argument("--pieces", help="Text file with valid pieces",
                        default=None)
    parser.add_argument("--model-config", help="Model configuration",
                        default=CONFIG)
    parser.add_argument("--out-dir", help="Output directory",
                        default='/tmp')
    args = parser.parse_args()

    folds = args.folds

    # Load model architecture
    if not isinstance(args.model_config, list):
        model_config = json.load(open(args.model_config))
    else:
        model_config = args.model_config

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    json.dump(model_config,
              open(os.path.join(args.out_dir, 'model_config.json'), 'w'),
              indent=2)

    if args.pieces is not None:
        print('valid_pieces')
        args.pieces = np.loadtxt(args.pieces, dtype=str)

    rng = np.random.RandomState(1984)

    datasets = []
    models = []
    target_idxs = []

    if args.cache and os.path.exists(args.cache):
        LOGGER.info('Loading data from {}'.format(args.cache))
        datasets = load_pyc_bz(args.cache)
    else:
        datasets = make_datasets(model_config,
                                 args.dataset_root_folder,
                                 args.dataset_name)
        if args.cache:
            LOGGER.info('Saving data to {}'.format(args.cache))
            save_pyc_bz(datasets, args.cache)

    for (mdatasets, in_names, out_names), config in zip(datasets, model_config):

        mdatasets = prepare_datasets_for_model(mdatasets, config)
        dataset = ConcatDataset(mdatasets)
        batch_size = config['train_args'].pop('batch_size')

        for fold in range(folds):
            #### Create train and validation data loaders #####
            train_set, test_set = split_datasets_by_piece(dataset.datasets, fold, folds, False)
            train_loader, valid_loader = DataLoader(train_set, batch_size=batch_size), \
                                         DataLoader(test_set, batch_size=batch_size)

            #### Construct Models ####

            model_cfg = config['model'].copy()
            model_cfg['args']['input_names'] = in_names
            model_cfg['args']['input_size'] = len(in_names)
            model_cfg['args']['output_names'] = config['parameter_names']
            model_cfg['args']['output_size'] = len(config['parameter_names'])
            model_cfg['args']['input_type'] = 'onsetwise' if config['onsetwise'] else 'notewise'
            model_name = ('-'.join(out_names) +
                          '-' + ('onsetwise' if config['onsetwise'] else 'notewise'))
            model_out_dir = os.path.join(args.out_dir, model_name)
            if not os.path.exists(model_out_dir):
                os.mkdir(model_out_dir)
            # save model config for later saving model
            json.dump(jsonize_dict(model_cfg),
                      open(os.path.join(model_out_dir, 'config.json'), 'w'),
                      indent=2)
            model = construct_model(model_cfg)

            loss = MultiMSELoss(config['parameter_names'])

            ### Construct the optimizer ####
            optim_name, optim_args = config['train_args']['optimizer_params']
            optim = getattr(torch.optim, optim_name)
            optim = optim(model.parameters(), **optim_args)

            trainer = SupervisedTrainer(model=model,
                                        train_loss=loss,
                                        valid_loss=loss,
                                        train_dataloader=train_loader,
                                        valid_dataloader=valid_loader,
                                        out_dir=model_out_dir,
                                        optimizer=optim,
                                        **config['train_args'])

            trainer.train()

            test_performance_names = set([t.name for t in test_set.datasets])
            test_pieces = [p for p in mdatasets if p.name in test_performance_names]

            render_fold_match(model, test_performance_names, fold)

