import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
# from torch.utils.data.sampler import SubsetRandomSampler

from partitura.utils import partition
from basismixer.predictive_models import (construct_model as c_model,
                                          SupervisedTrainer,
                                          MSELoss)
from basismixer.utils import load_pyc_bz, save_pyc_bz

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

RNG = np.random.RandomState(1984)

def construct_model(config, in_names, out_names, out_dir):
    model_cfg = config['model'].copy()
    model_cfg['args']['input_names'] = in_names
    model_cfg['args']['input_size'] = len(in_names)
    model_cfg['args']['output_names'] = out_names
    model_cfg['args']['output_size'] = len(out_names)
    model_cfg['args']['input_type'] = 'onsetwise' if config['onsetwise'] else 'notewise'
    model_name = ('-'.join(out_names) +
                  '-' + ('onsetwise' if config['onsetwise'] else 'notewise'))
    model_out_dir = os.path.join(out_dir, model_name)
    if not os.path.exists(model_out_dir):
        os.mkdir(model_out_dir)
    # save model config for later saving model
    config_out = os.path.join(model_out_dir, 'config.json')
    LOGGER.info('Saving config in {0}'.format(config_out))
    json.dump(jsonize_dict(model_cfg),
              open(config_out, 'w'),
              indent=2)
    model = c_model(model_cfg)

    return model, model_out_dir

def setup_output_directory(out_dir='/tmp/trained_models'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir

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

def split_datasets_by_piece(datasets, test_size=0.2, valid_size=0.2):

    by_piece = partition(lambda d: '_'.join(d.name.split('_')[:-1]), datasets)
    pieces = list(by_piece.keys())
    RNG.shuffle(pieces)

    n_test = max(1, int(np.round(test_size*len(pieces))))
    n_valid = max(1, int(np.round(valid_size*len(pieces))))
    n_train = len(pieces) - n_test - n_valid

    if n_train < 1:
        raise Exception('Not enough pieces to split datasets according '
                        'to the specified test/validation proportions')

    test_pieces = pieces[:n_test]
    valid_pieces = pieces[n_test:n_test+n_valid]
    train_pieces = pieces[n_test+n_valid:]

    test_set = [d for pd in [by_piece[p] for p in test_pieces] for d in pd]
    valid_set = [d for pd in [by_piece[p] for p in valid_pieces] for d in pd]
    train_set = [d for pd in [by_piece[p] for p in train_pieces] for d in pd]

    return (ConcatDataset(train_set),
            ConcatDataset(valid_set),
            ConcatDataset(test_set))


def split_datasets(datasets, test_size=0.2, valid_size=0.2):

    n_pieces = len(datasets)

    dataset_idx = np.arange(n_pieces)
    RNG.shuffle(dataset_idx)
    len_test = int(n_pieces * test_size)
    len_valid = np.maximum(int((n_pieces - len_test) * valid_size), 1)

    test_idxs = dataset_idx[:len_test]
    valid_idxs = dataset_idx[len_test:len_test + len_valid]
    train_idxs = dataset_idx[len_test + len_valid:]

    return (ConcatDataset([datasets[i] for i in train_idxs]),
            ConcatDataset([datasets[i] for i in valid_idxs]),
            ConcatDataset([datasets[i] for i in test_idxs]))




def train_model(model, train_set, valid_set,
                config, out_dir):
    batch_size = config['train_args'].pop('batch_size')

    #### Create train and validation data loaders #####
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_set,
                              batch_size=batch_size,
                              shuffle=False)

    loss = MSELoss()

    ### Construct the optimizer ####
    optim_name, optim_args = config['train_args']['optimizer']
    optim = getattr(torch.optim, optim_name)
    config['train_args']['optimizer'] = optim(model.parameters(), **optim_args)
    train_args = config['train_args']
    train_args.pop('seq_len', None)
    trainer = SupervisedTrainer(model=model,
                                train_loss=loss,
                                valid_loss=loss,
                                train_dataloader=train_loader,
                                valid_dataloader=valid_loader,
                                out_dir=out_dir,
                                **config['train_args'])
    trainer.train()
