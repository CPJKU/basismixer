#!/usr/bin/env python

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

logging.basicConfig(level=logging.INFO)

from basismixer.predictive_models import construct_model, SupervisedTrainer, MSELoss
from basismixer.utils import load_pyc_bz, save_pyc_bz, split_datasets_by_piece
from basismixer import make_datasets

LOGGER = logging.getLogger(__name__)

# def my_basis(part):
#     W = np.array([n.midi_pitch for n in part.notes_tied]).astype(np.float)
#     return W.reshape((-1, 1)), ['my']

basis_features = [
    "polynomial_pitch_feature",
    "duration_feature",
    "metrical_strength_feature",
]

CONFIG = [
    dict(
        onsetwise=False,
        basis_functions=[
            "polynomial_pitch_feature",
            # "loudness_direction_feature",
            # "tempo_direction_feature",
            "articulation_feature",
            "duration_feature",
            # my_basis,
            # "grace_feature",
            # "slur_feature",
            "fermata_feature",
            # 'metrical_feature'
            "metrical_strength_feature",
            "time_signature_feature",
            "relative_score_position_feature",
        ],
        parameter_names=["velocity_dev", "timing", "articulation_log"],
        seq_len=1,
        model=dict(
            constructor=["basismixer.predictive_models", "FeedForwardModel"],
            args=dict(hidden_size=128),
        ),
        train_args=dict(
            optimizer=["Adam", dict(lr=1e-4)],
            epochs=100,
            save_freq=10,
            early_stopping=10,
            batch_size=1000,
        ),
    ),
    dict(
        onsetwise=True,
        basis_functions=[
            "polynomial_pitch_feature",
            # "loudness_direction_feature",
            # "tempo_direction_feature",
            # "articulation_feature",
            "duration_feature",
            # "slur_feature",
            # "grace_feature",
            "fermata_feature",
            # 'metrical_feature'
            "metrical_strength_feature",
            "time_signature_feature",
            "relative_score_position_feature",
        ],
        parameter_names=[
            "velocity_trend",
            "beat_period_ratio_log",
            # "beat_period_mean",
            # "beat_period_std",
        ],
        seq_len=100,
        model=dict(
            constructor=["basismixer.predictive_models", "RecurrentModel"],
            args=dict(recurrent_size=128, n_layers=1, hidden_size=64),
        ),
        train_args=dict(
            optimizer=["Adam", dict(lr=1e-4)],
            epochs=10,
            save_freq=5,
            early_stopping=10,
            batch_size=50,
        ),
    ),
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


def write_executable_render(rendering_config, out_dir):

    rendering_config_path = os.path.join(out_dir, "config_for_rendering.json")
    # Write rendering config
    json.dump(
        rendering_config,
        open(rendering_config_path, "w"),
    )

    script_lines = [
        "#!/bin/bash\n",
        "# Path to the script",
        'scd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n',
        "# Set score_fn variable",
        'score_fn="$1"\n',
        "# Set midi_fn variable",
        'midi_fn="$2"',
        'if [ -z "$midi_fn" ]; then',
        '\tmidi_fn="${scd}/$(basename ${score_fn}).mid"',
        "fi\n",
        f'model_config="{os.path.abspath(rendering_config_path)}"\n',
        'BasisMixerRender "${score_fn}" "${midi_fn}" \\',
        '\t--model-config "${model_config}"',
    ]

    script = "\n".join(script_lines)

    script_fn = os.path.join(out_dir, "rendering_script.sh")
    with open(script_fn, "w") as f:
        f.write(script)

    # make script executable
    os.chmod(script_fn, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Model given a dataset")
    parser.add_argument(
        "dataset_name",
        choices=["asap", "4x22", "magaloff"],
        help="Folder with MusicXML files",
    )
    parser.add_argument("dataset_root_folder", help="Root folder of the dataset")
    parser.add_argument(
        "--cache",
        help=(
            "Path to pickled datasets file. If specified and the file exists, "
            "and the cached data matches the model specs"  # <---todo
            "the `dataset_root_folder` option will be ignored"
        ),
    )
    parser.add_argument(
        "--pieces",
        help="Text file with valid pieces",
        default=None,
    )
    parser.add_argument(
        "--model-config",
        help="Model configuration",
        default=CONFIG,
    )
    parser.add_argument("--out-dir", help="Output directory", default="/tmp")
    args = parser.parse_args()

    # Load model architecture
    if not isinstance(args.model_config, list):
        model_config = json.load(open(args.model_config))
    else:
        model_config = args.model_config

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    json.dump(
        model_config,
        open(os.path.join(args.out_dir, "model_config.json"), "w"),
        indent=2,
    )

    if args.pieces is not None:
        print("valid_pieces")
        args.pieces = np.loadtxt(args.pieces, dtype=str)

    rng = np.random.RandomState(1984)

    datasets = []
    models = []
    target_idxs = []

    if args.cache and os.path.exists(args.cache):
        LOGGER.info("Loading data from {}".format(args.cache))
        datasets = load_pyc_bz(args.cache)
    else:
        datasets = make_datasets(
            model_config,
            args.dataset_root_folder,
            args.dataset_name,
            valid_pieces=args.pieces,
        )
        if args.cache:
            LOGGER.info("Saving data to {}".format(args.cache))
            save_pyc_bz(datasets, args.cache)

    rendering_config = []
    for (mdatasets, in_names, out_names), config in zip(datasets, model_config):
        dataset = ConcatDataset(mdatasets)
        batch_size = config["train_args"].pop("batch_size")

        #### Create train and validation data loaders #####
        train_set, test_set = split_datasets_by_piece(mdatasets, 0, 5, False)
        train_loader, valid_loader = DataLoader(
            train_set, batch_size=batch_size
        ), DataLoader(test_set, batch_size=batch_size)

        #### Construct Models ####

        model_cfg = config["model"].copy()
        model_cfg["args"]["input_names"] = in_names
        model_cfg["args"]["input_size"] = len(in_names)
        model_cfg["args"]["output_names"] = out_names
        model_cfg["args"]["output_size"] = len(out_names)
        model_cfg["args"]["input_type"] = (
            "onsetwise" if config["onsetwise"] else "notewise"
        )
        model_name = (
            "-".join(out_names)
            + "-"
            + ("onsetwise" if config["onsetwise"] else "notewise")
        )
        model_out_dir = os.path.join(args.out_dir, model_name)
        if not os.path.exists(model_out_dir):
            os.mkdir(model_out_dir)
        # save model config for later saving model
        json.dump(
            jsonize_dict(model_cfg),
            open(os.path.join(model_out_dir, "config.json"), "w"),
            indent=2,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = construct_model(model_cfg, device=device)

        loss = MSELoss()

        ### Construct the optimizer ####
        optim_name, optim_args = config["train_args"]["optimizer"]
        optim = getattr(torch.optim, optim_name)
        config["train_args"]["optimizer"] = optim(
            model.parameters(),
            **optim_args,
        )

        trainer = SupervisedTrainer(
            model=model,
            train_loss=loss,
            valid_loss=loss,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            out_dir=model_out_dir,
            **config["train_args"],
        )

        trainer.train()

        rendering_config.append(
            [
                os.path.join(model_out_dir, "config.json"),
                os.path.join(model_out_dir, "best_model.pth"),
            ]
        )

    write_executable_render(rendering_config, args.out_dir)
