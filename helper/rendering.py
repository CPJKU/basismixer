import json
import os

import torch
import numpy as np
import subprocess
import soundfile
import tempfile
import logging

from IPython.display import display, Audio

from partitura import save_performance_midi, load_musicxml
from partitura.score import expand_grace_notes, unfold_part_maximal
from partitura.musicanalysis import make_note_feats
from basismixer.predictive_models import FullPredictiveModel, construct_model
from basismixer.performance_codec import get_performance_codec
from basismixer.basisfunctions import make_basis

from helper.predictions import setup_output_directory

LOGGER = logging.getLogger(__name__)

def path_to_trained_models(path=setup_output_directory()):
    if not os.path.exists(path):
        print('Models not found! Using sample models')
        path = './sample_data/models'
    return path
    
    
def render_midi(midi_fn):

    with tempfile.NamedTemporaryFile() as out_file:
        cmd = ['timidity', '-E', 'F', 'reverb=0', 'F', 'chorus=0',
               '--output-mono', '-Ov', '-o', out_file.name, midi_fn]
        try:
            ps = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ps.returncode != 0:
                LOGGER.error('Command {} failed with code {} (stderr: {})'
                             .format(cmd, ps.returncode, ps.stderr.decode('UTF8')))
                return False
        except FileNotFoundError as f:
            LOGGER.error('Executing "{}" returned  {}.'
                         .format(' '.join(cmd), f))
            return False
        data, fs = soundfile.read(out_file.name)
        aw = display(Audio(data=data, rate=fs, autoplay=True), display_id=True)
        return aw

    
def load_model(models_dir):
    models = []
    for f in os.listdir(models_dir):
        path = os.path.join(models_dir, f)
        if os.path.isdir(path):
            model_config = json.load(open(os.path.join(path, 'config.json')))
            params = torch.load(os.path.join(path, 'best_model.pth'), 
                                map_location=torch.device('cpu'))['state_dict']
        
            model = construct_model(model_config, params)
            models.append(model)
    
    
    
    output_names = list(set([name for out_name in [m.output_names for m in models] for name in out_name]))
    input_names = list(set([name for in_name in [m.input_names for m in models] for name in in_name]))
    input_names.sort()
    output_names.sort()

    default_values = dict(
        velocity_trend=64,
        velocity_dev=0,
        beat_period_standardized=0,
        timing=0,
        articulation_log=0,
        beat_period_mean=0.5,
        beat_period_std=0.1)
    all_output_names = list(default_values.keys())
    full_model = FullPredictiveModel(models, input_names,
                                     all_output_names, default_values)

    not_in_model_names = set(all_output_names).difference(output_names)
    
    print('Trained models include the following parameters:\n'
          + '\n'.join(output_names) + '\n\n'
          'The following parameters will use default values:\n'+
          '\n' + '\n'.join(['{0}:{1:.2f}'.format(k, default_values[k])
                            for k in not_in_model_names]))

    
    return full_model, output_names

def sanitize_performed_part(ppart):
    """Avoid negative durations in notes.

    """
    for n in ppart.notes:

        if n['note_off'] < n['note_on']:
            n['note_off'] = n['note_on']

        if n['sound_off'] < n['note_off']:
            n['sound_off'] = n['note_off']


def post_process_predictions(predictions):
    max_articulation = 1.5
    max_bps = 1
    max_timing = 0.2
    predictions['articulation_log'] = np.clip(predictions['articulation_log'],
                                              -max_articulation, max_articulation)
    predictions['velocity_dev'] = np.clip(predictions['velocity_dev'], 0, 0.8)
    predictions['beat_period_standardized'] = np.clip(predictions['beat_period_standardized'],
                                                      -max_bps, max_bps)
    predictions['timing'] = np.clip(predictions['timing'],
                                    -max_timing, max_timing)
    predictions['velocity_trend'][predictions['velocity_trend'] > 0.8] = 0.8
    



def compute_basis_from_xml(xml_fn, input_names):
    # Load MusicXML file
    part = load_musicxml(xml_fn, force_note_ids=True)
    expand_grace_notes(part)
    part = unfold_part_maximal(part)

    # Compute basis functions
    #_basis, bf_names = make_basis(part, list(set([bf.split('.')[0] for bf in input_names])))
    _basis, bf_names = make_note_feats(part, list(set([bf.split('.')[0] for bf in input_names])))
    basis = np.zeros((len(_basis), len(input_names)))
    for i, n in enumerate(input_names):
        try:
            ix = bf_names.index(n)
        except ValueError:
            continue
        basis[:, i] = _basis[:, ix]

    return basis, part

