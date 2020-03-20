"""
Utilities for rendering performances
"""
import numpy as np

DEFAULT_BPM = 108

DEFAULT_VALUES = dict(
    velocity_trend=55,
    velocity_dev=0,
    velocity=55,
    beat_period=60/DEFAULT_BPM,
    beat_period_log=np.log2(60/DEFAULT_BPM),
    beat_period_ratio=1,
    beat_period_ratio_log=0,
    beat_period_standardized=0,
    beat_period_mean=60/DEFAULT_BPM,
    beat_period_std=0.1,
    timing=0,
    articulation_log=0
)

RENDER_CONFIG = dict(
    velocity_trend=dict(
        normalization=None,
        velocity_trend_max=55,
        velocity_trend_min=5),
    velocity_dev=dict(
        normalization=None,
        velocity_dev_max=5,
        velocity_dev_min=3),
    velocity=dict(
        normalization=None,
        max=108,
        min=20),
    beat_period=dict(
        normalization=None,
        max=3,
        min=60 / 200),
    
)

def sanitize_performed_part(ppart):
    """Avoid negative durations in notes.

    """
    for n in ppart.notes:

        if n['note_off'] < n['note_on']:
            n['note_off'] = n['note_on']

        if n['sound_off'] < n['note_off']:
            n['sound_off'] = n['note_off']

def post_process_predictions(predictions, render_config=RENDER_CONFIG):
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
    predictions['velocity_trend'] *= 0.7


def get_default_values(default_values, names):
    """
    Get dictionary of default values
    """
    def_values = dict()
    for name in names:
        def_values[name] = default_values.get(name, DEFAULT_VALUES[name])
    return def_values

def get_all_output_names(names, error_on_conflicting_names=False):
    """Get all output names from list of output names of the models
    """

    tempo_options = [('beat_period', ),
                     ('beat_period_log', ),
                     ('beat_period_ratio', 'beat_period_mean'),
                     ('beat_period_ratio_log', 'beat_period_mean'),
                     ('beat_period_standardized', 'beat_period_mean', 'beat_period_std')]

    # Use the simplest parameters
    dynamics_params = ('velocity', )
    tempo_params = ('beat_period', )
    
    # Give priority to models with onset-wise parameters
    if 'velocity_trend' in names or 'velocity_dev' in names:
        dynamics_params = ('velocity_trend', 'velocity_dev')
        if 'velocity' in names:
            if error_on_conflicting_names:
                raise ValueError('Invalid combination of dynamics parameters')

    # overlap between names and options as the cardinality of the intersection 
    tp_overlap = np.array([len(set(option).intersection(names)) for option in tempo_options])

    # if no tempo param is predicted, choose the simplest parameter
    if np.all(tp_overlap == 0):
        tempo_params = tempo_options[0]

    else:
        best_option = np.where(tp_overlap == tp_overlap.max())[0]

        if len(best_option) > 1:
            if error_on_conflicting_names:
                raise ValueError('Invalid combination of tempo parameters')

            # if 'beat_period_mean' in names and 'beat_period_std' not in names:
            #     tempo_params = tempo_options[2]
            
        tempo_params = tempo_options[best_option[0]]

    params = dynamics_params + tempo_params + ('timing', 'articulation_log')

    return params
