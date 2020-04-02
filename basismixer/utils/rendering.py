"""
Utilities for rendering performances
"""
import numpy as np

from .generic import clip

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
        # normalization=dict(type='standardize',
        #                    mean=55 / 127,
        #                    std=10 / 127),
        normalization=None,
        max=108 / 127,
        min=15 / 127),
    velocity_dev=dict(
        # normalization=dict(type='standardize',
        #                    mean=5 / 127,
        #                    std=3 / 127),
        normalization=None,
        max=20 / 127,
        min=0 / 127),
    velocity=dict(
        normalization=None,
        max=108 / 127,
        min=20 / 127),
    beat_period=dict(
        # normalization=dict(type='standardize',
        #                    mean=60 / 72,
        #                    std=0.1),
        normalization=None,
        max=3,
        min=60 / 200),
    beat_period_log=dict(
        normalization=None,
        max=np.log2(3),
        min=np.log2(60 / 200)),
    beat_period_ratio=dict(
        normalization=None,
        max=3,
        min=1/3),
    beat_period_standardized=dict(
        # normalization=dict(type='standardize',
        #                    mean=0,
        #                    std=0.5),
        normalization=None,
        max=3,
        min=-3),
    beat_period_mean=dict(
        # normalization=dict(type='constant',
        #                    value=1.5),
        normalization=None,
        max=3,
        min=60 / 200),
    beat_period_std=dict(
        # normalization=dict(type='minmax',
        #                    highest=1,
        #                    lowest=0.1),
        normalization=None,
        max=2.5,
        min=0),
    timing=dict(
        # normalization=dict(type='minmax',
        #                    highest=0.02,
        #                    lowest=0.02),
        normalization=None,
        max=0.05,
        min=-0.05),
    articulation_log=dict(
        # normalization=dict(type='standardize',
        #                    mean=0,
        #                    std=0.3),
        normalization=None,
        max=1.5,
        min=-1.25)
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

    for pn in predictions.dtype.names:
        config = render_config.get(pn, RENDER_CONFIG[pn])
        predictions[pn] = normalize_predictions(predictions[pn], config['normalization'])
        clip(predictions[pn], low=config['min'], high=config['max'])
        print(pn, predictions[pn].min(), predictions[pn].mean(), predictions[pn].max())

def normalize_predictions(predictions, normalization_config):
    if normalization_config is None:
        return predictions
    else:
        norm_func = globals()[normalization_config.pop('type')]

        return norm_func(predictions, **normalization_config)
        

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

def standardize(x, mean, std):
    x_mean = x.mean()
    x_std = max(x.std(), 1e-6)
    # standardize
    x_hat = (x - x_mean) / x_std
    return (std * x_hat) + mean

def minmax(x, lowest, highest):
    x_max = x.max()
    x_min  = x.min()

    if np.isclose(x_max, x_min):
        x_hat = np.zeros_like(x)
    else:
        x_hat = (x - x_min) / (x_max - x_min)

    return (highest - lowest) * (x_hat) + lowest

def scaled_tanh(x, scale):
    return scale * np.tanh(x)

def constant(x, value):
    return np.ones_like(x) * value
