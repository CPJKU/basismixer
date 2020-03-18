#!/usr/bin/env python
"""
The top level of the package contains functions to extract score information
with basis functions and create performance codecs
"""

import pkg_resources

from basismixer.performance_codec import (
    PerformanceCodec,
    OnsetwiseDecompositionDynamicsCodec,
    TimeCodec,
)
from basismixer.basisfunctions import make_basis
# from basismixer.data import make_dataset
from basismixer.data import make_datasets

# define a version variable
__version__ = pkg_resources.get_distribution("basismixer").version

# An example basis configuration for didactic purposes
BASIS_CONFIG_EXAMPLE = pkg_resources.resource_filename("basismixer", 'assets/basis_config_example.json')

MODEL_CONFIG_EXAMPLE = pkg_resources.resource_filename("basismixer", 'assets/model_config_example.json')

# A small trained model for didactic purposes
TRAINED_MODEL_CONFIG_EXAMPLE = pkg_resources.resource_filename("basismixer", 'assets/sample_models/vienna_4x22_velocity_trend-beat_period_standardized-onsetwise/config.json')
TRAINED_MODEL_PARAMS_EXAMPLE = pkg_resources.resource_filename("basismixer", 'assets/sample_models/vienna_4x22_velocity_trend-beat_period_standardized-onsetwise/model_params.pth')

TOY_MODEL_CONFIG = [[TRAINED_MODEL_CONFIG_EXAMPLE,
                     TRAINED_MODEL_PARAMS_EXAMPLE]]
