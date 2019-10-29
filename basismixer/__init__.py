#!/usr/bin/env python

import pkg_resources

from basismixer.performance_codec import (
    PerformanceCodec,
    OnsetwiseDecompositionDynamicsCodec,
    TimeCodec
)
from basismixer.basisfunctions import make_basis
from basismixer.data import make_dataset

# define a version variable
__version__ = pkg_resources.get_distribution("basismixer").version

# An example basis configuration for didactic purposes  
BASIS_CONFIG_EXAMPLE = pkg_resources.resource_filename("basismixer", 'assets/basis_config_example.json')
