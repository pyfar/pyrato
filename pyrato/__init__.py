# -*- coding: utf-8 -*-

__author__ = \
    """The pyfar developers"""
__email__ = 'info@pyfar.org'
__version__ = '0.4.0'

from .rap import (
    reverberation_time_linear_regression
)
from .roomacoustics import (
    reverberation_time_energy_decay_curve,
    energy_decay_curve_analytic,
    air_attenuation_coefficient,
)
from .dsp import (
    find_impulse_response_maximum,
    find_impulse_response_start,
    time_shift,
    preprocess_rir,
    estimate_noise_energy,
)
from .edc import (
    schroeder_integration,
    energy_decay_curve_chu,
    energy_decay_curve_chu_lundeby,
    energy_decay_curve_lundeby,
    energy_decay_curve_truncation,
    intersection_time_lundeby,
)

__all__ = [
    'reverberation_time_linear_regression',
    'reverberation_time_energy_decay_curve',
    'schroeder_integration',
    'energy_decay_curve_analytic',
    'air_attenuation_coefficient',
    'find_impulse_response_maximum',
    'find_impulse_response_start',
    'time_shift',
    'preprocess_rir',
    'energy_decay_curve_chu',
    'energy_decay_curve_chu_lundeby',
    'energy_decay_curve_lundeby',
    'energy_decay_curve_truncation',
    'estimate_noise_energy',
    'intersection_time_lundeby',
]
