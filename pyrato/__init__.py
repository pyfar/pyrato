# -*- coding: utf-8 -*-

"""Top-level package for pyrato."""

__author__ = \
    """Marco Berzborn - Institute for Hearing Technology and Acoustics"""
__email__ = 'marco.berzborn@akustik.rwth-aachen.de'
__version__ = '0.3.0'


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
