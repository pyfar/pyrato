# -*- coding: utf-8 -*-

"""Top-level package for pyrato."""

__author__ = \
    """Marco Berzborn - Institute for Hearing Technology and Acoustics"""
__email__ = 'marco.berzborn@akustik.rwth-aachen.de'
__version__ = '0.3.2'


from .roomacoustics import (
    reverberation_time_energy_decay_curve,
    schroeder_integration,
    energy_decay_curve_analytic,
    air_attenuation_coefficient,
)
from .dsp import (
    find_impulse_response_maximum,
    find_impulse_response_start,
    filter_fractional_octave_bands,
    time_shift,
    center_frequencies_octaves,
    center_frequencies_third_octaves
)
from .edc import (
    preprocess_rir,
    energy_decay_curve_chu,
    energy_decay_curve_chu_lundeby,
    energy_decay_curve_lundeby,
    energy_decay_curve_truncation,
    estimate_noise_energy,
    intersection_time_lundeby,
)

__all__ = [
    'reverberation_time_energy_decay_curve',
    'schroeder_integration',
    'energy_decay_curve_analytic',
    'air_attenuation_coefficient',
    'find_impulse_response_maximum',
    'find_impulse_response_start',
    'filter_fractional_octave_bands',
    'time_shift',
    'center_frequencies_octaves',
    'center_frequencies_third_octaves',
    'preprocess_rir',
    'energy_decay_curve_chu',
    'energy_decay_curve_chu_lundeby',
    'energy_decay_curve_lundeby',
    'energy_decay_curve_truncation',
    'estimate_noise_energy',
    'intersection_time_lundeby',
]
