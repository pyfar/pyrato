"""Module for room acoustics related functions."""
# -*- coding: utf-8 -*-

__author__ = \
    """The pyfar developers"""
__email__ = 'info@pyfar.org'
__version__ = '0.4.1'


from .parametric import (
    air_attenuation_coefficient,
)

from . import edc
from . import parameters
from . import dsp
from . import analytic
from . import parametric

__all__ = [
    'edc',
    'parameters',
    'dsp',
    'analytic',
    'parametric',
    'air_attenuation_coefficient',
]
