# -*- coding: utf-8 -*-

__author__ = \
    """The pyfar developers"""
__email__ = 'info@pyfar.org'
__version__ = '0.4.0'


from .parametric import (
    air_attenuation_coefficient,
)

from . import edc
from . import rap
from . import dsp
from . import analytic
from . import parametric

from . import analytic

__all__ = [
    'edc',
    'rap',
    'dsp',
    'analytic',
    'parametric',
    'air_attenuation_coefficient'
]
