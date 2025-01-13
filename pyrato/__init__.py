# -*- coding: utf-8 -*-

__author__ = \
    """The pyfar developers"""
__email__ = 'info@pyfar.org'
__version__ = '0.3.2'


from .parametric import (
    air_attenuation_coefficient,
)

from . import edc
from . import rap
from . import dsp
from . import analytic
from . import parametric

__all__ = [
    'edc',
    'rap',
    'dsp',
    'analytic',
    'parametric',
    'air_attenuation_coefficient'
]
