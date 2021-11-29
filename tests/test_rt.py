#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for reverberation time related things. """
from pytest import raises

import numpy as np
import numpy.testing as npt

import pyrato as ra


def test_rt_from_edc():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = 10**(edc/10)
    RT = 1.
    TX = ['T20', 'T30', 'T40', 'T50', 'T60', 'LDT', 'EDT']
    for T in TX:
        RT_est = ra.reverberation_time_energy_decay_curve(edc_exp, times, T=T)
        npt.assert_allclose(RT_est, RT)


def test_rt_from_edc_error():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = 10**(edc/10)
    T = 'Bla'

    with raises(ValueError, match='is not a valid interval.'):
        ra.reverberation_time_energy_decay_curve(edc_exp, times, T=T)
    # npt.assert_allclose(RT_est, RT)
