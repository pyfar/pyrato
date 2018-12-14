#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for reverberation time related things. """

import numpy as np
from numpy import array

import numpy.testing as npt

import roomacoustics as ra


def test_rt_from_edc():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = 10**(edc/10)
    RT = 1.
    TX = ['T20', 'T30', 'T40', 'T50', 'T60']
    for T in TX:
        RT_est = ra.reverberation_time_energy_decay_curve(edc_exp, times, T=T)
        npt.assert_allclose(RT_est, RT)
