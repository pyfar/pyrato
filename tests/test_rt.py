#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for reverberation time related things. """
from pytest import raises

import numpy as np
import numpy.testing as npt

import pyrato as ra
import pyfar as pf
import pytest


@pytest.mark.parametrize(
    'tx', ['T20', 'T30', 'T40', 'T50', 'T60', 'LDT', 'EDT'])
def test_rt_from_edc(tx):
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est = ra.reverberation_time_energy_decay_curve(
        edc_exp, T=tx)
    npt.assert_allclose(RT_est, 1.)


def test_rt_from_edc_error():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)
    T = 'Bla'

    with raises(ValueError, match='is not a valid interval.'):
        ra.reverberation_time_energy_decay_curve(edc_exp, T=T)
