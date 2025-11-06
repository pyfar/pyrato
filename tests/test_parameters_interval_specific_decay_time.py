#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for reverberation time related things."""
import numpy as np
import numpy.testing as npt

import pyrato as ra
import pyfar as pf
import pytest


@pytest.mark.parametrize(
    'intvl', [[10,100],[100, 800],[200,1100]])
def test_isdt_from_edc(intvl):
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est = ra.parameters.interval_specific_decay_time(edc_exp,
                                            time_0=intvl[0], time_1=intvl[1])
    npt.assert_allclose(RT_est, 1.)


@pytest.mark.parametrize(
    'intvl', [[10,100],[100, 800],[200,1100]])
def test_isdt_from_edc_mulitchannel(intvl):
    times = np.linspace(0, 1.5, 2**9)
    Ts = np.array([1, 2, 1.5])
    m = -60
    edc = np.atleast_2d(m/Ts).T @ np.atleast_2d(times)
    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est = ra.parameters.interval_specific_decay_time(edc_exp,
                                            time_0=intvl[0], time_1=intvl[1])
    npt.assert_allclose(RT_est, Ts)



def test_isdt_from_edc_error():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)

    with pytest.raises(ValueError,
                       match='time_0 must be smaller than time_1.'):
        ra.parameters.interval_specific_decay_time(edc_exp,
                                                time_0=100, time_1=10)

    with pytest.raises(ValueError,
                       match='time_0 and time_1 must be in the range of'):
        ra.parameters.interval_specific_decay_time(edc_exp,
                                                time_0=10, time_1=1.6*1000)
