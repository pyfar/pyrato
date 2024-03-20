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
    RT_est = ra.reverberation_time_linear_regression(
        edc_exp, T=tx)
    npt.assert_allclose(RT_est, 1.)


@pytest.mark.parametrize(
    'tx', ['T20', 'T30', 'T40', 'T50', 'T60', 'LDT', 'EDT'])
def test_rt_from_edc_mulitchannel(tx):
    times = np.linspace(0, 1.5, 2**9)
    Ts = np.array([1, 2, 1.5])
    m = -60
    edc = np.atleast_2d(m/Ts).T @ np.atleast_2d(times)
    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est = ra.reverberation_time_linear_regression(
        edc_exp, T=tx)
    npt.assert_allclose(RT_est, Ts)


@pytest.mark.parametrize(
    'tx', ['T20', 'T30', 'T40', 'T50', 'T60', 'LDT', 'EDT'])
def test_rt_from_edc_mulitchannel_amplitude(tx):
    times = np.linspace(0, 5/2, 2**9)
    Ts = np.array([[1, 2, 1.5], [3, 4, 5]])
    As = np.array([[0, 3, 6], [1, 1, 1]])
    m = -60
    edc = np.zeros((*Ts.shape, times.size))
    for idx in np.ndindex(Ts.shape):
        edc[idx] = As[idx] + m*times/Ts[idx]

    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est, A_est = ra.reverberation_time_linear_regression(
        edc_exp, T=tx, return_intercept=True)
    npt.assert_allclose(RT_est, Ts)
    npt.assert_allclose(A_est, 10**(As/10))


def test_rt_from_edc_error():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)
    T = 'Bla'

    with raises(ValueError, match='is not a valid interval.'):
        ra.reverberation_time_linear_regression(edc_exp, T=T)
