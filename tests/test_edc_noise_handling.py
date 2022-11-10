#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The test_edc_noise_handling module provides the functionality to test
    the functions of the module edc_noise_handling.
"""

import numpy as np
import os
import numpy.testing as npt
from pyrato import edc as enh
from numpy import genfromtxt
import pyfar as pf
test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')


def mock_shift_samples_1d(*args, **kwargs):
    return np.array([76])


def mock_shift_samples_2d(*args, **kwargs):
    return np.array([76, 76])


def test_substracted_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'substracted_1D.csv'),
        delimiter=',')
    actual = enh._subtract_noise_from_squared_rir(rir**2)
    npt.assert_allclose(actual, expected)


def test_substracted_2D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'substracted_2D.csv'),
        delimiter=',')
    actual = enh._subtract_noise_from_squared_rir(rir**2)
    npt.assert_allclose(actual, expected)


def test_edc_truncation_1D():
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_truncation_1D.csv'),
        delimiter=','))

    actual = enh.energy_decay_curve_truncation(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=-np.inf)

    pf.plot.time(actual, dB=True, log_prefix=10)

    npt.assert_allclose(actual.time, expected)

    actual = enh.energy_decay_curve_truncation(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=15)

    mask = expected < 10**((-40+15)/10)
    expected[mask] = np.nan

    pf.plot.time(actual, dB=True, log_prefix=10)

    npt.assert_allclose(actual.time, expected)


def test_edc_truncation_2D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_truncation_2D.csv'),
        delimiter=','))

    actual = enh.energy_decay_curve_truncation(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=-np.inf)
    npt.assert_allclose(actual.time, expected)


def test_edc_lundeby_1D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_1D.csv'),
        delimiter=','))

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.energy_decay_curve_lundeby(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual.time, expected)


def test_edc_lundeby_2D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_2D.csv'),
        delimiter=','))

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.energy_decay_curve_lundeby(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual.time, expected)


def test_edc_lundeby_chu_1D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_chu_1D.csv'),
        delimiter=','))

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.energy_decay_curve_chu_lundeby(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual.time, expected)


def test_edc_lundeby_chu_2D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_chu_2D.csv'),
        delimiter=','))

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.energy_decay_curve_chu_lundeby(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual.time, expected)


def test_edc_chu_1D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_chu_1D.csv'),
        delimiter=','))

    actual = enh.energy_decay_curve_chu(
        rir,
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=None,
        plot=False)
    npt.assert_allclose(actual.time, expected)

    # Test with a sufficiently high threshold to ensure exact matching of nans
    threshold = 15
    actual = enh.energy_decay_curve_chu(
        rir,
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=threshold,
        plot=False)

    mask = expected <= 10**((-40+threshold)/10)
    expected[mask] = np.nan

    pf.plot.time(actual, dB=True, log_prefix=10)

    pf.plot.time(pf.TimeData(expected, actual.times), dB=True, log_prefix=10)
    npt.assert_allclose(actual.time, expected)


def test_edc_chu_2D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=','), 3e3)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'edc_chu_2D.csv'),
        delimiter=','))

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.energy_decay_curve_chu(
        rir,
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=None,
        plot=False)
    npt.assert_allclose(actual.time, expected)


def test_intersection_time_1D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'intersection_time_1D.csv'),
        delimiter=',')).T

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.intersection_time_lundeby(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=False,
        channel_independent=False,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_intersection_time_2D(monkeypatch):
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=','), 3000)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'intersection_time_2D.csv'),
        delimiter=','))

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.intersection_time_lundeby(
        rir,
        freq='broadband',
        is_energy=False,
        time_shift=False,
        channel_independent=False,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_energy_decay_curve_threshold():

    t_60 = 1
    m = -60/t_60

    n_samples = 10
    times = np.linspace(0, 1, n_samples)
    edc_log = np.atleast_2d(times * m)

    edc_log = np.tile(edc_log, (2, 3, 1))

    edc = enh._truncate_energy_decay_curve(10**(edc_log.copy()/10), 30)

    edc_ref = 10**(edc_log.copy()/10)
    edc_ref[..., n_samples//2:] = np.nan

    npt.assert_allclose(edc, edc_ref)


def test_truncate_energy_decay_curve():
    t_60 = 1
    m = -60/t_60

    n_samples = 10
    times = np.linspace(0, 1, n_samples)
    edc_log = np.atleast_2d(times * m)

    edc_log = np.tile(edc_log, (2, 3, 1))

    edc = pf.TimeData(10**(edc_log.copy()/10), times)
    edc_trunc = enh.truncate_energy_decay_curve(edc, 30)

    edc_ref = 10**(edc_log.copy()/10)
    edc_ref[..., n_samples//2:] = np.nan

    npt.assert_allclose(edc_trunc.time, edc_ref)
