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
test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')


def mock_shift_samples_1d(*args, **kwargs):
    return np.array([76])


def mock_shift_samples_2d(*args, **kwargs):
    return np.array([76, 76])


def test_noise_energy_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'noise_energy_1D.csv'),
        delimiter=',')
    actual = enh.estimate_noise_energy(
        rir,
        interval=[0.9, 1.0],
        is_energy=False)
    npt.assert_allclose(actual, expected)


def test_noise_energy_2D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'noise_energy_2D.csv'),
        delimiter=',')
    actual = enh.estimate_noise_energy(
        rir,
        interval=[0.9, 1.0],
        is_energy=False)
    npt.assert_allclose(actual, expected)


def test_preprocessing_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'preprocessing_1D.csv'),
        delimiter=',')[np.newaxis]
    actual = enh.preprocess_rir(
        rir,
        is_energy=False,
        time_shift=False,
        channel_independent=False)[0]
    npt.assert_allclose(actual, expected)


def test_preprocessing_2D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'preprocessing_2D.csv'),
        delimiter=',')
    actual = enh.preprocess_rir(
        rir,
        is_energy=False,
        time_shift=False,
        channel_independent=False)[0]
    npt.assert_allclose(actual, expected)


def test_preprocessing_time_shift_1D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'preprocessing_time_shift_1D.csv'),
        delimiter=',')[np.newaxis]

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.preprocess_rir(
        rir,
        is_energy=False,
        time_shift=True,
        channel_independent=False)[0]
    npt.assert_allclose(actual, expected)


def test_preprocessing_time_shift_2D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'preprocessing_time_shift_2D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.preprocess_rir(
        rir,
        is_energy=False,
        time_shift=True,
        channel_independent=False)[0]
    npt.assert_allclose(actual, expected)


def test_preprocessing_time_shift_channel_independent_1D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(
            test_data_path,
            'preprocessing_time_shift_channel_independent_1D.csv'),
        delimiter=',')[np.newaxis]

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.preprocess_rir(
        rir,
        is_energy=False,
        time_shift=True,
        channel_independent=True)[0]
    npt.assert_allclose(actual, expected)


def test_preprocessing_time_shift_channel_independent_2D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(
            test_data_path,
            'preprocessing_time_shift_channel_independent_2D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.preprocess_rir(
        rir,
        is_energy=False,
        time_shift=True,
        channel_independent=True)[0]
    npt.assert_allclose(actual, expected)


def test_smoothed_rir_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'smoothed_rir_1D.csv'),
        delimiter=',')[np.newaxis]
    actual = enh.smooth_rir(
        rir,
        sampling_rate=3000,
        smooth_block_length=0.075)[0]
    npt.assert_allclose(actual, expected)


def test_smoothed_rir_2D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'smoothed_rir_2D.csv'),
        delimiter=',')
    actual = enh.smooth_rir(
        rir,
        sampling_rate=3000,
        smooth_block_length=0.075)[0]
    npt.assert_allclose(actual, expected)


def test_substracted_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'substracted_1D.csv'),
        delimiter=',')
    actual = enh.subtract_noise_from_squared_rir(rir**2)
    npt.assert_allclose(actual, expected)


def test_substracted_2D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'substracted_2D.csv'),
        delimiter=',')
    actual = enh.subtract_noise_from_squared_rir(rir**2)
    npt.assert_allclose(actual, expected)


def test_edc_truncation_1D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_truncation_1D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.energy_decay_curve_truncation(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True)
    npt.assert_allclose(actual, expected)


def test_edc_truncation_2D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_truncation_2D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.energy_decay_curve_truncation(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True)
    npt.assert_allclose(actual, expected)


def test_edc_lundeby_1D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_1D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.energy_decay_curve_lundeby(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_edc_lundeby_2D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_2D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.energy_decay_curve_lundeby(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_edc_lundeby_chu_1D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_chu_1D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.energy_decay_curve_chu_lundeby(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_edc_lundeby_chu_2D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_lundeby_chu_2D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.energy_decay_curve_chu_lundeby(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_edc_chu_1D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_chu_1D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.energy_decay_curve_chu(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_edc_chu_2D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'edc_chu_2D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.energy_decay_curve_chu(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_intersection_time_1D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'intersection_time_1D.csv'),
        delimiter=',')[np.newaxis].T

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_1d)

    actual = enh.intersection_time_lundeby(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=False,
        channel_independent=False,
        plot=False)
    npt.assert_allclose(actual, expected)


def test_intersection_time_2D(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'intersection_time_2D.csv'),
        delimiter=',')

    # monkeypatch.setattr(
    #     dsp,
    #     "find_impulse_response_start",
    #     mock_shift_samples_2d)

    actual = enh.intersection_time_lundeby(
        rir,
        sampling_rate=3000,
        freq='broadband',
        is_energy=False,
        time_shift=False,
        channel_independent=False,
        plot=False)
    npt.assert_allclose(actual, expected)
