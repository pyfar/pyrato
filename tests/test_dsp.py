# -*- coding: utf-8 -*-

import os

import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from numpy import genfromtxt
import pyrato.dsp as dsp

test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')


def test_max_ir():
    n_samples = 2**10
    ir = np.zeros(n_samples)

    snr = 60

    noise = pf.signals.noise(n_samples, rms=10**(-snr/20), seed=1)

    start_sample = 24
    ir[start_sample] = 1
    ir = pf.Signal(ir, 44100)
    start_sample_est = dsp.find_impulse_response_maximum(ir)
    assert start_sample_est == start_sample

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_maximum(ir_awgn)
    assert start_sample_est == start_sample

    with pytest.warns(match='SNR seems lower'):
        start_sample_est = dsp.find_impulse_response_maximum(
            ir_awgn, threshold=200)

# ------------------
# Time shift
# ------------------


@pytest.mark.parametrize("shift_samples", [10, -10, 0])
def test_time_shift_left_right(shift_samples):
    n_samples = 2**9
    ir = pf.signals.impulse(n_samples, delay=20)

    ir_truth = pf.signals.impulse(n_samples, 20+shift_samples)
    ir_shifted = dsp.time_shift(ir, shift_samples)

    npt.assert_allclose(ir_shifted.time, ir_truth.time)


def test_time_shift_return_vals():
    n_samples = 2**9
    ir = pf.signals.impulse(n_samples, delay=20)

    ir_shifted = dsp.time_shift(ir, 1, circular_shift=True)
    assert type(ir_shifted) is pf.Signal

    ir_shifted = dsp.time_shift(ir, 1, circular_shift=False)
    assert type(ir_shifted) is pf.TimeData


def test_time_shift_non_circular_left_right():
    shift_samples = 10
    n_samples = 2**9

    ir = pf.signals.impulse(n_samples, delay=20)

    ir_truth = np.zeros(n_samples, dtype=float)
    ir_truth[20+shift_samples] = 1
    ir_truth[:shift_samples] = np.nan

    ir_truth = pf.TimeData(ir_truth, np.arange(n_samples)/ir.sampling_rate)

    ir_shifted = dsp.time_shift(ir, shift_samples, circular_shift=False)
    npt.assert_allclose(ir_shifted.time, ir_truth.time, equal_nan=True)

    shift_samples = -10
    ir_truth = np.zeros(n_samples, dtype=float)
    ir_truth[20+shift_samples] = 1
    ir_truth[shift_samples:] = np.nan
    ir_truth = pf.TimeData(ir_truth, np.arange(n_samples)/ir.sampling_rate)

    ir_shifted = dsp.time_shift(ir, shift_samples, circular_shift=False)
    npt.assert_allclose(ir_shifted.time, ir_truth.time, equal_nan=True)


# ----------------
# Noise power estimation
# ----------------

def test_estimate_noise_power():
    n_samples = 2**18
    rms = 10**(-40/20)
    noise = pf.signals.noise(n_samples, rms=rms)
    actual = dsp.estimate_noise_energy(noise)

    npt.assert_allclose(actual, rms**2, rtol=1e-3, atol=1e-3)


def test_estimate_noise_power_private():
    n_samples = 2**18
    rms = 10**(-40/20)
    noise = pf.signals.noise(n_samples, rms=rms)
    actual = dsp._estimate_noise_energy(noise.time)

    npt.assert_allclose(actual, rms**2, rtol=1e-3, atol=1e-3)


def test_noise_energy_1D():
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=','), 1)
    expected = genfromtxt(
        os.path.join(test_data_path, 'noise_energy_1D.csv'),
        delimiter=',')
    actual = dsp.estimate_noise_energy(
        rir,
        interval=[0.9, 1.0],
        is_energy=False)
    npt.assert_allclose(actual, expected)


def test_noise_energy_2D():
    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=','), 1)
    expected = genfromtxt(
        os.path.join(test_data_path, 'noise_energy_2D.csv'),
        delimiter=',')
    actual = dsp.estimate_noise_energy(
        rir,
        interval=[0.9, 1.0],
        is_energy=False)
    npt.assert_allclose(actual, expected)


def test_psnr():
    n_samples = 2**20
    peak_levels = np.array([0, -6, -10])
    noise_level = np.array([-20, -30, -40])
    imp = pf.signals.impulse(n_samples, amplitude=10**(peak_levels/20))
    awgn = pf.signals.noise(n_samples, rms=10**(noise_level/20), seed=7)
    psnr = dsp.peak_signal_to_noise_ratio(imp+awgn)

    npt.assert_allclose(
        1/psnr, 10**((peak_levels + noise_level)/10), rtol=1e-2, atol=1e-2)

# ----------------
# RIR preprocessing
# ----------------

def test_preprocessing_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    rir = pf.Signal(rir, 1)
    actual = dsp.preprocess_rir(
        rir,
        is_energy=False,
        shift=False,
        channel_independent=False)[0]

    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'preprocessing_1D.csv'),
        delimiter=','))
    npt.assert_allclose(actual.time, np.atleast_2d(expected))


def test_preprocessing_2D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=',')
    rir = pf.Signal(rir, 1)

    actual = dsp.preprocess_rir(
        rir,
        is_energy=False,
        shift=False,
        channel_independent=False)

    expected = genfromtxt(
        os.path.join(test_data_path, 'preprocessing_2D.csv'),
        delimiter=',')
    npt.assert_allclose(actual.time, expected)


def test_preprocessing_time_shift_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    rir = pf.Signal(rir, 1)

    actual = dsp.preprocess_rir(
        rir,
        is_energy=False,
        shift=True,
        channel_independent=False)[0]

    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'preprocessing_time_shift_1D.csv'),
        delimiter=','))
    npt.assert_allclose(actual.time, expected)


def test_preprocessing_time_shift_2D():
    rir = pf.Signal(
        genfromtxt(
            os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
            delimiter=','),
        1)

    expected = np.atleast_2d(genfromtxt(
        os.path.join(test_data_path, 'preprocessing_time_shift_2D.csv'),
        delimiter=','))

    actual = dsp.preprocess_rir(
        rir,
        is_energy=False,
        shift=True,
        channel_independent=False)
    npt.assert_allclose(actual.time, expected)


def test_preprocessing_time_shift_channel_independent_1D():
    rir = pf.Signal(
        genfromtxt(
            os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
            delimiter=','),
        1)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(
            test_data_path,
            'preprocessing_time_shift_channel_independent_1D.csv'),
        delimiter=','))

    actual = dsp.preprocess_rir(
        rir,
        is_energy=False,
        shift=True,
        channel_independent=True)[0]
    npt.assert_allclose(actual.time, expected)


def test_preprocessing_time_shift_channel_independent_2D():

    rir = pf.Signal(genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_2D.csv'),
        delimiter=','), 1)
    expected = np.atleast_2d(genfromtxt(
        os.path.join(
            test_data_path,
            'preprocessing_time_shift_channel_independent_2D.csv'),
        delimiter=','))

    actual = dsp.preprocess_rir(
        rir,
        is_energy=False,
        shift=True,
        channel_independent=True)
    npt.assert_allclose(actual.time, expected)


def test_smoothed_rir_1D():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')
    expected = genfromtxt(
        os.path.join(test_data_path, 'smoothed_rir_1D.csv'),
        delimiter=',')[np.newaxis]
    actual = dsp._smooth_rir(
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
    actual = dsp._smooth_rir(
        rir,
        sampling_rate=3000,
        smooth_block_length=0.075)[0]
    npt.assert_allclose(actual, expected)
