# -*- coding: utf-8 -*-

import os

import numpy as np
import numpy.testing as npt
import pytest
from numpy import genfromtxt

import pyrato.dsp as dsp

test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')


def test_start_ir_insufficient_snr():
    n_samples = 2**9
    ir = np.zeros(n_samples, dtype=float)
    ir[20] = 1

    snr = 15

    noise = np.random.randn(n_samples)
    noise = noise / np.sqrt(np.mean(np.abs(noise**2))) * 10**(-snr/20)
    ir_noise = ir + noise

    with pytest.raises(ValueError):
        dsp.find_impulse_response_start(ir_noise)


def test_start_ir():
    n_samples = 2**10
    ir = np.zeros(n_samples)

    snr = 60

    noise = np.random.randn(n_samples) * 10**(-snr/20)

    start_sample = 24
    ir[start_sample] = 1

    start_sample_est = dsp.find_impulse_response_start(ir)
    assert start_sample_est == start_sample - 1

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)
    assert start_sample_est == start_sample - 1


def test_start_ir_thresh():
    n_samples = 2**10
    ir = np.zeros(n_samples)

    start_sample = 24
    ir[start_sample] = 1
    ir[start_sample-4:start_sample] = 10**(-5/10)

    start_sample_est = dsp.find_impulse_response_start(ir, threshold=20)
    assert start_sample_est == start_sample - 4 - 1


def test_start_ir_multidim():
    n_samples = 2**10
    n_channels = 3
    ir = np.zeros((n_channels, n_samples))

    snr = 60

    noise = np.random.randn(n_channels, n_samples) * 10**(-snr/20)

    start_sample = [24, 5, 43]
    ir[[0, 1, 2], start_sample] = 1

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)

    npt.assert_allclose(start_sample_est, np.array(start_sample) - 1)

    ir = np.zeros((2, n_channels, n_samples))
    noise = np.random.randn(2, n_channels, n_samples) * 10**(-snr/20)

    start_sample_1 = [24, 5, 43]
    ir[0, [0, 1, 2], start_sample_1] = 1
    start_sample_2 = [14, 12, 16]
    ir[1, [0, 1, 2], start_sample_2] = 1

    start_samples = np.vstack((start_sample_1, start_sample_2))

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)

    npt.assert_allclose(start_sample_est, start_samples - 1)


def test_time_shift_right():
    shift_samples = 10
    n_samples = 2**9
    ir = np.zeros(n_samples, dtype=float)
    ir[20] = 1

    ir_truth = np.zeros(n_samples, dtype=float)
    ir_truth[20+shift_samples] = 1
    ir_shifted = dsp.time_shift(ir, shift_samples)

    npt.assert_allclose(ir_shifted, ir_truth)


def test_time_shift_left():
    shift_samples = 10
    n_samples = 2**9
    ir = np.zeros(n_samples, dtype=float)
    ir[20] = 1

    ir_truth = np.zeros(n_samples, dtype=float)
    ir_truth[20-shift_samples] = 1
    ir_shifted = dsp.time_shift(ir, -shift_samples)

    npt.assert_allclose(ir_shifted, ir_truth)


def test_time_shift_non_circular_right():
    shift_samples = 10
    n_samples = 2**9
    ir = np.zeros(n_samples, dtype=float)
    ir[20] = 1

    ir_truth = np.zeros(n_samples, dtype=float)
    ir_truth[20+shift_samples] = 1
    ir_truth[:shift_samples] = np.nan
    ir_shifted = dsp.time_shift(ir, shift_samples, circular_shift=False)

    npt.assert_allclose(ir_shifted, ir_truth, equal_nan=True)


def test_time_shift_non_circular_left():
    shift_samples = 10
    n_samples = 2**9
    ir = np.zeros(n_samples, dtype=float)
    ir[20] = 1

    ir_truth = np.zeros(n_samples, dtype=float)
    ir_truth[20-shift_samples] = 1
    ir_truth[n_samples-shift_samples:] = np.nan
    ir_shifted = dsp.time_shift(ir, -shift_samples, circular_shift=False)

    npt.assert_allclose(ir_shifted, ir_truth, equal_nan=True)


def test_time_shift_multitim():
    shift_samples = 10
    n_samples = 2**10
    n_channels = 3
    ir = np.zeros((n_channels, n_samples))

    start_sample = [24, 5, 43]
    ir[[0, 1, 2], start_sample] = 1

    ir_truth = np.zeros((n_channels, n_samples), dtype=float)
    start_sample_truth = [24+shift_samples, 5+shift_samples, 43+shift_samples]
    ir_truth[[0, 1, 2], start_sample_truth] = 1

    ir_shifted = dsp.time_shift(ir, shift_samples)

    npt.assert_allclose(ir_shifted, ir_truth)

    ir_truth = np.zeros((n_channels, n_samples), dtype=float)
    start_sample_truth = [24-shift_samples, 5-shift_samples, 43-shift_samples]
    ir_truth[[0, 1, 2], start_sample_truth] = 1

    ir_shifted = dsp.time_shift(ir, -shift_samples)

    npt.assert_allclose(ir_shifted, ir_truth)


def test_time_shift_multitim_multishift():
    shift_samples = [10, 2, 4]
    n_samples = 40
    n_channels = 3
    ir = np.zeros((n_channels, n_samples), dtype=float)

    start_sample = [24, 5, 13]
    ir[[0, 1, 2], start_sample] = 1

    ir_truth = np.zeros((n_channels, n_samples), dtype=float)
    start_sample_truth = [
        24+shift_samples[0],
        5+shift_samples[1],
        13+shift_samples[2]]
    ir_truth[[0, 1, 2], start_sample_truth] = 1

    ir_shifted = dsp.time_shift(ir, shift_samples)

    npt.assert_allclose(ir_shifted, ir_truth)

    ir_truth = np.zeros((n_channels, n_samples), dtype=float)
    start_sample_truth = [
        24-shift_samples[0],
        5-shift_samples[1],
        13-shift_samples[2]]
    ir_truth[[0, 1, 2], start_sample_truth] = 1

    ir_shifted = dsp.time_shift(ir, -np.array(shift_samples, dtype=int))

    npt.assert_allclose(ir_shifted, ir_truth)


def test_start_room_impulse_response():
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')

    actual = dsp.find_impulse_response_start(rir, threshold=20)

    expected = 0

    npt.assert_allclose(actual, expected)


def test_start_room_impulse_response_shfted(monkeypatch):
    rir = genfromtxt(
        os.path.join(test_data_path, 'analytic_rir_psnr50_1D.csv'),
        delimiter=',')

    rir_shifted = np.roll(rir, 128, axis=-1)
    actual = dsp.find_impulse_response_start(rir_shifted, threshold=20)

    expected = 128

    npt.assert_allclose(actual, expected)


def test_start_ir_thresh_invalid():
    n_samples = 2**10
    ir = np.zeros(n_samples)

    start_sample = 24
    ir[start_sample] = 1
    # ir[start_sample-4:start_sample] = 10**(-10/10)
    ir[0:start_sample] = 10**(-5/10)

    start_sample_est = dsp.find_impulse_response_start(ir, threshold=20)
    assert start_sample_est == 0


def test_start_ir_thresh_invalid_osci():
    n_samples = 2**10
    ir = np.zeros(n_samples)

    start_sample = 24
    ir[start_sample] = 1
    ir[start_sample-4:start_sample] = 10**(-30/10)
    ir[0:start_sample-4] = 10**(-5/10)

    start_sample_est = dsp.find_impulse_response_start(ir, threshold=20)
    assert start_sample_est == 0


def test_max_ir():
    n_samples = 2**10
    ir = np.zeros(n_samples)

    snr = 60

    noise = np.random.randn(n_samples) * 10**(-snr/20)

    start_sample = 24
    ir[start_sample] = 1

    start_sample_est = dsp.find_impulse_response_maximum(ir)
    assert start_sample_est == start_sample

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_maximum(ir_awgn)
    assert start_sample_est == start_sample
