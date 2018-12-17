# -*- coding: utf-8 -*-

import pytest
import numpy as np
import numpy.testing as npt
import scipy.signal as ssignal
import roomacoustics.dsp as dsp


def test_start_ir_insufficient_snr():
    n_samples = 2**9
    ir = np.zeros(n_samples, dtype=np.double)
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
    assert start_sample_est == start_sample

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)
    assert start_sample_est == start_sample


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

    npt.assert_allclose(start_sample_est, start_sample)

    ir = np.zeros((2, n_channels, n_samples))
    noise = np.random.randn(2, n_channels, n_samples) * 10**(-snr/20)

    start_sample_1 = [24, 5, 43]
    ir[0, [0, 1, 2], start_sample_1] = 1
    start_sample_2 = [14, 12, 16]
    ir[1, [0, 1, 2], start_sample_2] = 1

    start_samples = np.vstack((start_sample_1, start_sample_2))

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)

    npt.assert_allclose(start_sample_est, start_samples)



