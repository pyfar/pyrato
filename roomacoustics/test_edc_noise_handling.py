#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for edc noise handling related things. """

import numpy as np
from numpy import array
import os
import numpy.testing as npt
import roomacoustics as ra
from roomacoustics import edc_noise_handling as enh

def test_estimate_noise_energy():
    sampling_rate = 44100
    duration = 1
    alphas = [0.9, 0.1]
    surfaces = [2, 5*2]
    volume = 2*2*2
    times = np.linspace(0,duration,sampling_rate*duration)
    edc = ra.energy_decay_curve_analytic(
        surfaces, alphas, volume, times, method='sabine')
    PSNR = 50    # (peak)signal to noise ratio in dB
    noise = 10**(-(PSNR) / 10)
    edc_noise = edc + noise # add background noise (10 dB difference between peak and mean)
    energy = 10*np.log10(enh.estimate_noise_energy(edc_noise))
    truth = -PSNR
    npt.assert_almost_equal(energy, truth, decimal=5) # all_close
    # fester seed

def test_remove_silence_at_beginning_and_square_data():
    dirname = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(dirname, '../resources/analytic_rir_psnr50_ml.bin')
    with open(filename, 'rb') as f:
        analytic_rir_psnr50 = np.fromfile(f, dtype=float)
    analytic_rir_psnr50 /= np.amax(np.abs(analytic_rir_psnr50))

    shift = analytic_rir_psnr50.size - enh.remove_silence_at_beginning_and_square_data(analytic_rir_psnr50).size
    truth = 534
    # auf nicht-negativität prüfen?
    npt.assert_equal(shift, truth) # all_close

def test_smooth_edc():
    # Ausreichend, nur die Dimensionen zu testen?
    dirname = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(dirname, '../resources/analytic_rir_psnr50_ml.bin')
    with open(filename, 'rb') as f:
        analytic_rir_psnr50 = np.fromfile(f, dtype=float)
    analytic_rir_psnr50 /= np.amax(np.abs(analytic_rir_psnr50))
    sampling_rate = 44100
    smooth_block_length = 0.075

    time_window_data, time_vector_window, time_vector = enh.smooth_edc(
        analytic_rir_psnr50, sampling_rate, smooth_block_length)

    n_samples = analytic_rir_psnr50.shape[0]
    n_samples_per_block = np.round(smooth_block_length * sampling_rate, 0)
    n_blocks = int(np.floor(n_samples/n_samples_per_block))

    # TEST CORRECT DIMENSIONS:
    npt.assert_equal(time_window_data.shape[-1], n_blocks)
    npt.assert_equal(time_vector_window.shape[-1], n_blocks)
    npt.assert_equal(time_vector.shape[-1], analytic_rir_psnr50.shape[-1])


def test_substract_noise_from_edc():
    dirname = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(dirname, '../resources/analytic_rir_psnr50_ml.bin')
    with open(filename, 'rb') as f:
        analytic_rir_psnr50 = np.fromfile(f, dtype=float)
    analytic_rir_psnr50 /= np.amax(np.abs(analytic_rir_psnr50))
    data = analytic_rir_psnr50**2

    region_start = 0.9
    region_end = 1
    region_start_idx = int(np.round(data.shape[-1]*region_start))
    region_end_idx = int(np.round(data.shape[-1]*region_end))
    result = enh.substract_noise_from_edc(data)
    noise_energy = np.mean(np.abs(result[region_start_idx:region_end_idx]))

    npt.assert_allclose(noise_energy, 0, atol=1e-04)


def test_energy_decay_curve_truncation():
    dirname = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(dirname, '../resources/analytic_rir_psnr50_ml.bin')
    with open(filename, 'rb') as f:
        analytic_rir_psnr50 = np.fromfile(f, dtype=float)


    return
def test_energy_decay_curve_lundeby():
    return
def test_energy_decay_curve_chu():
    return
def test_energy_decay_curve_chu_lundeby():
    return
def test_intersection_time_lundeby():
    return
