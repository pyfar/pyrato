#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for edc noise handling related things. """

import numpy as np
import os
import numpy.testing as npt
from roomacoustics import edc_noise_handling as enh
from numpy import genfromtxt

def test_noise_energy_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/noise_energy_1D.csv', delimiter=',')
    actual = enh.estimate_noise_energy(rir, interval=[0.9, 1.0], is_energy=False)
    npt.assert_allclose(actual, expected)

def test_noise_energy_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/noise_energy_2D.csv', delimiter=',')
    actual = enh.estimate_noise_energy(rir, interval=[0.9, 1.0], is_energy=False)
    npt.assert_allclose(actual, expected)

def test_preprocessing_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/preprocessing_1D.csv', delimiter=',')
    actual = enh.preprocess_rir(rir, is_energy=False, time_shift=False, channel_independent=False)[0]
    npt.assert_allclose(actual, expected)

def test_preprocessing_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/preprocessing_2D.csv', delimiter=',')
    actual = enh.preprocess_rir(rir, is_energy=False, time_shift=False, channel_independent=False)[0]
    npt.assert_allclose(actual, expected)

def test_preprocessing_time_shift_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/preprocessing_time_shift_1D.csv', delimiter=',')
    actual = enh.preprocess_rir(rir, is_energy=False, time_shift=True, channel_independent=False)[0]
    npt.assert_allclose(actual, expected)

def test_preprocessing_time_shift_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/preprocessing_time_shift_2D.csv', delimiter=',')
    actual = enh.preprocess_rir(rir, is_energy=False, time_shift=True, channel_independent=False)[0]
    npt.assert_allclose(actual, expected)

def test_preprocessing_time_shift_channel_independent_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/preprocessing_time_shift_channel_independent_1D.csv', delimiter=',')
    actual = enh.preprocess_rir(rir, is_energy=False, time_shift=True, channel_independent=True)[0]
    npt.assert_allclose(actual, expected)

def test_preprocessing_time_shift_channel_independent_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/preprocessing_time_shift_channel_independent_2D.csv', delimiter=',')
    actual = enh.preprocess_rir(rir, is_energy=False, time_shift=True, channel_independent=True)[0]
    npt.assert_allclose(actual, expected)

def test_smoothed_rir_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/smoothed_rir_1D.csv', delimiter=',')
    actual = enh.smooth_rir(rir, sampling_rate=3000, smooth_block_length=0.075)[0]
    npt.assert_allclose(actual, expected)

def test_smoothed_rir_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/smoothed_rir_2D.csv', delimiter=',')
    actual = enh.smooth_rir(rir, sampling_rate=3000, smooth_block_length=0.075)[0]
    npt.assert_allclose(actual, expected)

def test_substracted_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/substracted_1D.csv', delimiter=',')
    actual = enh.subtract_noise_from_squared_rir(rir**2)
    npt.assert_allclose(actual, expected)

def test_substracted_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/substracted_2D.csv', delimiter=',')
    actual = enh.subtract_noise_from_squared_rir(rir**2)
    npt.assert_allclose(actual, expected)

def test_edc_truncation_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_truncation_1D.csv', delimiter=',')
    actual = enh.energy_decay_curve_truncation(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True)
    npt.assert_allclose(actual, expected)

def test_edc_truncation_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_truncation_2D.csv', delimiter=',')
    actual = enh.energy_decay_curve_truncation(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True)
    npt.assert_allclose(actual, expected)

def test_edc_lundeby_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_lundeby_1D.csv', delimiter=',')
    actual = enh.energy_decay_curve_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    npt.assert_allclose(actual, expected)

def test_edc_lundeby_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_lundeby_2D.csv', delimiter=',')
    actual = enh.energy_decay_curve_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    npt.assert_allclose(actual, expected)

def test_edc_lundeby_chu_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_lundeby_chu_1D.csv', delimiter=',')
    actual = enh.energy_decay_curve_chu_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    npt.assert_allclose(actual, expected)

def test_edc_lundeby_chu_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_lundeby_chu_2D.csv', delimiter=',')
    actual = enh.energy_decay_curve_chu_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    npt.assert_allclose(actual, expected)

def test_edc_chu_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_chu_1D.csv', delimiter=',')
    actual = enh.energy_decay_curve_chu(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    npt.assert_allclose(actual, expected)

def test_edc_chu_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/edc_chu_2D.csv', delimiter=',')
    actual = enh.energy_decay_curve_chu(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    npt.assert_allclose(actual, expected)

def test_intersection_time_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/intersection_time_1D.csv', delimiter=',')
    actual = enh.intersection_time_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=False,
        channel_independent=False, plot=False)
    npt.assert_allclose(actual, expected)

def test_intersection_time_2D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/intersection_time_2D.csv', delimiter=',')
    actual = enh.intersection_time_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=False,
        channel_independent=False, plot=False)
    npt.assert_allclose(actual, expected)

def test_noise_energy_from_edc_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_1D.csv', delimiter=',')
    expected = genfromtxt('../test_data/noise_energy_from_edc_1D.csv', delimiter=',')
    edc_lundeby_chu_1D = enh.energy_decay_curve_chu_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    intersection_time_1D = enh.intersection_time_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=False,
        channel_independent=False, plot=False)
    actual = enh.estimate_noise_energy_from_edc(
        edc_lundeby_chu_1D, intersection_time_1D[0], sampling_rate=3000)
    npt.assert_allclose(actual, expected)

def test_noise_energy_from_edc_1D():
    rir = genfromtxt('../test_data/analytic_rir_psnr50_2D.csv', delimiter=',')
    expected = genfromtxt('../test_data/noise_energy_from_edc_2D.csv', delimiter=',')
    edc_lundeby_chu_2D = enh.energy_decay_curve_chu_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=True,
        channel_independent=False, normalize=True, plot=False)
    intersection_time_2D = enh.intersection_time_lundeby(
        rir, sampling_rate=3000, freq='broadband', is_energy=False, time_shift=False,
        channel_independent=False, plot=False)
    actual = enh.estimate_noise_energy_from_edc(
        edc_lundeby_chu_2D, intersection_time_2D[0], sampling_rate=3000)
    npt.assert_allclose(actual, expected)
