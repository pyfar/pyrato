#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for STI. """

import pytest
import numpy as np
from pyfar import Signal
from roomacoustics import sti


def test_sti_1D():
    expected = np.ones((1))
    sig = Signal(np.zeros((1, 131072)), 44100)
    sig.time[0] = 1
    array = sti.sti(sig)
    np.testing.assert_allclose(array, expected, rtol=1e-2, atol=0)


def test_sti_2D():
    expected = np.ones((2, 2))
    sig = Signal(np.zeros((2, 2, 131072)), 44100)
    sig.time[0, 0, 0] = 1
    sig.time[1, 0, 0] = 1
    sig.time[0, 1, 0] = 1
    sig.time[1, 1, 0] = 1
    array = sti.sti(sig)
    np.testing.assert_allclose(array, expected, rtol=1e-2, atol=0)


def test_sti_warn_length():
    sig = Signal(np.zeros((1, 31072)), 44100)
    sig.time[0] = 1
    with pytest.warns(UserWarning, match="Signal length below 1.6 seconds."):
        sti.sti(sig)


def test_sti_male():
    expected = np.array([0.6098])
    sig = Signal(np.zeros((1, 131072)), 44100)
    sig.time[0, 0] = 1
    sig.time[0, 1000] = 1
    sig.time[0, 10000] = 1
    array = sti.sti(sig, gender="male")
    np.testing.assert_allclose(array, expected, rtol=1e-3, atol=0)


def test_sti_female():
    expected = np.array([0.6116])
    sig = Signal(np.zeros((1, 131072)), 44100)
    sig.time[0, 0] = 1
    sig.time[0, 1000] = 1
    sig.time[0, 10000] = 1
    array = sti.sti(sig, gender="female")
    np.testing.assert_allclose(array, expected, rtol=1e-3, atol=0)


def test_sti_value_error_gender():
    sig = Signal(np.zeros((1, 31072)), 44100)
    sig.time[0] = 1
    with pytest.raises(ValueError, match="Gender is 'generic' but must be "
                       "'male' or 'female'."):
        sti.sti(sig, gender="generic")


def test_sti_lvl_snr_1D():
    expected = np.array([0.8933])
    sig = Signal(np.zeros((1, 131072)), 44100)
    sig.time[0, 0] = 1
    sig.time[0, 1000] = 1
    lvl = np.array([70, 70, 70, 70, 70, 70, 70])
    sn = np.array([30, 30, 30, 30, 30, 30, 30])
    array = sti.sti(sig, level=lvl, snr=sn)
    np.testing.assert_allclose(array, expected, rtol=1e-3, atol=0)


def test_sti_lvl_snr_2D():
    expected = np.array([0.8934, 0.8888])
    sig = Signal(np.zeros((2, 131072)), 44100)
    sig.time[:, 0] = 1
    sig.time[:, 1000] = 1
    lvl = np.array([[70, 70, 70, 70, 70, 70, 70],
                    [80, 70, 60, 50, 40, 50, 60]])
    sn = np.array([[30, 30, 30, 30, 30, 30, 30], [10, 20, 30, 40, 50, 60, 70]])
    array = sti.sti(sig, level=lvl, snr=sn)
    np.testing.assert_allclose(array, expected, rtol=1e-3, atol=0)


def test_sti_warn_snr_low():
    sig = Signal(np.zeros((1, 31072)), 44100)
    sig.time[0] = 1
    lvl = np.array([70, 70, 70, 70, 70, 70, 70])
    sn = np.array([10, 30, 30, 30, 30, 30, 30])
    with pytest.warns(UserWarning, match="SNR should be at least 20 dB for "
                      "every octave band."):
        sti.sti(sig, level=lvl, snr=sn)


def test_sti_warn_data_type_not_given():
    sig = Signal(np.zeros((1, 31072)), 44100)
    sig.time[0] = 1
    with pytest.warns(UserWarning, match="Data type is considered as "
                      "acoustical. Consideration of masking effects not valid "
                      "for electrically obtained signals."):
        sti.sti(sig)


def test_sti_warn_data_type_unknown():
    sig = Signal(np.zeros((1, 31072)), 44100)
    sig.time[0] = 1
    with pytest.raises(ValueError, match="Data_type is 'generic' but must "
                       "be 'electrical' or 'acoustical'."):
        sti.sti(sig, data_type="generic")


def test_sti_value_error_snr():
    sig = Signal(np.zeros((1, 31072)), 44100)
    sig.time[0] = 1
    sn = np.array([30, 30, 30, 30, 30, 30])
    with pytest.raises(ValueError, match="SNR consists of wrong number of "
                       "components."):
        sti.sti(sig, snr=sn)


def test_sti_value_error_level():
    sig = Signal(np.zeros((1, 31072)), 44100)
    sig.time[0] = 1
    lvl = np.array([30, 30, 30, 30, 30, 30])
    with pytest.raises(ValueError, match="Level consists of wrong number of "
                       "components."):
        sti.sti(sig, level=lvl)


def test_sti_amb_false():
    expected = np.array([0.8864])
    sig = Signal(np.zeros((1, 131072)), 44100)
    sig.time[0, 0] = 1
    sig.time[0, 1000] = 1
    lvl = np.array([70, 70, 70, 70, 70, 70, 70])
    sn = np.array([20, 20, 20, 20, 20, 20, 20])
    array = sti.sti(sig, level=lvl, snr=sn, amb=False)
    np.testing.assert_allclose(array, expected, rtol=1e-3, atol=0)
