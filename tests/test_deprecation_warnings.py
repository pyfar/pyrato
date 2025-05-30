import pytest
import pyfar as pf
import pyrato
import numpy as np


def test_warning_fractional_octave_bands():

    with pytest.warns(DeprecationWarning, match='0.5.0'):
        sig = pf.Signal([1, 2, 3], 44100)
        pyrato.dsp.filter_fractional_octave_bands(
            sig, 3, freq_range=(20, 16000))


def test_warning_center_frequencies_thirds():

    nominal = np.array([
        25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000,
        6300, 8000, 10000, 12500, 16000, 20000], dtype=float)

    with pytest.warns(DeprecationWarning, match='0.5.0'):
        nom = pyrato.dsp.center_frequencies_third_octaves()[0]

    np.testing.assert_allclose(nom, nominal)


def test_warning_center_frequencies_octaves():
    with pytest.warns(DeprecationWarning, match='0.5.0'):
        nom, exact = pyrato.dsp.center_frequencies_octaves()


def test_warning_start_ir():
    with pytest.warns(DeprecationWarning, match='0.5.0'):

        sig = pf.Signal([0, 0, 1, 0, 0], 44100)
        pyrato.dsp.find_impulse_response_start(sig)


def test_warning_rt_edc():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)
    with pytest.warns(DeprecationWarning, match='0.5.0'):
        pyrato.reverberation_time_energy_decay_curve(
            edc_exp)
