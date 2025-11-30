import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
import numpy.testing as npt
import re

from pyrato.parameters import clarity
from pyrato.parameters import _energy_ratio

# parameter clarity tests
def test_clarity_accepts_timedata_returns_correct_type(make_edc):
    energy = np.concatenate(([1, 1, 1, 1], np.zeros(124)))
    edc = make_edc(energy=energy)

    result = clarity(edc, early_time_limit=4)  # 4 ms
    assert isinstance(result, (float, np.ndarray))
    assert result.shape == edc.cshape

def test_clarity_rejects_non_timedata_input():
    invalid_input = np.array([1, 2, 3])
    expected_error_message = "Input must be a pyfar.TimeData object."

    with pytest.raises(TypeError, match=re.escape(expected_error_message)):
        clarity(invalid_input)

def test_clarity_rejects_non_numeric_early_time_limit(make_edc):
    edc = make_edc()
    invalid_time_limit = "not_a_number"
    expected_error_message = "early_time_limit must be a number."

    with pytest.raises(TypeError, match=re.escape(expected_error_message)):
        clarity(edc, invalid_time_limit)

def test_clarity_rejects_complex_timedata():
    complex_data = pf.TimeData(np.array([1+1j, 2+2j, 3+3j]),
                               np.arange(3) / 1000, is_complex=True)
    expected_error_message = "Complex-valued input detected. Clarity is"
    "only defined for real TimeData."

    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        clarity(complex_data, early_time_limit=2)

def test_clarity_rejects_invalid_time_range(make_edc):
    energy = np.zeros(128)
    edc = make_edc(energy=energy)
    actual_signal_length_ms = edc.signal_length * 1000

    # Test negative time limit
    expected_error_message = "early_time_limit must be in the range of 0"
    f"and {actual_signal_length_ms}."
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        clarity(edc, early_time_limit=-1)

    # Test time limit beyond signal length
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        clarity(edc, early_time_limit=200000)

def test_clarity_preserves_multichannel_shape(make_edc):
    energy = np.ones((2,2,10)) / (1+np.arange(10))
    edc = make_edc(energy=energy, sampling_rate=10)
    output = clarity(edc, early_time_limit=80)
    assert edc.cshape == output.shape


def test_clarity_returns_nan_for_zero_signal():
    edc = pf.TimeData(np.zeros((1, 128)), np.arange(128) / 1000)
    result = clarity(edc)
    assert np.isnan(result)


def test_clarity_calculates_known_reference_value(make_edc):
    # Linear decay → early_time_limit at 1/2 energy -> ratio = 1 -> 0 dB
    edc_vals = np.array([1.0, 0.75, 0.5, 0.0])  # monotonic decay
    edc = make_edc(energy=edc_vals, sampling_rate=1000)

    result = clarity(edc, early_time_limit=2)
    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_clarity_values_for_given_ratio(make_edc):
    energy_early = 1
    energy_late = .5
    energy = np.zeros((3, 1000))
    edc = make_edc(energy=energy,
                               sampling_rate=1000,
                               dynamic_range = 120.0)
    edc.time[..., 10] = energy_early
    edc.time[..., 100] = energy_late
    edc = ra.edc.schroeder_integration(edc, is_energy=True)
    edc = pf.dsp.normalize(edc, reference_method='max')
    result = clarity(edc, early_time_limit=80)
    clarity_value_db = 10 * np.log10(energy_early/energy_late)
    npt.assert_allclose(result, clarity_value_db, atol=1e-6)

def test_clarity_for_exponential_decay(make_edc):
    rt60 = 2.0  # seconds
    sampling_rate = 1000
    total_samples = 2000
    early_cutoff = 80  # ms

    # Generate EDC
    edc = make_edc(rt=rt60,
                               sampling_rate=sampling_rate,
                               total_samples=total_samples)
    result = clarity(edc, early_time_limit=early_cutoff)

    # Analytical expected value
    te = early_cutoff / 1000  # convert ms to seconds
    a = 13.8155 / rt60
    expected_ratio = np.exp(a * te) - 1
    expected_dB = 10 * np.log10(expected_ratio)
    np.testing.assert_allclose(result, expected_dB, atol=1e-6)

# _energy_ratio tests
def test_energy_ratio_accepts_timedata_and_returns_correct_shape(make_edc):
    energy = np.linspace(1, 0, 10)
    edc = make_edc(energy=energy, sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    result = _energy_ratio(limits, edc, edc)
    assert isinstance(result, np.ndarray)
    assert result.shape == edc.cshape

def test_energy_ratio_rejects_non_timedata_input():
    invalid_input = np.arange(10)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    expected_message = "pyfar.TimeData"
    with pytest.raises(TypeError, match=expected_message):
        _energy_ratio(limits, invalid_input, invalid_input)

def test_energy_ratio_rejects_if_second_edc_is_not_timedata(make_edc):
    edc = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    with pytest.raises(TypeError, match="pyfar.TimeData"):
        _energy_ratio(limits, edc, "invalid_type")

def test_energy_ratio_rejects_non_numpy_array_limits(make_edc):
    edc = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    result = _energy_ratio([0.0, 0.001, 0.0, 0.005], edc, edc)
    assert isinstance(result, np.ndarray)

def test_energy_ratio_rejects_wrong_shape_limits(make_edc):
    edc = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    wrong_shape_limits = np.array([0.0, 0.001, 0.005])  # Only 3 elements
    with pytest.raises(ValueError, match="limits must have shape"):
        _energy_ratio(wrong_shape_limits, edc, edc)

def test_energy_ratio_computes_known_ratio_correctly(make_edc):
    """
    If EDC is linear, energy ratio should be 1.

    numerator = e(lim3)-e(lim4) = (1.0 - 0.75) = 0.25
    denominator = e(lim1)-e(lim2) = (0.75 - 0.5) = 0.25
    ratio = 1
    """
    edc_vals = np.array([1.0, 0.75, 0.5, 0.25])
    edc = make_edc(energy=edc_vals, sampling_rate=1000)

    # For linear EDC:
    limits = np.array([0.0, 0.001, 0.001, 0.002])
    result = _energy_ratio(limits, edc, edc)
    npt.assert_allclose(result, 1.0, atol=1e-12)

def test_energy_ratio_handles_multichannel_data_correctly(make_edc):
    energy = np.linspace(1, 0, 10)
    multi = np.stack([energy, energy * 0.5])
    edc = make_edc(energy=multi, sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    result = _energy_ratio(limits, edc, edc)
    assert result.shape == edc.cshape

def test_energy_ratio_returns_nan_for_zero_denominator(make_edc):
    """If denominator e(lim1)-e(lim2)=0, expect NaN (invalid ratio)."""
    energy = np.ones(10)
    edc = make_edc(energy=energy, sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.002, 0.003])
    result = _energy_ratio(limits, edc, edc)
    assert np.isnan(result)

def test_energy_ratio_matches_reference_case(make_edc):
    """
    Analytical reference:
    EDC = exp(-a*t). For exponential decay, ratio known analytically.
    """
    sampling_rate = 1000
    a = 13.8155  # decay constant
    times = np.arange(1000) / sampling_rate
    edc_vals = np.exp(-a * times)
    edc = make_edc(energy=edc_vals, sampling_rate=sampling_rate)

    limits = np.array([0.0, 0.02, 0.0, 0.05])
    lim1, lim2, lim3, lim4 = limits

    analytical_ratio = (
        (np.exp(-a*lim3) - np.exp(-a*lim4)) /
        (np.exp(-a*lim1) - np.exp(-a*lim2))
    )

    result = _energy_ratio(limits, edc, edc)
    npt.assert_allclose(result, analytical_ratio, atol=1e-8)

def test_energy_ratio_works_with_two_different_edcs(make_edc):
    """
    Energy ratio between two different EDCs.
    should compute distinct ratio.
    """
    edc1 = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    edc2 = make_edc(energy=np.linspace(1, 0, 10) ** 2, sampling_rate=1000)

    limits = np.array([0.0, 0.002, 0.0, 0.004])
    # Expect a ratio != 1 because edc2 decays faster
    ratio = _energy_ratio(limits, edc1, edc2)
    assert np.all(np.isfinite(ratio))
    assert not np.allclose(ratio, 1.0)
