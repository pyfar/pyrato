import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
from pyrato.parameters import clarity
import numpy.testing as npt
import re


def test_fixture_returns_timedata_object(make_edc_from_energy):
    edc = make_edc_from_energy()
    assert isinstance(edc, pf.TimeData)
    assert edc.time.ndim >= 1  # has time dimension
    assert np.all(edc.time.shape)  # not empty

def test_fixture_preserves_input_shape(make_edc_from_energy):
    custom_energy = np.ones((2, 3, 10))
    edc = make_edc_from_energy(energy=custom_energy)
    assert edc.time.shape == custom_energy.shape

@pytest.mark.parametrize("invalid_energy", ["not_an_array", 5, None])
def test_fixture_rejects_invalid_energy_type(make_edc_from_energy,
                                             invalid_energy):
    if invalid_energy is None:
        pytest.skip("None is allowed and defaults to baseline")
    with pytest.raises(TypeError):
        make_edc_from_energy(energy=invalid_energy)

def test_fixture_handles_zero_energy(make_edc_from_energy):
    snr = 90  # -90 dBFS
    edc = make_edc_from_energy(energy=np.zeros(10), dynamic_range=snr)
    assert np.allclose(edc.time, 10 ** (-snr / 10))

def test_fixture_generates_correct_exponential_decay(make_edc_from_energy):
    rt, sr, n = 2.0, 1000, 1000
    edc = make_edc_from_energy(rt=rt, sampling_rate=sr, total_samples=n)
    t = np.arange(n) / sr
    expected = np.exp(-13.8155 * t / rt)[np.newaxis, :]  # match shape
    np.testing.assert_allclose(edc.time, expected / np.max(expected),
                               rtol=1e-7)

def test_fixture_returns_baseline_edc_when_no_inputs(make_edc_from_energy):
    edc = make_edc_from_energy()
    assert np.allclose(edc.time, edc.time[0])  # constant
    assert np.all(edc.time >= 0)

@pytest.mark.parametrize("normalize", [True, False])
def test_fixture_normalization_behavior(make_edc_from_energy, normalize):
    energy = np.array([0.5, 0.5, 0.4])[np.newaxis, :]
    edc = make_edc_from_energy(energy=energy, normalize=normalize)
    if normalize:
        assert np.isclose(np.max(edc.time), 1.0)
    else:
        assert not np.isclose(np.max(edc.time), 1.0)

def test_fixture_multichannel_normalization(make_edc_from_energy):
    energy = np.array([[1.0, 0.5, 0.25], [0.2, 0.1, 0.05]])
    edc = make_edc_from_energy(energy=energy, normalize=True)
    assert np.isclose(np.max(edc.time), 1.0)
    assert edc.time.shape == energy.shape

def test_fixture_respects_sampling_rate(make_edc_from_energy):
    sr, n = 500, 100
    edc = make_edc_from_energy(rt=1.0, sampling_rate=sr, total_samples=n)
    np.testing.assert_allclose(edc.times, np.arange(n) / sr)

@pytest.mark.parametrize("invalid_rt", [-1, 0, "fast"])
def test_fixture_rejects_invalid_rt(make_edc_from_energy, invalid_rt):
    with pytest.raises((ValueError, TypeError)):
        make_edc_from_energy(rt=invalid_rt)

def test_fixture_accepts_raw_energy_data(make_edc_from_energy):
    custom = np.linspace(1, 0, 100)
    edc = make_edc_from_energy(energy=custom)
    # Last sample may be clamped to dynamic range limit
    expected = np.clip(custom, 10 ** (-65 / 10), None)
    np.testing.assert_allclose(edc.time, expected[np.newaxis, :], rtol=1e-7)


def test_clarity_accepts_timedata_returns_correct_type(make_edc_from_energy):
    energy = np.concatenate(([1, 1, 1, 1], np.zeros(124)))
    edc = make_edc_from_energy(energy=energy)

    result = clarity(edc, early_time_limit=4)  # 4 ms
    assert isinstance(result, (float, np.ndarray))
    assert result.shape == edc.cshape

def test_clarity_rejects_non_timedata_input():
    invalid_input = np.array([1, 2, 3])
    expected_error_message = "Input must be a pyfar.TimeData object."

    with pytest.raises(TypeError, match=re.escape(expected_error_message)):
        clarity(invalid_input)

def test_clarity_rejects_non_numeric_early_time_limit(make_edc_from_energy):
    edc = make_edc_from_energy()
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

def test_clarity_rejects_invalid_time_range(make_edc_from_energy):
    energy = np.zeros(128)
    edc = make_edc_from_energy(energy=energy)
    actual_signal_length_ms = edc.signal_length * 1000

    # Test negative time limit
    expected_error_message = "early_time_limit must be in the range of 0"
    f"and {actual_signal_length_ms}."
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        clarity(edc, early_time_limit=-1)

    # Test time limit beyond signal length
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        clarity(edc, early_time_limit=200000)

def test_clarity_preserves_multichannel_shape(make_edc_from_energy):
    energy = np.ones((2,2,10)) / (1+np.arange(10))
    edc = make_edc_from_energy(energy=energy, sampling_rate=10)
    output = clarity(edc, early_time_limit=80)
    assert edc.cshape == output.shape


def test_clarity_returns_nan_for_zero_signal():
    edc = pf.TimeData(np.zeros((1, 128)), np.arange(128) / 1000)
    result = clarity(edc)
    assert np.isnan(result)


def test_clarity_calculates_known_reference_value(make_edc_from_energy):
    # Linear decay → early_time_limit at 1/2 energy -> ratio = 1 -> 0 dB
    edc_vals = np.array([1.0, 0.75, 0.5, 0.0])  # monotonic decay
    edc = make_edc_from_energy(energy=edc_vals, sampling_rate=1000)

    result = clarity(edc, early_time_limit=2)
    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_clarity_values_for_given_ratio(make_edc_from_energy):
    energy_early = 1
    energy_late = .5
    energy = np.zeros((3, 1000))
    edc = make_edc_from_energy(energy=energy,
                               sampling_rate=1000,
                               dynamic_range = 120.0)
    edc.time[..., 10] = energy_early
    edc.time[..., 100] = energy_late
    edc = ra.edc.schroeder_integration(edc, is_energy=True)
    edc = pf.dsp.normalize(edc, reference_method='max')
    result = clarity(edc, early_time_limit=80)
    clarity_value_db = 10 * np.log10(energy_early/energy_late)
    npt.assert_allclose(result, clarity_value_db, atol=1e-6)

def test_clarity_for_exponential_decay(make_edc_from_energy):
    rt60 = 2.0  # seconds
    sampling_rate = 1000
    total_samples = 2000
    early_cutoff = 80  # ms

    # Generate EDC
    edc = make_edc_from_energy(rt=rt60,
                               sampling_rate=sampling_rate,
                               total_samples=total_samples)
    result = clarity(edc, early_time_limit=early_cutoff)

    # Analytical expected value
    te = early_cutoff / 1000  # convert ms to seconds
    a = 13.8155 / rt60
    expected_ratio = np.exp(a * te) - 1
    expected_dB = 10 * np.log10(expected_ratio)
    np.testing.assert_allclose(result, expected_dB, atol=1e-6)
