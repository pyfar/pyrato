import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
from pyrato.parameters import clarity
import numpy.testing as npt
import re


#fixture implementation clarity tests
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


def test_clarity_geometric_decay_solution(make_edc_from_energy):
    sampling_rate = 1000
    decay_factor = 0.9
    total_samples = 200
    early_cutoff = 80  # ms

    edc = make_edc_from_energy(case="geometric", sampling_rate=sampling_rate, decay_factor=decay_factor, total_samples=total_samples)

    squared_factor = decay_factor ** 2
    early_energy = (1 - squared_factor ** early_cutoff) / (1 - squared_factor)
    late_energy = (
        squared_factor**early_cutoff - squared_factor**total_samples
    ) / (1 - squared_factor)
    expected_db = 10 * np.log10(early_energy / late_energy)

    result = clarity(edc, early_time_limit=early_cutoff)
    np.testing.assert_allclose(result, expected_db, atol=1e-6)

def test_clarity_values_for_given_ratio(make_edc_from_energy):
    energy_early = 1
    energy_late = .5
    energy = np.zeros((3, 1000))
    edc = make_edc_from_energy(energy=energy, sampling_rate=1000)
    edc.time[..., 10] = energy_early
    edc.time[..., 100] = energy_late
    edc = ra.edc.schroeder_integration(edc, is_energy=True)
    edc = pf.dsp.normalize(edc, reference_method='max')
    result = clarity(edc, early_time_limit=80)
    clarity_value_db = 10 * np.log10(energy_early/energy_late)
    npt.assert_allclose(result, clarity_value_db, atol=1e-6)

def test_clarity_from_truth_edc(make_edc_from_energy):
    # real-EDC from test_edc:test_edc_eyring
    edc = make_edc_from_energy(case="eyring analytical",  sampling_rate=250)

    real_edc = edc.time
    times = edc.times

    te = 0.08  # 80 ms
    idx = np.argmin(np.abs(times - te))
    edc_val = real_edc[0, idx]

    early_energy = real_edc[0, 0] - edc_val
    late_energy = edc_val
    expected_c80 = 10 * np.log10(early_energy / late_energy)

    result = clarity(edc, early_time_limit=80)
    np.testing.assert_allclose(result, expected_c80, atol=1e-6)



@pytest.mark.parametrize(
    'tx', ['T15', 'T20', 'T30', 'T40', 'T50', 'T60', 'LDT', 'EDT'])
def test_rt_from_edc(tx):
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est = ra.parameters.reverberation_time_linear_regression(
        edc_exp, T=tx)
    npt.assert_allclose(RT_est, 1.)


@pytest.mark.parametrize(
    'tx', ['T15', 'T20', 'T30', 'T40', 'T50', 'T60', 'LDT', 'EDT'])
def test_rt_from_edc_mulitchannel(tx):
    times = np.linspace(0, 1.5, 2**9)
    Ts = np.array([1, 2, 1.5])
    m = -60
    edc = np.atleast_2d(m/Ts).T @ np.atleast_2d(times)
    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est = ra.parameters.reverberation_time_linear_regression(
        edc_exp, T=tx)
    npt.assert_allclose(RT_est, Ts)


@pytest.mark.parametrize(
    'tx', ['T15', 'T20', 'T30', 'T40', 'T50', 'T60', 'LDT', 'EDT'])
def test_rt_from_edc_mulitchannel_amplitude(tx):
    times = np.linspace(0, 5/2, 2**9)
    Ts = np.array([[1, 2, 1.5], [3, 4, 5]])
    As = np.array([[0, 3, 6], [1, 1, 1]])
    m = -60
    edc = np.zeros((*Ts.shape, times.size))
    for idx in np.ndindex(Ts.shape):
        edc[idx] = As[idx] + m*times/Ts[idx]

    edc_exp = pf.TimeData(10**(edc/10), times)
    RT_est, A_est = ra.parameters.reverberation_time_linear_regression(
        edc_exp, T=tx, return_intercept=True)
    npt.assert_allclose(RT_est, Ts)
    npt.assert_allclose(A_est, 10**(As/10))


def test_rt_from_edc_error():
    times = np.linspace(0, 1.5, 2**9)
    m = -60
    edc = times * m
    edc_exp = pf.TimeData(10**(edc/10), times)
    T = 'Bla'

    with pytest.raises(ValueError, match='is not a valid interval.'):
        ra.parameters.reverberation_time_linear_regression(edc_exp, T=T)
