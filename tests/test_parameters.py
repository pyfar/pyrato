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

    edc = make_edc_from_energy(case="geometric", sampling_rate=sampling_rate)

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
    real_edc = np.array([
        1.00000000e+00, 8.39817186e-01, 7.05292906e-01, 5.92317103e-01,
        4.97438083e-01, 4.17757051e-01, 3.50839551e-01, 2.94641084e-01,
        2.47444646e-01, 2.07808266e-01, 1.74520953e-01, 1.46565696e-01,
        1.23088390e-01, 1.03371746e-01, 8.68133684e-02, 7.29073588e-02,
        6.12288529e-02, 5.14210429e-02, 4.31842755e-02, 3.62668968e-02,
        3.04575632e-02, 2.55787850e-02, 2.14815032e-02, 1.80405356e-02,
        1.51507518e-02, 1.27238618e-02, 1.06857178e-02, 8.97404943e-03,
        7.53656094e-03, 6.32933340e-03, 5.31548296e-03, 4.46403394e-03,
        3.74897242e-03, 3.14845147e-03, 2.64412365e-03, 2.22058049e-03,
        1.86488165e-03, 1.56615966e-03, 1.31528780e-03, 1.10460130e-03,
        9.27663155e-04, 7.79067460e-04, 6.54274242e-04, 5.49470753e-04,
        4.61454981e-04, 3.87537824e-04, 3.25460924e-04, 2.73327678e-04,
        2.29545281e-04, 1.92776072e-04,
    ])
    times = np.linspace(0, 0.25, len(real_edc))
    edc = make_edc_from_energy(case="real", sampling_rate=len(real_edc)*4)

    te = 0.08  # 80 ms
    idx = np.argmin(np.abs(times - te))
    edc_val = real_edc[idx]

    early_energy = real_edc[0] - edc_val
    late_energy = edc_val
    expected_c80 = 10 * np.log10(early_energy / late_energy)

    result = clarity(edc, early_time_limit=80)
    np.testing.assert_allclose(result, expected_c80, atol=1e-6)
