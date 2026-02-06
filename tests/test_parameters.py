import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
import numpy.testing as npt
import re

from pyrato.parameters import clarity
from pyrato.parameters import _energy_ratio
from pyrato.edc import energy_decay_curve_chu

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
@pytest.mark.parametrize(
    "limits",
    [[0.0, 0.001, 0.0, 0.005],
    (0.0, 0.001, 0.0, 0.005),
    np.array([0.0, 0.001, 0.0, 0.005])],
)
def test_energy_ratio_accepts_timedata_and_limits_returns_correct_shape(limits,
                                                                        make_edc):
    """Test return shape of pyfar.TimeData and accepted limits input types."""
    energy = np.linspace((1,0.5),(0,0),1000).T
    edc = make_edc(energy=energy, sampling_rate=1000)
    result = _energy_ratio(limits, edc, edc)
    assert isinstance(result, np.ndarray)
    assert result.shape == edc.cshape

def test_energy_ratio_rejects_non_timedata_input():
    """Reject wrong input type of EDC."""
    invalid_input = np.arange(10)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    expected_message = "energy_decay_curve1 must be a pyfar.TimeData " \
    "or derived object."
    with pytest.raises(TypeError, match=expected_message):
        _energy_ratio(limits, invalid_input, invalid_input)

def test_energy_ratio_rejects_if_second_edc_is_not_timedata(make_edc):
    """Reject if second EDC is of wrong type."""
    edc = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    with pytest.raises(
        TypeError,
        match="energy_decay_curve2 must be a pyfar.TimeData",
    ):
        _energy_ratio(limits, edc, "invalid_type")

def test_energy_ratio_rejects_wrong_shape_limits(make_edc):
    """Limits array wrong shape."""
    edc = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    wrong_shape_limits = np.array([0.0, 0.001, 0.005])  # Only 3 elements
    with pytest.raises(ValueError, match="limits must have shape"):
        _energy_ratio(wrong_shape_limits, edc, edc)

def test_energy_ratio_rejects_wrong_type_limits(make_edc):
    """Rejects wrong limits type correctly."""
    edc = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    wrong_type_limits = "3, 2, 0.5, 1"  # string
    with pytest.raises(TypeError,
                       match="limits must be a numpy ndarray"):
        _energy_ratio(wrong_type_limits, edc, edc)

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

def test_energy_ratio_np_inf_limits(make_edc):
    """
    Check if np.inf limits are handled correctly.
    """
    energy = [1,0,0,0] # four samples (Dirac)
    edc = make_edc(energy=energy, sampling_rate = 1.0) #sampling rate = 1 sec

    # For linear EDC:
    # should yield 0 - 0 / 1 - 0 = 0
    limits = np.array([0, np.inf, np.inf, np.inf])
    result = _energy_ratio(limits, edc, edc)
    npt.assert_allclose(result, 0.0, atol=1e-12)

@pytest.mark.parametrize(
    "energy",
    [
        # 1D, single channel
        np.linspace(1, 0, 10),
        # 2D, two channels
        np.stack([
            np.linspace(1, 0, 10),
            np.linspace(0.5, 0, 10),
        ]),
        # 3D – deterministic 2×3×4 “multichannel” structure
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
    ],
)
def test_energy_ratio_preserves_multichannel_shape_correctly(energy, make_edc):
    """Preserves any multichannel shape (1,), (2,), (2,3,)."""
    edc = make_edc(energy=energy, sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.0, 0.003])

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
    r"""
    Analytical reference:
    EDC = exp(-a*t). For exponential decay, ratio known analytically from
    .. math::
        ER = \frac{
            \displaystyle e(lim3) - e(lim4)
        }{
            \displaystyle e(lim1) - e(lim2)
        }.
    where :math:`[lim1, ..., lim4]` are the time limits and here
    the energy ratio is efficiently computed from the EDC :math:`e(t)'.
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
    Energy ratio between two different EDCs should compute distinct ratio.
    """
    edc1 = make_edc(energy=np.linspace(1, 0, 10), sampling_rate=1000)
    edc2 = make_edc(energy=np.linspace(1, 0, 10) ** 2, sampling_rate=1000)

    limits = np.array([0.0, 0.002, 0.0, 0.004])
    # Expect a ratio != 1 because edc2 decays faster
    ratio = _energy_ratio(limits, edc1, edc2)
    assert not np.allclose(ratio, 1.0)

def test_energy_ratio_rejects_limits_outside_time_range(make_edc):
    """Limits outside valid time range are rejected."""
    edc1 = make_edc(energy=np.linspace(1, 0, 100), sampling_rate=1000)
    edc2 = make_edc(energy=np.linspace(1, 0, 100), sampling_rate=1000)
    max_time = edc1.times[-1]

    # Test negative limit
    limits_negative = np.array([-0.01, 0.02, 0.02, 0.05])
    with pytest.raises(
        ValueError,
        match=r"limits\[0:2\] must be between 0 and",
    ):
        _energy_ratio(limits_negative, edc1, edc2)

    # Test limit beyond signal length
    limits_too_large = np.array([0.0, 0.02, 0.02, max_time + 0.01])
    with pytest.raises(
        ValueError,
        match=r"limits\[2:4\] must be between 0 and",
    ):
        _energy_ratio(limits_too_large, edc1, edc2)

def test_energy_ratio_handles_different_edc_lengths(make_edc):
    """Validation uses the shorter EDC's time range."""
    edc1 = make_edc(energy=np.linspace(1, 0, 100), sampling_rate=1000)
    edc2 = make_edc(energy=np.linspace(1, 0, 50), sampling_rate=1000)

    # Limit valid for edc1 but not edc2
    limits = np.array([0.0, 0.02, 0.02, 0.06])  # 0.06s > edc2.times[-1]

    with pytest.raises(
        ValueError,
        match=r"limits\[2:4\] must be between 0 and",
    ):
        _energy_ratio(limits, edc1, edc2)

def test_energy_ratio_with_clarity(make_edc):
    """
    Test for _energy_ratio in-use of a RAP-function to check if an edc
    ending with NaN is handled correctly.
    """
    energy = np.ones(1000)
    energy[900:] = np.nan #last ~100ms elements np.nan
    edc = make_edc(energy=energy, sampling_rate=1000)
    early_time_limit_sec = 0.08

    limits = np.array([early_time_limit_sec,
                        np.inf,
                        0.0,
                        early_time_limit_sec])

    result = _energy_ratio(
        limits=limits,
        energy_decay_curve1=edc,
        energy_decay_curve2=edc,
    )
    assert not np.isnan(result)

def test_energy_ratio_with_clarity_nan_limit(make_edc):
    """
    Test for _energy_ratio in-use of a RAP-function to check if np.inf
    are hanled correctly. Should return 1.
    """
    energy = np.ones(1000)
    energy[900:] = np.nan #last ~100ms elements np.nan
    edc = make_edc(energy=energy, sampling_rate=1000)

    limits = np.array([0.0,
                        np.inf,
                        0.0,
                        np.inf])

    result = _energy_ratio(
        limits=limits,
        energy_decay_curve1=edc,
        energy_decay_curve2=edc,
    )
    assert np.allclose(result, 1.0)
