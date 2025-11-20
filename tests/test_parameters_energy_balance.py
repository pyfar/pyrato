import numpy as np
import pytest
import pyfar as pf
import numpy.testing as npt

from pyrato.parameters import _energy_ratio


def make_edc_from_energy(energy, sampling_rate=1000):
    """Helper: build normalized EDC TimeData from an energy curve."""
    energy = np.asarray(energy, dtype=float)
    energy = energy / np.max(energy) if np.max(energy) != 0 else energy
    times = np.arange(energy.shape[-1]) / sampling_rate
    if energy.ndim == 1:
        energy = energy[np.newaxis, :]
    return pf.TimeData(energy, times)


# --- Basic type and shape tests ---
def test_energy_balance_accepts_timedata_and_returns_correct_shape():
    energy = np.linspace(1, 0, 10)
    edc = make_edc_from_energy(energy)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    result = _energy_ratio(limits, edc, edc)
    assert isinstance(result, np.ndarray)
    assert result.shape == edc.cshape


def test_energy_balance_rejects_non_timedata_input():
    invalid_input = np.arange(10)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    expected_message = "pyfar.TimeData"
    with pytest.raises(TypeError, match=expected_message):
        _energy_ratio(limits, invalid_input, invalid_input)


def test_energy_balance_rejects_if_second_edc_is_not_timedata():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    with pytest.raises(TypeError, match="pyfar.TimeData"):
        _energy_ratio(limits, edc, "invalid_type")


def test_energy_balance_rejects_non_numpy_array_limits():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    with pytest.raises(TypeError, match="limits must be a numpy ndarray"):
        _energy_ratio([0.0, 0.001, 0.0, 0.005], edc, edc)


def test_energy_balance_rejects_wrong_shape_limits():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    wrong_shape_limits = np.array([0.0, 0.001, 0.005])  # Only 3 elements
    with pytest.raises(ValueError, match="limits must have shape"):
        _energy_ratio(wrong_shape_limits, edc, edc)

# --- Functional correctness ---
def test_energy_balance_computes_known_ratio_correctly():
    """
    If EDC is linear, energy balance ratio should be 1.

    numerator = e(lim3)-e(lim4) = (1.0 - 0.75) = 0.25
    denominator = e(lim1)-e(lim2) = (0.75 - 0.5) = 0.25
    ratio = 1
    """
    edc_vals = np.array([1.0, 0.75, 0.5, 0.25])
    edc = make_edc_from_energy(edc_vals, sampling_rate=1000)

    # For linear EDC:
    limits = np.array([0.0, 0.001, 0.001, 0.002])
    result = _energy_ratio(limits, edc, edc)
    npt.assert_allclose(result, 1.0, atol=1e-12)


def test_energy_balance_handles_multichannel_data_correctly():
    energy = np.linspace(1, 0, 10)
    multi = np.stack([energy, energy * 0.5])
    edc = make_edc_from_energy(multi)
    limits = np.array([0.0, 0.001, 0.0, 0.005])
    result = _energy_ratio(limits, edc, edc)
    assert result.shape == edc.cshape


def test_energy_balance_returns_nan_for_zero_denominator():
    """If denominator e(lim1)-e(lim2)=0, expect NaN (invalid ratio)."""
    energy = np.ones(10)
    edc = make_edc_from_energy(energy)
    limits = np.array([0.0, 0.001, 0.002, 0.003])
    result = _energy_ratio(limits, edc, edc)
    assert np.isnan(result)


def test_energy_balance_matches_reference_case():
    """
    Analytical reference:
    EDC = exp(-a*t). For exponential decay, ratio known analytically.
    """
    sampling_rate = 1000
    a = 13.8155  # decay constant
    times = np.arange(1000) / sampling_rate
    edc_vals = np.exp(-a * times)
    edc = pf.TimeData(edc_vals[np.newaxis, :], times)

    limits = np.array([0.0, 0.02, 0.0, 0.05])
    lim1, lim2, lim3, lim4 = limits

    analytical_ratio = (
        (np.exp(-a*lim3) - np.exp(-a*lim4)) /
        (np.exp(-a*lim1) - np.exp(-a*lim2))
    )

    result = _energy_ratio(limits, edc, edc)
    npt.assert_allclose(result, analytical_ratio, atol=1e-8)


def test_energy_balance_works_with_two_different_edcs():
    """
    Energy balance between two different EDCs.
    should compute distinct ratio.
    """
    times = np.linspace(0, 0.009, 10)
    edc1 = pf.TimeData(np.linspace(1, 0, 10)[np.newaxis, :], times)
    edc2 = pf.TimeData((np.linspace(1, 0, 10) ** 2)[np.newaxis, :], times)

    limits = np.array([0.0, 0.002, 0.0, 0.004])
    # Expect a ratio != 1 because edc2 decays faster
    ratio = _energy_ratio(limits, edc1, edc2)
    assert np.all(np.isfinite(ratio))
    assert not np.allclose(ratio, 1.0)
