import numpy as np
import pytest
import pyfar as pf
import numpy.testing as npt
import re

from pyrato.parameters import __energy_balance


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
    result = __energy_balance(0.0, 0.005, 0.0, 0.001, edc, edc)
    assert isinstance(result, np.ndarray)
    assert result.shape == edc.cshape


def test_energy_balance_rejects_non_timedata_input():
    invalid_input = np.arange(10)
    expected_message = "pyfar.TimeData"
    with pytest.raises(TypeError, match=expected_message):
        __energy_balance(0.0, 0.005, 0.0, 0.001, invalid_input, invalid_input)


def test_energy_balance_rejects_if_second_edc_is_not_timedata():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    with pytest.raises(TypeError, match="pyfar.TimeData"):
        __energy_balance(0.0, 0.005, 0.0, 0.001, edc, "invalid_type")


def test_energy_balance_rejects_non_numeric_limits():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    with pytest.raises(TypeError, match="lim1 must be numeric."):
        __energy_balance("not_a_number", 1, 0, 1, edc, edc)


def test_energy_balance_rejects_invalid_limit_order():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    with pytest.raises(ValueError, match="If scalars, require lim1 < lim2."):
        __energy_balance(1.0, 0.5, 0.0, 1.0, edc, edc)
    with pytest.raises(ValueError, match="If scalars, require lim3 < lim4."):
        __energy_balance(0.0, 1.0, 1.0, 0.5, edc, edc)


# --- Functional correctness ---
def test_energy_balance_computes_known_ratio_correctly():
    """If EDC is linear, energy balance ratio should be 1."""
    edc_vals = np.array([1.0, 0.75, 0.5, 0.25])
    edc = make_edc_from_energy(edc_vals, sampling_rate=1000)

    # For linear EDC: numerator = e(lim3)-e(lim4) = (1.0 - 0.75) = 0.25
    #                 denominator = e(lim1)-e(lim2) = (0.75 - 0.5) = 0.25
    # ratio = 1
    result = __energy_balance(0.001, 0.002, 0.0, 0.001, edc, edc)
    npt.assert_allclose(result, 1.0, atol=1e-12)


def test_energy_balance_handles_multichannel_data_correctly():
    energy = np.linspace(1, 0, 10)
    multi = np.stack([energy, energy * 0.5])
    edc = make_edc_from_energy(multi)
    result = __energy_balance(0.0, 0.005, 0.0, 0.001, edc, edc)
    assert result.shape == edc.cshape


def test_energy_balance_returns_nan_for_zero_denominator():
    """If denominator e(lim1)-e(lim2)=0, expect NaN (invalid ratio)."""
    energy = np.ones(10)
    edc = make_edc_from_energy(energy)
    result = __energy_balance(0.0, 0.001, 0.002, 0.003, edc, edc)
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

    lim1, lim2, lim3, lim4 = 0.0, 0.05, 0.0, 0.02

    analytical_ratio = (
        (np.exp(-a*lim3) - np.exp(-a*lim4)) /
        (np.exp(-a*lim1) - np.exp(-a*lim2))
    )

    result = __energy_balance(lim1, lim2, lim3, lim4, edc, edc)
    npt.assert_allclose(result, analytical_ratio, atol=1e-8)


def test_energy_balance_works_with_two_different_edcs():
    """Energy balance between two different EDCs should compute distinct ratio."""
    times = np.linspace(0, 0.009, 10)
    edc1 = pf.TimeData(np.linspace(1, 0, 10)[np.newaxis, :], times)
    edc2 = pf.TimeData((np.linspace(1, 0, 10) ** 2)[np.newaxis, :], times)

    # Expect a ratio != 1 because edc2 decays faster
    ratio = __energy_balance(0.0, 0.004, 0.0, 0.002, edc1, edc2)
    assert np.all(np.isfinite(ratio))
    assert not np.allclose(ratio, 1.0)
