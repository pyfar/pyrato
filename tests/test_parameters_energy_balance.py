import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
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
    result = __energy_balance(edc, 0.0, 0.005, 0.0, 0.001)
    assert isinstance(result, np.ndarray)
    assert result.shape == edc.cshape


def test_energy_balance_rejects_non_numeric_limits():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    with pytest.raises(TypeError, match="lim1 must be numeric."):
        __energy_balance(edc, "not_a_number", 1, 0, 1)


def test_energy_balance_rejects_invalid_limit_order():
    edc = make_edc_from_energy(np.linspace(1, 0, 10))
    with pytest.raises(ValueError, match="If scalars, require lim1 < lim2."):
        __energy_balance(edc, 1.0, 0.5, 0.0, 1.0)
    with pytest.raises(ValueError, match="If scalars, require lim3 < lim4."):
        __energy_balance(edc, 0.0, 1.0, 1.0, 0.5)


# --- Functional correctness ---
def test_energy_balance_computes_known_ratio_correctly():
    edc_vals = np.array([1.0, 0.75, 0.5, 0.25])
    edc = make_edc_from_energy(edc_vals, sampling_rate=1000)

    # For linear EDC: e(lim3)-e(lim4) = (1.0 - 0.75) = 0.25
    #                 e(lim1)-e(lim2) = (0.75 - 0.5) = 0.25
    # ratio = 1 -> 0 dB
    result = __energy_balance(edc, 0.001, 0.002, 0.0, 0.001)
    npt.assert_allclose(result, 0.0, atol=1e-12)


def test_energy_balance_handles_multichannel_data_correctly():
    """Ensure the output keeps channel shape."""
    energy = np.linspace(1, 0, 10)
    multi = np.stack([energy, energy * 0.5])
    edc = make_edc_from_energy(multi)
    result = __energy_balance(edc, 0.0, 0.005, 0.0, 0.001)
    assert result.shape == edc.cshape


def test_energy_balance_returns_nan_for_zero_denominator():
    """If denominator e(lim1)-e(lim2)=0, expect NaN (log10 invalid)."""
    energy = np.ones(10)
    edc = make_edc_from_energy(energy)
    result = __energy_balance(edc, 0.0, 0.001, 0.002, 0.003)
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

    # Analytical ratio:
    # e(t) = exp(-a t)
    # numerator = e(lim3) - e(lim4) = exp(-a*lim3) - exp(-a*lim4)
    # denominator = e(lim1) - e(lim2) = exp(-a*lim1) - exp(-a*lim2)
    analytical_ratio = (
        (np.exp(-a*lim3) - np.exp(-a*lim4)) /
        (np.exp(-a*lim1) - np.exp(-a*lim2))
    )
    expected_db = 10 * np.log10(analytical_ratio)

    result = __energy_balance(edc, lim1, lim2, lim3, lim4)
    npt.assert_allclose(result, expected_db, atol=1e-8)