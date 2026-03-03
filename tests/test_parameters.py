import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
import numpy.testing as npt
import re

from pyrato.parameters import clarity
from pyrato.parameters import sound_strength
from pyrato.parameters import definition
from pyrato.parameters import early_lateral_energy_fraction
from pyrato.parameters import _energy_ratio
from pyrato.parameters import late_lateral_sound_level

# parameter clarity tests
@pytest.mark.parametrize(
    ("energy", "expected_shape"),
    [
        # 1D single channel
        (np.linspace(1, 0, 1000), (1,)),
        # 2D two channels
        (np.linspace((1, 0.5), (0, 0), 1000).T, (2,)),
        # 3D multichannel (2x3 channels)
        (np.arange(2 * 3 * 1000).reshape(2, 3, 1000), (2, 3)),
    ],
)
def test_clarity_accepts_timedata_and_returns_correct_shape(
    energy, expected_shape, make_edc,
):
    """Test return shape and type of pyfar.TimeData input."""
    edc = make_edc(energy=energy, sampling_rate=1000)
    result = clarity(edc)
    assert isinstance(result, (float, np.ndarray))
    assert result.shape == expected_shape
    assert result.shape == edc.cshape

def test_clarity_rejects_non_numeric_early_time_limit(make_edc):
    """Rejects non-number type early_time_limit."""
    edc = make_edc()
    invalid_time_limit = "not_a_number"
    expected_error_message = "early_time_limit must be a number."

    with pytest.raises(TypeError, match=re.escape(expected_error_message)):
        clarity(edc, invalid_time_limit)

def test_clarity_returns_nan_for_zero_signal():
    """Correct return of NaN for zero signal."""
    edc = pf.TimeData(np.zeros((1, 128)), np.arange(128) / 1000)
    result = clarity(edc)
    assert np.isnan(result)

def test_clarity_calculates_known_reference_value(make_edc):
    """
    Linear decay → early_time_limit at 1/2 energy -> ratio = 1 -> 0 dB.
    Monotonic decay, 1 sample = 1ms.
    """
    edc_vals = np.array([1.0, 0.75, 0.5, 0.0])
    edc = make_edc(energy=edc_vals, sampling_rate=1000)

    result = clarity(edc, early_time_limit=2)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)

def test_clarity_values_for_given_ratio(make_edc):
    """Clarity validation for a given ratio from analytical baseline."""
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
    npt.assert_allclose(result, clarity_value_db, atol=1e-5)

def test_clarity_for_exponential_decay(make_edc):
    """Clarity validation for analytical solution from exponential decay."""
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

def test_strength_returns_zero_db_for_identical_edcs(make_edc):
    """Return 0 dB when room and reference EDCs are identical."""
    energy = np.array([1.0, 0.8, 0.4, 0.2])
    edc_room= make_edc(energy=energy, sampling_rate=1000, normalize=False)
    result = sound_strength(edc_room, edc_room)
    npt.assert_allclose(result, 0.0, atol=1e-12)

def test_strength_matches_known_reference_ratio(make_edc):
    """Calculate correct strength for a known energy ratio (2:1 → +3.01 dB)."""
    edc_room= make_edc(
        energy=np.array([2.0, 1.0, 0.0, 0.0]),
        sampling_rate=1000,
        normalize=False,
        dynamic_range=200,
    )
    edc_free_field = make_edc(
        energy=np.array([1.0, 0.5, 0.0, 0.0]),
        sampling_rate=1000,
        normalize=False,
        dynamic_range=200,
    )

    expected = 10 * np.log10(2.0)
    result = sound_strength(edc_room, edc_free_field)
    npt.assert_allclose(result, expected, atol=1e-8)

def test_strength_preserves_multichannel_shape(make_edc):
    """Preserve multichannel shape and compute strength per channel
    independently.
    """
    energy_omni = np.stack([
        np.array([1.0, 0.8, 0.4, 0.2]),
        np.array([2.0, 1.6, 0.8, 0.4]),
    ])
    energy_free_field = np.stack([
        np.array([1.0, 0.8, 0.4, 0.2]),
        np.array([1.0, 0.8, 0.4, 0.2]),
    ])
    edc_room= make_edc(
        energy=energy_omni, sampling_rate=1000, normalize=False)
    edc_free_field = make_edc(
        energy=energy_free_field, sampling_rate=1000, normalize=False)

    result = sound_strength(edc_room, edc_free_field)

    assert result.shape == edc_room.cshape
    npt.assert_allclose(result[0], 0.0, atol=1e-12)
    npt.assert_allclose(result[1], 10*np.log10(2.0), atol=1e-8)

def test_strength_handles_very_short_edcs(make_edc):
    """Handle very short EDCs when integrating over [0, inf]."""
    edc_room= make_edc(
        energy=np.array([2.0, 1.0]), sampling_rate=1000, normalize=False)
    edc_free_field = make_edc(
        energy=np.array([1.0, 0.5]), sampling_rate=1000, normalize=False)

    result = sound_strength(edc_room, edc_free_field)
    npt.assert_allclose(result, 10*np.log10(2.0), atol=1e-12)

def test_sound_strength_scales_correctly_with_single_edc_scaling(make_edc):
    """Scaling only one EDC changes G by the expected dB offset.

    - Scaling edc_roomby factor k  →  G increases by 10*log10(k).
    - Scaling edc_free_field by k   →  G decreases by 10*log10(k).
    """
    room = np.array([2.0, 1.0, 0.4, 0.0])
    ref  = np.array([1.0, 0.5, 0.2, 0.0])
    factor = 4.0
    expected_offset = 10 * np.log10(factor)   # ≈ 6.02 dB

    edc_room= make_edc(energy=room, sampling_rate=1000, normalize=False)
    edc_ref  = make_edc(energy=ref,  sampling_rate=1000, normalize=False)

    baseline = sound_strength(edc_room, edc_ref)

    # Only omni scaled → G should rise by expected_offset
    edc_room_scaled = make_edc(
        energy=factor * room, sampling_rate=1000, normalize=False)
    result_omni_scaled = sound_strength(edc_room_scaled, edc_ref)
    npt.assert_allclose(result_omni_scaled, baseline + expected_offset,
                        atol=1e-8)

    # Only free_field scaled → G should drop by expected_offset
    edc_ref_scaled = make_edc(
        energy=factor * ref, sampling_rate=1000, normalize=False)
    result_ref_scaled = sound_strength(edc_room, edc_ref_scaled)
    npt.assert_allclose(result_ref_scaled, baseline - expected_offset,
                        atol=1e-8)

def test_sound_strength_returns_nan_for_zero_denominator_signal():
    """Correct return of NaN for division by zero signal."""
    edc_room= pf.TimeData(np.ones((1, 128)), np.arange(128) / 1000)
    edc_free_field = pf.TimeData(np.zeros((1, 128)), np.arange(128) / 1000)
    with pytest.warns(RuntimeWarning):
        result = sound_strength(edc_room, edc_free_field)
    assert np.isnan(result)

# parameters definition tests
@pytest.mark.parametrize(
    ("energy", "expected_shape"),
    [
        # 1D single channel
        (np.linspace(1, 0, 1000), (1,)),
        # 2D two channels
        (np.linspace((1, 0.5), (0, 0), 1000).T, (2,)),
        # 3D multichannel (2x3 channels)
        (np.arange(2 * 3 * 1000).reshape(2, 3, 1000), (2, 3)),
    ],
)
def test_definition_accepts_timedata_and_returns_correct_shape(
    energy, expected_shape, make_edc,
):
    """Test return shape and type of pyfar.TimeData input."""
    edc = make_edc(energy=energy, sampling_rate=1000)
    result = definition(edc)
    assert isinstance(result, (float, np.ndarray))
    assert result.shape == expected_shape
    assert result.shape == edc.cshape

def test_definition_rejects_non_numeric_early_time_limit(make_edc):
    """Rejects non-number type early_time_limit."""
    edc = make_edc()
    invalid_time_limit = "not_a_number"
    expected_error_message = "early_time_limit must be a number."

    with pytest.raises(TypeError, match=re.escape(expected_error_message)):
        definition(edc, invalid_time_limit)

def test_definition_returns_zero_for_zero_signal(make_edc):
    """
    Definition must return 0.0 for a zero-energy signal.

    The fixture clips all values to a small noise floor (min_energy),
    so the EDC is flat. The denominator becomes min_energy - 0 (since
    np.inf maps to zero in _energy_ratio), and the numerator becomes
    min_energy - min_energy = 0. The result is therefore 0.0, not NaN.
    """
    edc = make_edc(energy=np.zeros(128), sampling_rate=1000)
    result = definition(edc)
    assert np.isclose(result, 0.0)

def test_definition_calculates_known_reference_value(make_edc):
    """
    Linear decay → early_time_limit at index 2 -> ratio = 0.5
    Monotonic decay, 1 sample = 1ms.
    """
    edc_vals = np.array([1.0, 0.75, 0.5, 0.25, 0.0])  # monotonic decay
    edc = make_edc(energy=edc_vals[np.newaxis, :], sampling_rate=1000)

    result = definition(edc, early_time_limit=2)
    expected = 0.5
    np.testing.assert_allclose(result, expected, atol=1e-5)

def test_definition_for_exponential_decay(make_edc):
    """Definition validation for analytical solution from exponential decay."""
    rt60 = 2.0  # seconds
    sampling_rate = 1000
    total_samples = 2000
    early_cutoff = 80  # ms

    # Generate EDC
    edc = make_edc(rt=rt60,
                   sampling_rate=sampling_rate,
                   total_samples=total_samples)
    result = definition(edc, early_time_limit=early_cutoff)

    # Analytical expected value
    te = early_cutoff / 1000  # convert ms to seconds
    a = 13.8155 / rt60
    expected_ratio = 1- np.exp(-a * te)
    np.testing.assert_allclose(result, expected_ratio, atol=1e-5)


def test_definition_values_for_given_ratio(make_edc):
    """Definition validation for a given ratio from analytical baseline."""
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
    result = definition(edc, early_time_limit=80)
    definition_value = energy_early/(energy_late+energy_early)
    np.testing.assert_allclose(result, definition_value, atol=1e-5)

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
    "multichannel_energy",
    [
        # 1D, single channel
        np.linspace(1, 0, 10),
        # 2D, two channels
        np.stack([
            np.linspace(1, 0, 10),
            np.linspace(0.5, 0, 10),
        ]),
        # 3D – deterministic 2×3×4 "multichannel" structure
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
    ],
    ids=["1D_single_channel", "2D_two_channels", "3D_deterministic"],
)
def test_energy_ratio_preserves_multichannel_shape_correctly(
    multichannel_energy,
    make_edc,
):
    """Preserves any multichannel shape (1,), (2,), (2,3,)."""
    edc = make_edc(energy=multichannel_energy, sampling_rate=1000)
    limits = np.array([0.0, np.inf, 0.0, 0.003])

    result = _energy_ratio(limits, edc, edc)

    assert result.shape == edc.cshape

@pytest.mark.parametrize(
    "multichannel_energy",
    [
        np.linspace(1, 0, 10),
        np.stack([
            np.linspace(1, 0, 10),
            np.linspace(0.5, 0, 10),
        ]),
        np.arange(2 * 3 * 4).reshape(2, 3, 4),
    ],
    ids=["1D", "2D", "3D"],
)
@pytest.mark.parametrize(
    "limits_config",
    [
        (np.array([0.0, 0.003, 0.0, np.inf]), "infinite_numerator"),
        (np.array([0.0, np.inf, 0.0, 0.003]), "infinite_denominator"),
        (np.array([0.0, np.inf, 0.0, np.inf]), "infinite_both"),
    ],
    ids=["numerator_inf", "denominator_inf", "both_inf"],
)
def test_energy_ratio_infinite_limits_multichannel(
    multichannel_energy,
    make_edc,
    limits_config,
):
    """
    Handle infinite limits in various combinations with multichannel EDCs.

    Tests three scenarios:
    - Infinite numerator limit only (lim4 = ∞)
    - Infinite denominator limit only (lim2 = ∞)
    - Both infinite limits (lim2 = ∞ and lim4 = ∞)

    """
    limits, _description = limits_config
    edc = make_edc(energy=multichannel_energy, sampling_rate=1000)

    result = _energy_ratio(limits, edc, edc)

    assert result.shape == edc.cshape

def test_energy_ratio_returns_nan_for_zero_denominator(make_edc):
    """If denominator e(lim1)-e(lim2)=0, expect NaN (invalid ratio)."""
    energy = np.ones(10)
    edc = make_edc(energy=energy, sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.002, 0.003])
    with pytest.warns(
        RuntimeWarning, match="invalid value encountered in divide",
    ):
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

def test_energy_ratio_handles_different_channel_shapes(make_edc):
    """
    Test that _energy_ratio raises an error for EDCs with different channel
    shapes.
    """
    # Create EDC with cshape (2,)
    energy1 = np.stack([
        np.linspace(1, 0, 100),
        np.linspace(0.8, 0, 100),
    ])
    edc1 = make_edc(energy=energy1, sampling_rate=1000)
    assert edc1.cshape == (2,)

    # Create EDC with cshape (3,)
    energy2 = np.stack([
        np.linspace(1, 0, 100),
        np.linspace(0.8, 0, 100),
        np.linspace(0.6, 0, 100),
    ])
    edc2 = make_edc(energy=energy2, sampling_rate=1000)
    assert edc2.cshape == (3,)

    # Limits for both numerator and denominator
    limits = np.array([0.0, 0.02, 0.0, 0.05])

    # Should raise ValueError due to shape mismatch with clear message
    with pytest.raises(
        ValueError,
        match="energy_decay_curve1 and energy_decay_curve2 must have the same "
              "channel shape",
    ):
        _energy_ratio(limits, edc1, edc2)


# parameter early lateral energy fraction (JLF) tests
@pytest.mark.parametrize(
    ("energy", "expected_shape"),
    [
        # 1D single channel
        (np.linspace(1, 0, 1000), (1,)),
        # 2D two channels
        (np.linspace((1, 0.5), (0, 0), 1000).T, (2,)),
        # 3D multichannel (2x3 channels)
        (np.arange(2 * 3 * 1000).reshape(2, 3, 1000), (2, 3)),
    ],
)
def test_JLF_accepts_timedata_and_returns_correct_shape(
    energy, expected_shape, make_edc,
):
    """Return type and shape of pyfar.TimeData input for identical edcs."""
    edc = make_edc(energy=energy, sampling_rate=1000)
    result = early_lateral_energy_fraction(edc, edc)
    assert isinstance(result, (float, np.ndarray))
    assert result.shape == expected_shape
    assert result.shape == edc.cshape

def test_JLF_returns_nan_for_zero_denominator_signal():
    """Correct return of NaN for division by zero signal."""
    edc1 = pf.TimeData(np.ones((1, 128)), np.arange(128) / 1000)
    edc2 = pf.TimeData(np.zeros((1, 128)), np.arange(128) / 1000)
    result = early_lateral_energy_fraction(edc1, edc2)
    assert np.isnan(result)

def test_JLF_calculates_known_reference_value(make_edc):
    """
    Construct simple deterministic EDCs:
    e(0) = 1
    e(0.08) = 0
    e_L(0.005) = 0.5
    e_L(0.08) = 0
    Expected:
    JLF = (0.5 / 1) = 0.5.
    """
    pad = np.zeros(100)
    edc_omni = np.concatenate((
        np.array([1.0]),
        pad,
    ))
    edc_lateral = np.concatenate((
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
        pad,
    ))

    edc_omni = make_edc(energy=edc_omni, sampling_rate=1000)
    edc_lateral = make_edc(energy=edc_lateral,
                           sampling_rate=1000, normalize=False)

    result = early_lateral_energy_fraction(edc_omni, edc_lateral)
    np.testing.assert_allclose(result, 0.5, atol=1e-5)

def test_JLF_is_within_ISO3382_range(make_edc):
    """
    Smoke test: J_LF must fall within the empirically observed range
    reported in ISO 3382 for concert halls: 0.05 to 0.35 (linear).

    Signal design:
    - Onset delay of 3 ms applied manually to both EDCs (co-located mics).
    - Omni RT = 2.0 s, full amplitude.
    - Lateral RT = 2.1 s (slightly longer), amplitude scaled to 0.25
      reflecting the figure-eight mic's reduced sensitivity to the
      direct sound.
    """
    sampling_rate = 44100
    total_samples = 200000
    onset_samples = int(0.003 * sampling_rate)  # 3 ms

    # Build exponential decay manually and prepend onset zeros
    t = np.arange(total_samples - onset_samples) / sampling_rate
    decay_omni    = np.exp(-13.8155 / 2.0 * t)
    decay_lateral = 0.25 * np.exp(-13.8155 / 2.1 * t)

    zeros = np.zeros(onset_samples)
    energy_lateral = make_edc(energy=np.concatenate([zeros, decay_lateral]),
                           sampling_rate=sampling_rate, normalize=False)
    energy_omni = make_edc(energy=np.concatenate([zeros, decay_omni]),
                           sampling_rate=sampling_rate, normalize=False)

    edc_lateral = ra.edc.schroeder_integration(energy_lateral, is_energy=True)
    edc_omni = ra.edc.schroeder_integration(energy_omni, is_energy=True)

    result = early_lateral_energy_fraction(edc_omni, edc_lateral).item()

    # ISO 3382 typical range: 0.05–0.35
    assert 0.05 <= result <= 0.35, (
        f"J_LF = {result:.2f} is outside the ISO 3382 typical range "
        f"[0.05, 0.35]"
    )

def test_JLF_for_exponential_decay_analytical(make_edc):
    """
    JLF validation for analytical solution from exponential decay.

    For an exponential EDC: e(t) = exp(-a*t), where a = 13.8155 / RT

    JLF = (
        (e_L(0.005) - e_L(0.08)) /   # lateral, 5ms to 80ms
        (e_omni(0)  - e_omni(0.08))   # omni,    0ms to 80ms
    )
    """
    sampling_rate = 1000
    total_samples = 2000

    edc_omni = make_edc(rt=2.0, sampling_rate=sampling_rate,
                        total_samples=total_samples)
    edc_lateral = make_edc(rt=2.2, sampling_rate=sampling_rate,
                           total_samples=total_samples)

    result = early_lateral_energy_fraction(edc_omni, edc_lateral)

    # Decay constants
    a_omni = 13.8155 / 2.0
    a_lat  = 13.8155 / 2.2

    # Denominator: e_omni(0) - e_omni(0.08).
    expected_omni = np.exp(-a_omni * 0.0) - np.exp(-a_omni * 0.08)
    # Numerator: e_L(0.005) - e_L(0.08).
    expected_lateral = np.exp(-a_lat * 0.005) - np.exp(-a_lat * 0.08)

    expected = expected_lateral / expected_omni
    np.testing.assert_allclose(result, expected, atol=1e-5)


# late_lateral_level tests
@pytest.mark.parametrize(
    ("energy", "expected_shape"),
    [
        (np.linspace(1, 0, 1000), (1,)),
        (np.linspace((1, 0.5), (0, 0), 1000).T, (2,)),
        (np.arange(2 * 3 * 1000).reshape(2, 3, 1000), (2, 3)),
    ],
)
def test_LJ_accepts_timedata_and_returns_correct_shape(
    energy, expected_shape, make_edc,
):
    """Return type and shape of pyfar.TimeData input for identical edcs."""
    edc = make_edc(energy=energy, sampling_rate=1000)
    result = late_lateral_sound_level(edc, edc)

    assert isinstance(result, (float, np.ndarray))
    assert result.shape == expected_shape
    assert result.shape == edc.cshape

def test_LJ_returns_nan_for_zero_denominator_signal():
    """Correct return of NaN for division by zero signal."""
    edc_ref = pf.TimeData(np.zeros((1, 128)), np.arange(128) / 1000)
    edc_lat = pf.TimeData(np.ones((1, 128)), np.arange(128) / 1000)

    result = late_lateral_sound_level(edc_ref, edc_lat)
    assert np.isnan(result)

def test_LJ_calculates_known_reference_value(make_edc):
    """
    Construct simple deterministic EDCs:
    e_10(0)   = 1
    e_L(0.08) = 0.5
    Expected:
    LJ = 10*log10(0.5) = -3.0103 dB.
    """

    pad = np.zeros(200)

    edc_ref = np.concatenate(([1.0], pad))

    # 80 ms at 1 kHz sampling rate -> index 80
    edc_lateral = np.concatenate((
        np.zeros(80),
        np.array([0.5]),
        pad,
    ))

    edc_ref = make_edc(energy=edc_ref, sampling_rate=1000)
    edc_lateral = make_edc(energy=edc_lateral,
                           sampling_rate=1000, normalize=False)

    result = late_lateral_sound_level(edc_ref, edc_lateral)

    expected = 10 * np.log10(0.5)
    np.testing.assert_allclose(result, expected, atol=1e-5)

def test_LJ_for_exponential_decay_analytical(make_edc):
    """
    LJ validation for analytical exponential decay:
    e(t) = exp(-a*t)
    LJ = 10*log10(e_L(0.08) / e_10(0)).
    """

    sampling_rate = 1000
    total_samples = 2000

    # emulating free field impulse response
    edc_ref = make_edc(rt=0.01,
                       sampling_rate=sampling_rate,
                       total_samples=total_samples)

    edc_lat = make_edc(rt=2.2,
                       sampling_rate=sampling_rate,
                       total_samples=total_samples)

    result = late_lateral_sound_level(edc_ref, edc_lat)

    a_lat = 13.8155 / 2.2

    expected = 10 * np.log10(np.exp(-a_lat * 0.08))

    np.testing.assert_allclose(result, expected, atol=1e-5)
