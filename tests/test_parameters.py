import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
import numpy.testing as npt
import re

import os
from pyfar import Signal, signals
from pyrato.parameters import speech_transmission_index_indirect
from pyrato.parameters import modulation_transfer_function
from pyrato.parameters import _sti_calc


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


def test_sti_data_input():
    """
    TypeError is raised when input data is not a pyfar.Signal.
    """
    sig = np.zeros(100)
    match="Input data must be a pyfar.Signal."
    with pytest.raises(TypeError, match=match):
        speech_transmission_index_indirect(sig)

def test_sti_snr_value_error():
    """
    ValueError is raised when SNR consists
    of the wrong number of components.
    """
    sig = Signal(np.zeros(70560), 44100)
    snr = np.zeros(6)  # Incorrect: only 6 bands instead of 7
    match = r"Input 'snr' must have shape.*7 octave bands"
    with pytest.raises(ValueError, match=match):
        speech_transmission_index_indirect(sig, snr=snr)

def test_sti_level_value_error():
    """
    ValueError is raised when level consists
    of the wrong number of components.
    """
    sig = Signal(np.zeros(70560), 44100)
    level = np.zeros(6)  # Incorrect: only 6 bands instead of 7
    match = r"Input 'level' must have shape.*7 octave bands"
    with pytest.raises(ValueError, match=match):
        speech_transmission_index_indirect(sig, level=level)

def test_sti_snr_warning():
    """
    UserWarning is raised when SNR is less than 20 dB for every octave band.
    """
    sig = Signal(np.zeros(70560), 44100)
    snr = np.ones(7) * 10  # SNR less than 20 dB
    match = "Input 'snr' should be at least 20 dB for every octave band."
    with pytest.warns(UserWarning, match=match):
        speech_transmission_index_indirect(sig, snr=snr)

def test_sti_level_warning():
    """
    UserWarning is raised when level is less than 1 dB for every octave band.
    """
    sig = Signal(np.zeros(70560), 44100)
    level = np.ones(7) * 0.5  # level less than 1 dB
    match = "Input 'level' should be at least 1 dB for every octave band."
    with pytest.warns(UserWarning, match=match):
        speech_transmission_index_indirect(sig, level=level)

def test_sti_warn_length():
    """
    ValueError is raised when the input signal is less than 1.6 seconds long.
    """
    sig = Signal(np.ones((31072)), 44100)
    match  = "Input signal must be at least 1.6 seconds long."
    with pytest.raises(ValueError, match=match):
        speech_transmission_index_indirect(sig)

def test_sti_warn_data_type_unknown():
    """
    ValueError is raised when an unknown data type is given.
    """
    sig = Signal(np.zeros(70560), 44100)
    with pytest.raises(ValueError, match="Data_type is 'generic' but must "
                       "be 'electrical' or 'acoustical'."):
        speech_transmission_index_indirect(sig, rir_type="generic")

def test_sti_ambient_noise_type_error():
    """
    TypeError is raised when ambient_noise is not a boolean.
    """
    sig = Signal(np.zeros(70560), 44100)
    match = "ambient_noise must be a boolean."
    with pytest.raises(TypeError, match=match):
        speech_transmission_index_indirect(sig, ambient_noise="yes")

def test_sti_1D_shape():
    """
    Output shape is correct for a 1D input signal.
    """
    shape_expected = (2,)
    sig = Signal(np.ones((2,70560)), 44100)
    array = speech_transmission_index_indirect(sig, rir_type="acoustical")
    assert array.shape == shape_expected

def test_sti_2D_shape():
    """
    Output shape is correct for a 2D input signal.
    """
    shape_expected = (2,2)
    sig = Signal(np.ones((2,2,70560)), 44100)
    sti_test = speech_transmission_index_indirect(sig, rir_type="acoustical")
    assert sti_test.shape == shape_expected

def test_sti_snr_shape_broadcast():
    """
    SNR with shape (7,) is correctly broadcast to all channels.
    """
    sig = Signal(np.ones((2, 70560)), 44100)
    snr = np.ones(7) * 30  # Shape (7,) should work for any channel count
    sti_test = speech_transmission_index_indirect(sig, snr=snr)
    assert sti_test.shape == (2,)

def test_sti_level_shape_broadcast():
    """
    Level with shape (7,) is correctly broadcast to all channels.
    """
    sig = Signal(np.ones((2, 70560)), 44100)
    level = np.ones(7) * 60  # Shape (7,) should work for any channel count
    snr = np.ones(7) * 30
    sti_test = speech_transmission_index_indirect(sig, level=level, snr=snr)
    assert sti_test.shape == (2,)

def test_sti_multichannel_different_snr_level():
    """
    Different channels can have different SNR and level values.
    """
    sig = Signal(np.ones((2, 70560)), 44100)
    # Channel 0: high SNR/level, Channel 1: lower SNR/level
    snr = np.array([
        [40, 40, 40, 40, 40, 40, 40],  # Channel 0: high SNR
        [10, 10, 10, 10, 10, 10, 10],  # Channel 1: low SNR
    ])
    level = np.array([
        [70, 70, 70, 70, 70, 70, 70],  # Channel 0: high level
        [50, 50, 50, 50, 50, 50, 50],  # Channel 1: lower level
    ])

    with pytest.warns(UserWarning, match="Input 'snr' should be at least 20 dB"):
        sti_test = speech_transmission_index_indirect(
            sig, level=level, snr=snr)

    # Different SNR/level should produce different STI values
    assert sti_test.shape == (2,)
    assert sti_test[0] != sti_test[1]
    # Higher SNR/level should generally produce higher STI
    assert sti_test[0] > sti_test[1]

def test_sti_unit_impuls():
    """
    STI value for a unit impulse signal.
    Ideal case: STI = 1.
    """
    sti_expected = 1
    sig = signals.impulse(70560)
    sti_test = speech_transmission_index_indirect(sig, rir_type="acoustical")
    np.testing.assert_allclose(sti_test, sti_expected,atol=0.01)

def test_sti_ir():
    """
    STI value for a simulated IR.
    Compare with WinMF - Measurement Software.
    """
    sti_expected =  0.86
    time = np.loadtxt(os.path.join(
        os.path.dirname(__file__), "test_data", "ir_simulated.csv"))
    ir = Signal(time, 44100)
    sti_test = speech_transmission_index_indirect(ir, rir_type="acoustical")
    np.testing.assert_allclose(sti_test, sti_expected,atol=0.01)

def test_sti_ir_level_snr():
    """
    STI value for a simulated IR.
    Considered level and snr values.
    Compare with WinMF - Measurement Software.
    """

    sti_expected =  0.62
    level = np.array([54, 49 , 54, 48, 45,40 , 31])
    noise_level = np.array([53, 48, 46, 42, 38, 34, 30])
    snr = level - noise_level
    time = np.loadtxt(os.path.join(
        os.path.dirname(__file__), "test_data", "ir_simulated.csv"))
    ir = Signal(time, 44100)
    with pytest.warns(UserWarning, match="Input 'snr' should be at least 20 dB"):
        sti_test = speech_transmission_index_indirect(
            ir, rir_type="acoustical", level=level, snr=snr)
    np.testing.assert_allclose(sti_test, sti_expected, atol=0.01)

def test_sti_electrical_vs_acoustical():
    """
    STI for electrical signals differs from acoustical due to masking effects.
    """
    sig = signals.impulse(70560)
    level = np.ones(7) * 60
    snr = np.ones(7) * 30

    sti_acoustical = speech_transmission_index_indirect(
        sig, rir_type="acoustical", level=level, snr=snr)
    sti_electrical = speech_transmission_index_indirect(
        sig, rir_type="electrical", level=level, snr=snr)

    # Electrical should have higher STI (no masking)
    assert sti_electrical >= sti_acoustical

def test_sti_ambient_noise_effect():
    """
    Ambient noise correction affects STI values.
    """
    sig = signals.impulse(70560)
    level = np.ones(7) * 60
    snr = np.ones(7) * 20

    sti_with_noise = speech_transmission_index_indirect(
        sig, rir_type="acoustical", level=level, snr=snr, ambient_noise=True)
    sti_without_noise = speech_transmission_index_indirect(
        sig, rir_type="acoustical", level=level, snr=snr, ambient_noise=False)

    # Ambient noise correction reduces STI
    assert sti_with_noise != sti_without_noise
    assert sti_without_noise > sti_with_noise

def test_mtf_data_input():
    """
    TypeError is raised when input data is not a pyfar.Signal.
    """
    sig = np.zeros(70560)
    snr = np.ones(7) * 30
    match = "Input data must be a pyfar.Signal."
    with pytest.raises(TypeError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=None,
            snr=snr, ambient_noise=True)

def test_mtf_multichannel_error():
    """
    ValueError is raised when input has more than one channel.
    """
    sig = Signal(np.ones((2, 70560)), 44100)
    snr = np.ones(7) * 30
    match = "Input must be a single-channel impulse response"
    with pytest.raises(ValueError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=None,
            snr=snr, ambient_noise=True)

def test_mtf_short_signal_error():
    """
    ValueError is raised when signal is shorter than 1.6 seconds.
    """
    sig = Signal(np.ones(44100), 44100)  # 1 second
    snr = np.ones(7) * 30
    match = "Input signal must be at least 1.6 seconds long"
    with pytest.raises(ValueError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=None,
            snr=snr, ambient_noise=True)

def test_mtf_snr_type_error():
    """
    TypeError is raised when snr is not a numpy array.
    """
    sig = signals.impulse(70560)
    snr = [30, 30, 30, 30, 30, 30, 30]  # List instead of array
    match = "snr must be a numpy array."
    with pytest.raises(TypeError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=None,
            snr=snr, ambient_noise=True)

def test_mtf_snr_shape_error():
    """
    ValueError is raised when snr has wrong shape.
    """
    sig = signals.impulse(70560)
    snr = np.ones(6)  # Wrong: 6 bands instead of 7
    match = re.escape('snr must have shape (7,) for 7 octave bands')
    with pytest.raises(ValueError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=None,
            snr=snr, ambient_noise=True)

def test_mtf_level_type_error():
    """
    TypeError is raised when level is not a numpy array or None.
    """
    sig = signals.impulse(70560)
    snr = np.ones(7) * 30
    level = [60, 60, 60, 60, 60, 60, 60]  # List instead of array
    match = "level must be a numpy array or None."
    with pytest.raises(TypeError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=level,
            snr=snr, ambient_noise=True)

def test_mtf_level_shape_error():
    """
    ValueError is raised when level has wrong shape.
    """
    sig = signals.impulse(70560)
    snr = np.ones(7) * 30
    level = np.ones(6) * 60  # Wrong: 6 bands instead of 7
    match = r"Level must have shape \(7,\) for 7 octave bands"
    with pytest.raises(ValueError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=level,
            snr=snr, ambient_noise=True)

def test_mtf_ambient_noise_type_error():
    """
    TypeError is raised when ambient_noise is not a boolean.
    """
    sig = signals.impulse(70560)
    snr = np.ones(7) * 30
    match = "ambient_noise must be a boolean."
    with pytest.raises(TypeError, match=match):
        modulation_transfer_function(
            sig, rir_type="acoustical", level=None,
            snr=snr, ambient_noise="yes")

def test_mtf_shape():
    """
    MTF output shape is (7, 14) for a valid impulse response.
    """
    sig = signals.impulse(70560)
    snr = np.ones(7) * 30

    mtf = modulation_transfer_function(
        sig, rir_type="acoustical", level=None, snr=snr, ambient_noise=True)

    assert mtf.shape == (7, 14)

def test_mtf_snr_reduction():
    """
    MTF is reduced when SNR decreases.
    """
    sig = signals.impulse(70560)

    mtf_high = modulation_transfer_function(
        sig,
        rir_type="acoustical",
        level=None,
        snr=np.ones(7) * 100,
        ambient_noise=False,
    )

    with pytest.warns(UserWarning, match="Input 'snr' should be at least 20 dB"):
        mtf_low = modulation_transfer_function(
            sig,
            rir_type="acoustical",
            level=None,
            snr=np.ones(7) * 10,
            ambient_noise=False,
        )

    assert np.all(mtf_low < mtf_high)

def test_mtf_ambient_noise_effect():
    """
    Ambient noise correction reduces MTF values.
    """
    sig = signals.impulse(70560)
    level = np.ones(7) * 60
    snr = np.ones(7) * 20

    mtf_no_amb = modulation_transfer_function(
        sig, "acoustical", level=level, snr=snr, ambient_noise=False,
    )

    mtf_amb = modulation_transfer_function(
        sig, "acoustical", level=level, snr=snr, ambient_noise=True,
    )

    assert np.all(mtf_amb <= mtf_no_amb)

def test_mtf_electrical_vs_acoustical():
    """
    Auditory masking is applied only for acoustical signals.
    """
    sig = signals.impulse(70560)
    level = np.ones(7) * 65
    snr = np.ones(7) * 20

    mtf_ac = modulation_transfer_function(
        sig, "acoustical", level=level, snr=snr, ambient_noise=True,
    )

    mtf_el = modulation_transfer_function(
        sig, "electrical", level=level, snr=snr, ambient_noise=True,
    )

    assert np.any(mtf_ac != mtf_el)

def test_mtf_bounds():
    """
    MTF values are bounded between 0 and 1.
    """
    sig = signals.impulse(70560)
    snr = np.ones(7) * 5

    with pytest.warns(UserWarning, match="Input 'snr' should be at least 20 dB"):
        mtf = modulation_transfer_function(
            sig, "acoustical", level=None, snr=snr, ambient_noise=True,
        )

    assert np.all(mtf >= 0.0)
    assert np.all(mtf <= 1.0)
    """
    TypeError is raised when mtf is not a numpy array.
    """
    mtf = [[0.5] * 14] * 7  # List instead of array
    match = "mtf must be a numpy array."
    with pytest.raises(TypeError, match=match):
        _sti_calc(mtf)

def test_sti_calc_mtf_shape_error():
    """
    ValueError is raised when mtf has wrong shape.
    """
    mtf = np.ones((6, 14))  # Wrong: 6 bands instead of 7
    match = (
        r"mtf must have shape \(7, 14\) for 7 octave bands "
        r"and 14 modulation frequencies"
    )
    with pytest.raises(ValueError, match=match):
        _sti_calc(mtf)

def test_sti_calc_mtf_zero_clipping():
    """
    _sti_calc correctly handles MTF=0 (SNR=-inf) by clipping to -15 dB.
    """
    # MTF = 0 leads to log10(0 / (1-0)) = log10(0) = -inf
    mtf = np.zeros((7, 14))
    sti = _sti_calc(mtf)

    # With SNR clipped to -15 dB, TI = (-15 + 15)/30 = 0, STI should be 0
    assert sti == 0.0

def test_sti_calc_mtf_one_clipping():
    """
    _sti_calc correctly handles MTF=1 (SNR=+inf) by clipping to 15 dB.
    """
    # MTF = 1 leads to log10(1 / (1-1)) = log10(1/0) = +inf
    mtf = np.ones((7, 14))
    sti = _sti_calc(mtf)

    # With SNR clipped to 15 dB, TI = (15 + 15)/30 = 1, STI should be 1
    assert sti == 1.0

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

@pytest.mark.parametrize(
    "energy",
    [
        # 1D, einzelner Kanal
        np.linspace(1, 0, 10),
        # 2D, zwei Kanäle
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

    # Ergebnis muss exakt dieselbe Kanalform haben wie das EDC
    assert result.shape == edc.cshape

def test_energy_ratio_returns_nan_for_zero_denominator(make_edc):
    """If denominator e(lim1)-e(lim2)=0, expect NaN (invalid ratio)."""
    energy = np.ones(10)
    edc = make_edc(energy=energy, sampling_rate=1000)
    limits = np.array([0.0, 0.001, 0.002, 0.003])
    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        result = _energy_ratio(limits, edc, edc)
    assert np.isnan(result)
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
