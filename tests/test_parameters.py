import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
from pyrato.parameters import clarity
import numpy.testing as npt
import re

import os
from pyfar import Signal, signals
from pyrato.parameters import speech_transmission_index
from pyrato.parameters import modulation_transfer_function


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
        speech_transmission_index(sig)

def test_sti_snr_value_error():
    """
    ValueError is raised when SNR consists
    of the wrong number of components.
    """
    sig = Signal(np.zeros(70560), 44100)
    snr = np.zeros(6)  # Incorrect number of components
    match = "SNR consists of wrong number of components."
    with pytest.raises(ValueError, match=match):
        speech_transmission_index(sig, snr=snr)

def test_sti_snr_warning():
    """
    UserWarning is raised when SNR is less than 20 dB for every octave band.
    """
    sig = Signal(np.zeros(70560), 44100)
    snr = np.ones(7) * 10  # SNR less than 20 dB
    match = "SNR should be at least 20 dB for every octave band."
    with pytest.warns(UserWarning, match=match):
        speech_transmission_index(sig, snr=snr)

def test_sti_warn_length():
    """
    ValueError is raised when the input signal is less than 1.6 seconds long.
    """
    sig = Signal(np.ones((31072)), 44100)
    match  = "Input signal must be at least 1.6 seconds long."
    with pytest.raises(ValueError, match=match):
        speech_transmission_index(sig)

def test_sti_warn_data_type_unknown():
    """
    ValueError is raised when an unknown data type is given.
    """
    sig = Signal(np.zeros(70560), 44100)
    with pytest.raises(ValueError, match="Data_type is 'generic' but must "
                       "be 'electrical' or 'acoustical'."):
        speech_transmission_index(sig, data_type="generic")

def test_sti_1D_shape():
    """
    Output shape is correct for a 1D input signal.
    """
    shape_expected = (2,)
    sig = Signal(np.ones((2,70560)), 44100)
    array = speech_transmission_index(sig, data_type="acoustical")
    assert array.shape == shape_expected

def test_sti_2D_shape():
    """
    Output shape is correct for a 2D input signal.
    """
    shape_expected = (2,2)
    sig = Signal(np.ones((2,2,70560)), 44100)
    sti_test = speech_transmission_index(sig, data_type="acoustical")
    assert sti_test.shape == shape_expected

def test_sti_unit_impuls():
    """
    STI value for a unit impulse signal.
    Ideal case: STI = 1.
    """
    sti_expected = 1
    sig = signals.impulse(70560)
    sti_test = speech_transmission_index(sig, data_type="acoustical")
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
    sti_test = speech_transmission_index(ir, data_type="acoustical")
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
    sti_test = speech_transmission_index(ir,
                                         data_type="acoustical",
                                         level=level,snr=snr)
    np.testing.assert_allclose(sti_test, sti_expected,atol=0.01)

def test_mtf_shape():
    """
    MTF output shape is (7, 14) for a valid impulse response.
    """
    sig = signals.impulse(70560)
    snr = np.ones(7) * 30

    mtf = modulation_transfer_function(
        sig, data_type="acoustical", level=None, snr=snr, amb=True
    )

    assert mtf.shape == (7, 14)

def test_mtf_snr_reduction():
    """
    MTF is reduced when SNR decreases.
    """
    sig = signals.impulse(70560)

    mtf_high = modulation_transfer_function(
        sig,
        data_type="acoustical",
        level=None,
        snr=np.ones(7) * 100,
        amb=False,
    )

    mtf_low = modulation_transfer_function(
        sig,
        data_type="acoustical",
        level=None,
        snr=np.ones(7) * 10,
        amb=False,
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
        sig, "acoustical", level=level, snr=snr, amb=False
    )

    mtf_amb = modulation_transfer_function(
        sig, "acoustical", level=level, snr=snr, amb=True
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
        sig, "acoustical", level=level, snr=snr, amb=True
    )

    mtf_el = modulation_transfer_function(
        sig, "electrical", level=level, snr=snr, amb=True
    )

    assert np.any(mtf_ac != mtf_el)

def test_mtf_bounds():
    """
    MTF values are bounded between 0 and 1.
    """
    sig = signals.impulse(70560)
    snr = np.ones(7) * 5

    mtf = modulation_transfer_function(
        sig, "acoustical", level=None, snr=snr, amb=True
    )

    assert np.all(mtf >= 0.0)
    assert np.all(mtf <= 1.0)
