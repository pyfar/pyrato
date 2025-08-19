import numpy as np
import pytest
import pyfar as pf
from pyrato.parameters import clarity

# Input types, Output type
# -------------------
def test_accepts_signal_object():
    sig = pf.signals.impulse(
        n_samples=128, delay=0, amplitude=1, sampling_rate=44100
    )
    result = clarity(sig, early_time_limit=80)

    assert isinstance(result, (float, np.ndarray))
    assert result.shape == sig.cshape

def test_rejects_non_signal_input():
    with pytest.raises(AttributeError):  
        clarity(np.array([1,2,3]))



# Multichannel shape
# -------------------
def test_multichannel_shape():
    brir = pf.signals.files.binaural_room_impulse_response(diffuse_field_compensation=False, sampling_rate=48000)

    output = clarity(brir, early_time_limit=80)

    assert brir.cshape == output.shape




# Edge cases
# -------------------
def test_empty_signal_returns_nan():
    sig = pf.Signal(np.zeros(16), 44100)
    result = clarity(sig, early_time_limit=80)
    assert np.isnan(result) or result == -np.inf


def test_unusual_time_limit_warns():
    sig = pf.signals.impulse( n_samples=128, delay=0, amplitude=1, sampling_rate=44100)
    with pytest.warns(UserWarning):
        clarity(sig, early_time_limit=0.05)




# Energie- vs. Amplitudensignal
# -------------------
# def test_energy_signal_vs_amplitude_signal():
#     # gleicher Inhalt, einmal als "energy" markiert
#     sig_amp = pf.signals.impulse( n_samples=128, delay=0, amplitude=1, sampling_rate=44100)
#     sig_energy = pf.Signal(sig_amp.time**2, 44100)

#     # label "energy" automatically assigned?
#     result_amp = clarity(sig_amp, early_time_limit=80)
#     result_energy = clarity(sig_energy, early_time_limit=80)

#     # same values
#     np.testing.assert_allclose(result_amp, result_energy)


# Reference cases
# -------------------
def test_known_reference_case():
    # Artificial signal: 2 Samples early = 2 energy, 2 Samples late = 2 energy
    data = np.array([1, 1, 1, 1] + [0]*124)
    sig = pf.Signal(data, sampling_rate=1000)
    result = clarity(sig, early_time_limit=2)  # 2ms limit -> even early/late division
    # Early = Late => log10(1) = 0 dB
    assert np.isclose(result, 0.0, atol=1e-6)

def load_c80_from_rew(file_path):
    c80_values = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():  # nur Zeilen, die mit Zahl anfangen
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 16:
                continue
            try:
                c80 = float(parts[15])
                c80_values.append(c80)
            except ValueError:
                continue

    return np.array(c80_values, dtype=np.float32)


# test with reference impulse response C80
def test_reference_impulse_response_filtered():
    rir = pf.signals.files.room_impulse_response(sampling_rate=48000)

    # filter into octave bands
    rir_octave_filtered = pf.dsp.filter.fractional_octave_bands(rir, num_fractions=1)
    # rir_third_octave_filtered = pf.dsp.filter.fractional_octave_bands(rir, num_fractions=3)

    c80_rir_octave_bands = clarity(rir_octave_filtered, early_time_limit=80)

    # tolerance?
    REW_c80_rir_octave_band = load_c80_from_rew("tests/test_data/example_rir48kHz_REWdata.txt")
    
    
    assert np.allclose(REW_c80_rir_octave_band, c80_rir_octave_bands, atol=0.01)




# test with reference impulse response C50


# test with reference impulse response C1000
# what happens if early_time limit is longer than IR?
# what happens if early_time_limit is out of logical bounds?

