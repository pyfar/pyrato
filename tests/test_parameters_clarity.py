import numpy as np
import pytest
import pyfar as pf
from pyrato.parameters import clarity


def test_clarity_accepts_signal_object_and_returns_correct_type():
    impulse_signal = pf.signals.impulse(
        n_samples=128, delay=0, amplitude=1, sampling_rate=44100
    )
    result = clarity(impulse_signal, early_time_limit=1)

    assert isinstance(result, (float, np.ndarray))
    assert result.shape == impulse_signal.cshape


def test_clarity_rejects_non_signal_input():
    with pytest.raises(AttributeError):  
        clarity(np.array([1, 2, 3]))


def test_clarity_preserves_multichannel_shape():
    brir = pf.signals.files.binaural_room_impulse_response(
        diffuse_field_compensation=False, 
        sampling_rate=48000
    )
    output = clarity(brir, early_time_limit=80)

    assert brir.cshape == output.shape


def test_clarity_returns_nan_for_zero_signal():
    silent_signal = pf.Signal(np.zeros(4096), 44100)
    result = clarity(silent_signal)
    assert np.isnan(result) or result == -np.inf


def test_clarity_warns_for_unusually_short_time_limit():
    impulse_signal = pf.signals.impulse(
        n_samples=128, delay=0, amplitude=1, sampling_rate=44100
    )
    with pytest.warns(UserWarning):
        clarity(impulse_signal, early_time_limit=0.05)


def test_clarity_calculates_known_reference_value():
    # 2 samples early energy = 2, 2 samples late energy = 2 → ratio = 1 → 0 dB
    signal_data = np.concatenate(([1, 1, 1, 1], np.zeros(124)))
    test_signal = pf.Signal(signal_data, sampling_rate=1000)
    
    result = clarity(test_signal, early_time_limit=2)
    
    assert np.isclose(result, 0.0, atol=1e-6)


def test_clarity_matches_analytical_geometric_decay_solution():
    sampling_rate = 1000
    decay_factor = 0.9
    total_samples = 200
    early_cutoff = 80  # ms

    time_axis = np.arange(total_samples)
    decaying_signal = pf.Signal(
        decay_factor ** time_axis, 
        sampling_rate=sampling_rate
    )

    squared_factor = decay_factor ** 2
    early_energy = (1 - squared_factor ** early_cutoff) / (1 - squared_factor)
    late_energy = (squared_factor ** early_cutoff - squared_factor ** total_samples) / (1 - squared_factor)
    expected_db = 10 * np.log10(early_energy / late_energy)

    result = clarity(decaying_signal, early_time_limit=early_cutoff)

    assert np.isclose(result, expected_db, atol=1e-6)