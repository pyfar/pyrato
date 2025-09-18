import numpy as np
import pytest
import pyfar as pf
import pyrato as ra
from pyrato.parameters import clarity

def make_edc_from_energy(energy, sampling_rate=1000):
    """Helper: build normalized EDC TimeData from an energy curve."""
    energy = np.asarray(energy, dtype=float)

    if np.max(energy) == 0:
        edc_norm = energy
    else:
        edc_norm = energy / np.max(energy)

    times = np.arange(edc_norm.shape[-1]) / sampling_rate

    # ensure shape is (n_channels, n_samples)
    if edc_norm.ndim == 1:
        edc_norm = edc_norm[np.newaxis, :]

    return pf.TimeData(edc_norm, times)


def test_clarity_accepts_timedata_and_returns_correct_type():
    energy = np.concatenate(([1, 1, 1, 1], np.zeros(124)))
    edc = make_edc_from_energy(energy, sampling_rate=1000)

    result = clarity(edc, te=2)  # 2 ms
    assert isinstance(result, (float, np.ndarray))
    assert result.shape == edc.cshape


def test_clarity_rejects_non_timedata_input():
    with pytest.raises(TypeError):
        clarity(np.array([1, 2, 3]))


def test_clarity_preserves_multichannel_shape():
    energy = np.ones((2,2,10)) / (1+np.arange(10))
    edc = make_edc_from_energy(energy, rir.sampling_rate)
    output = clarity(edc, te=80)
    assert edc.cshape == output.shape


def test_clarity_returns_nan_for_zero_signal():
    edc = pf.TimeData(np.zeros((1, 128)), np.arange(128) / 1000)
    result = clarity(edc)
    assert np.isnan(result)


def test_clarity_warns_for_unusually_short_time_limit():
    energy = np.ones(128)
    edc = make_edc_from_energy(energy, sampling_rate=44100)
    with pytest.warns(UserWarning):
        clarity(edc, te=0.05)


def test_clarity_calculates_known_reference_value():
    # Linear decay → te at 1/2 energy -> ratio = 1 -> 0 dB
    edc_vals = np.array([1.0, 0.75, 0.5, 0.0])  # monotonic decay
    times = np.arange(len(edc_vals)) / 1000
    edc = pf.TimeData(edc_vals[np.newaxis, :], times)

    result = clarity(edc, te=2)
    assert np.isclose(result, 0.0, atol=1e-6)


def test_clarity_matches_analytical_geometric_decay_solution():
    sampling_rate = 1000
    decay_factor = 0.9
    total_samples = 200
    early_cutoff = 80  # ms

    time_axis = np.arange(total_samples)
    energy = decay_factor ** (2 * time_axis)  # squared amplitude
    edc = make_edc_from_energy(energy, sampling_rate=sampling_rate)

    squared_factor = decay_factor ** 2
    early_energy = (1 - squared_factor ** early_cutoff) / (1 - squared_factor)
    late_energy = (
        squared_factor**early_cutoff - squared_factor**total_samples
    ) / (1 - squared_factor)
    expected_db = 10 * np.log10(early_energy / late_energy)

    result = clarity(edc, te=early_cutoff)
    assert np.isclose(result, expected_db, atol=1e-6)


def test_clarity_from_truth_edc():
    # Truth-EDC from test_edc.py
    truth = np.array([
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
    times = np.linspace(0, 0.25, len(truth))
    edc = pf.TimeData(truth[np.newaxis, :], times)

    te = 0.08  # 80 ms
    idx = np.argmin(np.abs(times - te))
    edc_val = truth[idx]

    early_energy = truth[0] - edc_val
    late_energy = edc_val
    expected_c80 = 10 * np.log10(early_energy / late_energy)

    result = clarity(edc, te=80)
    assert np.isclose(result, expected_c80, atol=1e-6)