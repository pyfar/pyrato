import numpy as np
import pytest
import pyfar as pf


def test_fixture_returns_timedata_object(make_edc):
    edc = make_edc()
    assert isinstance(edc, pf.TimeData)
    assert edc.time.ndim >= 1  # has time dimension
    assert np.all(edc.time.shape)  # not empty

def test_fixture_preserves_input_shape(make_edc):
    custom_energy = np.ones((2, 3, 10))
    edc = make_edc(energy=custom_energy)
    assert edc.time.shape == custom_energy.shape

@pytest.mark.parametrize("invalid_energy", ["not_an_array", 5])
def test_fixture_rejects_invalid_energy_type(make_edc,
                                             invalid_energy):
    with pytest.raises(TypeError):
        make_edc(energy=invalid_energy)

def test_fixture_handles_zero_energy(make_edc):
    dynamic_range = 90  # -90 dBFS
    edc = make_edc(energy=np.zeros(10), dynamic_range=dynamic_range)
    assert np.allclose(edc.time, 10 ** (-dynamic_range / 10))

def test_fixture_generates_correct_exponential_decay(make_edc):
    rt, sr, n = 2.0, 1000, 1000
    edc = make_edc(rt=rt, sampling_rate=sr, total_samples=n)
    t = np.arange(n) / sr
    expected = np.exp(-13.8155 * t / rt)[np.newaxis, :]  # match shape
    np.testing.assert_allclose(edc.time, expected / np.max(expected),
                               rtol=1e-7)

def test_fixture_returns_baseline_edc_when_no_inputs(make_edc):
    edc = make_edc()
    assert np.allclose(edc.time, edc.time[0])  # constant
    assert np.all(edc.time >= 0)

@pytest.mark.parametrize("normalize", [True, False])
def test_fixture_normalization_behavior(make_edc, normalize):
    energy = np.array([0.5, 0.5, 0.4])[np.newaxis, :]
    edc = make_edc(energy=energy, normalize=normalize)
    if normalize:
        assert np.isclose(np.max(edc.time), 1.0)
    else:
        assert not np.isclose(np.max(edc.time), 1.0)

def test_fixture_multichannel_normalization(make_edc):
    energy = np.array([[1.0, 0.5, 0.25], [0.2, 0.1, 0.05]])
    edc = make_edc(energy=energy, normalize=True)
    assert np.isclose(np.max(edc.time), 1.0)
    assert edc.time.shape == energy.shape

def test_fixture_respects_sampling_rate(make_edc):
    sr, n = 500, 100
    edc = make_edc(rt=1.0, sampling_rate=sr, total_samples=n)
    np.testing.assert_allclose(edc.times, np.arange(n) / sr)

@pytest.mark.parametrize("invalid_rt", [-1, 0, "fast"])
def test_fixture_rejects_invalid_rt(make_edc, invalid_rt):
    with pytest.raises((ValueError, TypeError)):
        make_edc(rt=invalid_rt)

def test_fixture_accepts_raw_energy_data(make_edc):
    custom = np.linspace(1, 0, 100)
    edc = make_edc(energy=custom)
    # Last sample may be clamped to dynamic range limit
    expected = np.clip(custom, 10 ** (-65 / 10), None)
    np.testing.assert_allclose(edc.time, expected[np.newaxis, :], rtol=1e-7)
