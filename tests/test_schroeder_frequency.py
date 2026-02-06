import numpy as np
import pytest
from pyrato.parametric import schroeder_frequency

def test_schroeder_frequency_scalar():
    """Test with scalar float inputs."""
    f_s = schroeder_frequency(100.0, 1.0)
    expected = 2000 * np.sqrt(1.0 / 100.0)
    assert np.isclose(f_s, expected)


def test_schroeder_frequency_array():
    """Test with matching array inputs."""
    volumes = np.array([100.0, 200.0, 400.0])
    reverb_times = np.array([1.0, 2.0, 4.0])
    f_s = schroeder_frequency(volumes, reverb_times)
    expected = 2000 * np.sqrt(reverb_times / volumes)
    np.testing.assert_allclose(f_s, expected,rtol=1e-5)


def test_schroeder_frequency_broadcasting():
    """Test with scalar and array combination (broadcasting)."""
    volume = 100.0
    reverb_times = np.array([0.5, 1.0, 2.0])
    with pytest.raises(ValueError, match="volume and reverberation_time " \
    "must have compatible shapes, either same shape or one is scalar."):
        schroeder_frequency(volume, reverb_times)


@pytest.mark.parametrize(("volume", "reverb_time"), [(0, 1.0),
                                                  (-10, 1.0),
                                                  (100, 0),
                                                  (100, -2)])
def test_invalid_values(volume, reverb_time):
    """Test that non-positive values raise ValueError."""
    with pytest.raises(ValueError, match="volume and reverberation_time " \
    "must be positiv"):
        schroeder_frequency(volume, reverb_time)


def test_shape_mismatch():
    """Test that mismatched shapes raise ValueError."""
    volumes = np.array([100, 200])
    reverb_times = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="volume and reverberation_time " \
    "must have compatible shapes, either same shape or one is scalar."):
        schroeder_frequency(volumes, reverb_times)


@pytest.mark.parametrize(("volume", "reverb_time"), [
    ("abc", 1.0),
    (100, "1.0"),
    (None, 1.0),
])
def test_invalid_types(volume, reverb_time):
    """Test that non-numeric inputs raise TypeError."""
    with pytest.raises(TypeError):
        schroeder_frequency(volume, reverb_time)
