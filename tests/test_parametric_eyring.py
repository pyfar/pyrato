"""Tests for Eyring's equation."""
"""Tests for Eyring's equation."""
import numpy.testing as npt
import pytest
from pyrato.parametric import reverberation_time_eyring
import numpy as np

@pytest.mark.parametrize(
        ("mean_absorption", "expected"),
        [
            ([0.2, 0.2], [0.481, 0.481]),
            (1, 0),
            (0, np.inf),
            ([0, 1, 0], [np.inf, 0, np.inf]),
        ])
def test_analytic_Eyring(mean_absorption, expected):
import numpy as np

@pytest.mark.parametrize(
        ("mean_absorption", "expected"),
        [
            ([0.2, 0.2], [0.481, 0.481]),
            (1, 0),
            (0, np.inf),
            ([0, 1, 0], [np.inf, 0, np.inf]),
        ])
def test_analytic_Eyring(mean_absorption, expected):
    volume = 64
    surface_area = 96
    reverb_test = reverberation_time_eyring(volume, surface_area, mean_absorption)
    npt.assert_allclose(reverb_test, expected, rtol=1e-3)

@pytest.mark.parametrize(("volume", "surface_area"), [(0, -2), (-1, 0)])
def test_input_geometry_Eyring(volume, surface_area):
    mean_absorption = 0.2
    with pytest.raises(ValueError, match="should be larger than 0"):
        reverberation_time_eyring(volume, surface_area, mean_absorption)

@pytest.mark.parametrize("mean_absorption", [-1, 2])
def test_input_absorption_Eyring(mean_absorption):
    volume = 64
    surface_area = 96
    with pytest.raises(ValueError, match="should be between 0 and 1"):
        reverberation_time_eyring(volume, surface_area, mean_absorption)

def test_error_speed_of_sound_Eyring():
    volume = 64
    surface_area = 96
    mean_absorption = 0.2
    with pytest.raises(ValueError, match="should be larger than 0"):
        reverberation_time_eyring(
            volume, surface_area, mean_absorption,
            speed_of_sound=-1)
