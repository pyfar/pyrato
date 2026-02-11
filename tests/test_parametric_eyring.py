"""Tests for Eyring's equation"""
import numpy.testing as npt
import pytest
from pyrato.parametric import reverberation_time_eyring

def test_analytic_Eyring():
    volume = 64
    surface = 96
    mean_alpha = 0.2
    reverb_test = reverberation_time_eyring(volume,surface,mean_alpha)
    npt.assert_allclose(reverb_test, 0.481, rtol=1e-3)

@pytest.mark.parametrize("volume, surface", [(0, -2), (-1, 0)])
def test_input_geometry_Eyring(volume, surface):
    mean_alpha = 0.2
    with pytest.raises(ValueError, match="should be larger than 0"):
        reverberation_time_eyring(volume,surface,mean_alpha)

@pytest.mark.parametrize("mean_alpha", [-1, 2])
def test_input_alpha_Eyring(mean_alpha):
    volume = 64
    surface = 96
    with pytest.raises(
        ValueError, match="mean_alpha should be between 0 and 1"
    ):
        reverberation_time_eyring(volume,surface,mean_alpha)
