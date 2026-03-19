"""Tests for parametric related things."""

import numpy as np
import numpy.testing as npt
import pytest
import pyrato.parametric as parametric
from pyrato.parametric import mean_free_path
import pyrato as ra


@pytest.mark.parametrize(("volume","reverberation_time","expected_critical_distance"),
        [(100, 1, 0.57),(200, 2, 0.57),(50, 0.5, 0.57),(150, 3, 0.403050865)])
def test_critical_distance_calculate(volume, reverberation_time,
                                     expected_critical_distance):
    critical_distance = parametric.critical_distance(
        volume, reverberation_time)
    npt.assert_allclose(critical_distance,
                        expected_critical_distance,
                        atol=1e-2)


@pytest.mark.parametrize(("volume","reverberation_time"),
        [(100, 0),(0, 2),(0, 0),(-20, 3),(20, -1)])
def test_critical_distance_error(volume, reverberation_time):
    with pytest.raises(ValueError, match='must be greater than zero.'):
        parametric.critical_distance(volume, reverberation_time)


@pytest.mark.parametrize(
        ('volume', 'surface_area'),
        [(512, 384), (1, 11), (1, 12345)])
def test_mean_free_path(volume, surface_area):
    result = mean_free_path(volume, surface_area)
    assert result > 0

def test_mean_free_path_wrong_volume():
    with pytest.raises(ValueError, match="is smaller than 0."):
        mean_free_path(-1, 100)

def test_mean_free_path_wrong_surface_area():
    with pytest.raises(ValueError, match="is smaller than 0."):
        mean_free_path(100, -1)


def test_reverberation_time_sabine():
    volume = 2 * 2 * 2
    surface_area = 2 + 5 * 2
    mean_absorption = (2 * 0.9 + 5 * 2 * 0.1) / surface_area
    npt.assert_allclose(
        ra.parametric.reverberation_time_sabine(
            volume, surface_area, mean_absorption),
        0.46,
        atol=1e-2,
    )

def test_sabine_speed_of_sound_non_positive():
    message = "Speed of sound should be larger than 0"
    volume = 8
    surface_area = 12
    mean_absorption = 0.2
    with pytest.raises(ValueError, match=message):
        ra.parametric.reverberation_time_sabine(
            volume, surface_area, mean_absorption, speed_of_sound=0)

def test_sabine_alphas_outside_range():
    message = r"mean_absorption should be between 0 and 1"
    volume = 8
    surface_area = 2
    mean_absorption = [2, 0]
    with pytest.raises(ValueError, match=message):
        ra.parametric.reverberation_time_sabine(
            volume, surface_area, mean_absorption)

def test_sabine_negative_surfaces():
    message = r"Surface area should be larger than 0"
    volume = 8
    surface_area = -1
    mean_absorption = 0.5
    with pytest.raises(ValueError, match=message):
        ra.parametric.reverberation_time_sabine(
            volume, surface_area, mean_absorption)

def test_sabine_negative_volume():
    message = r"Volume should be larger than 0"
    volume = -3
    surface_area = 2
    mean_absorption = 0.5
    with pytest.raises(ValueError, match=message):
        ra.parametric.reverberation_time_sabine(
            volume, surface_area, mean_absorption)

def test_sabine_zero_absorption_area():
    mean_absorption = 0
    surface_area = 1
    volume = 3
    npt.assert_equal(
        ra.parametric.reverberation_time_sabine(
            volume, surface_area, mean_absorption),
        np.inf,
    )
