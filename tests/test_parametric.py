"""Tests for parametric related things."""

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


def test_calculate_sabine_reverberation_time():
    alphas = [0.9, 0.1]
    surfaces = [2, 5*2]
    volume = 2*2*2
    assert ra.parametric.calculate_sabine_reverberation_time(surfaces,
            alphas, volume) == 0.46

def test_sabine_array_size_match():
    message = "Size of alphas and surfaces ndarray sizes must match."
    alphas = [0.9, 0.1, 0.3]
    surfaces = [2, 5*2]
    volume = 2*2*2
    with pytest.raises(ValueError, match=message):
        ra.parametric.calculate_sabine_reverberation_time(surfaces,
                                                          alphas, volume)

def test_sabine_alphas_outside_range():
    message=r"Absorption coefficient values must be in range"
    alphas = [2, 0]
    surfaces = [1, 1]
    volume = 2*2*2
    with pytest.raises(ValueError, match = message):
        ra.parametric.calculate_sabine_reverberation_time(surfaces,
                                                          alphas, volume)

def test_sabine_negative_surfaces():
    message = r"Surface areas cannot be negative"
    alphas = [1, 0]
    surfaces = [-1, 1]
    volume = 2*2*2
    with pytest.raises(ValueError, match=message):
        ra.parametric.calculate_sabine_reverberation_time(surfaces,
                                                        alphas, volume)

def test_sabine_negative_volume():
    message=r"Volume cannot be negative."
    alphas = [1, 0]
    surfaces = [1, 1]
    volume = -3
    with pytest.raises(ValueError, match=message):
        ra.parametric.calculate_sabine_reverberation_time(surfaces,
                                                          alphas, volume)

def test_sabine_zero_absorption_area():
    alphas = [1, 0]
    surfaces = [0, 1]
    volume = 3
    with pytest.raises(ZeroDivisionError):
        ra.parametric.calculate_sabine_reverberation_time(surfaces,
                                                        alphas, volume)
