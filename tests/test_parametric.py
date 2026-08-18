"""Tests for parametric related things."""

import numpy as np
import numpy.testing as npt
import pytest
import pyrato.parametric as parametric
from pyrato.parametric import mean_free_path
import pyrato as ra
import pyrato


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


@pytest.mark.parametrize(
    'times', [
        np.linspace(0, 1, 100),
        [0, 1, 2, 3, 4, 5],
    ],
)
def test_reflection_density(times):
    """Test if average reflection density returns correct values."""
    volume = 100
    speed_of_sound = 343

    density = pyrato.parametric.average_reflection_density(
        volume,
        times,
        speed_of_sound,
    )

    times_array = np.asarray(times, dtype=float)
    reference = 4 * np.pi * speed_of_sound**3 * times_array**2 / volume

    np.testing.assert_allclose(
        np.squeeze(density.time),
        reference,
    )


def test_reflection_density_errors():
    """Test if all errors are raised correctly."""

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_reflection_density(
            volume=-1,
            times=np.linspace(0, 1, 10),
        )

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_reflection_density(
            volume=0,
            times=np.linspace(0, 1, 10),
        )

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_reflection_density(
            volume=100,
            times=np.linspace(-1, 1, 10),
        )

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_reflection_density(
            volume=100,
            times=np.linspace(0, 1, 10),
            speed_of_sound=-300,
        )


@pytest.mark.parametrize(
    'times', [
        np.linspace(0, 1, 100),
        [0, 1, 2, 3, 4, 5],
    ],
)
def test_reflection_number(times):
    """Test if function returns correct values."""
    volume = 100
    speed_of_sound = 343

    number_of_reflections = pyrato.parametric.average_number_of_reflections(
        volume,
        times,
        speed_of_sound,
    )

    times_array = np.asarray(times, dtype=float)
    reference = 4 * np.pi * speed_of_sound**3 * times_array**3 / volume / 3

    np.testing.assert_allclose(
        np.squeeze(number_of_reflections.time),
        reference,
    )


def test_reflection_number_errors():
    """Test if all errors are raised correctly."""

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_number_of_reflections(
            volume=0,
            times=np.linspace(0, 1, 10),
        )

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_number_of_reflections(
            volume=-1,
            times=np.linspace(0, 1, 10),
        )

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_number_of_reflections(
            volume=100,
            times=np.linspace(-1, 1, 10),
        )

    with pytest.raises(ValueError, match="must be positive"):
        pyrato.parametric.average_number_of_reflections(
            volume=100,
            times=np.linspace(0, 1, 10),
            speed_of_sound=-300,
        )
