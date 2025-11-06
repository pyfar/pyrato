import numpy as np
from numpy import array

import numpy.testing as npt
import pyfar as pf
import pytest

import pyrato as ra

def test_calculate_sabine_reverberation_time():
    alphas = [0.9, 0.1]
    surfaces = [2, 5*2]
    volume = 2*2*2
    assert ra.parametric.calculate_sabine_reverberation_time(surfaces, alphas, volume) == 0.46

def test_sabine_array_size_match():
    with pytest.raises(ValueError, match="Size of alphas and surfaces ndarray sizes must match."):
        alphas = [0.9, 0.1, 0.3]
        surfaces = [2, 5*2]
        volume = 2*2*2
        ra.parametric.calculate_sabine_reverberation_time(surfaces, alphas, volume)

def test_sabine_alphas_outside_range():
    with pytest.raises(ValueError, match=r"Absorption coefficient values must be in range"):
        alphas = [2, 0]
        surfaces = [1, 1]
        volume = 2*2*2
        ra.parametric.calculate_sabine_reverberation_time(surfaces, alphas, volume)

def test_sabine_negative_surfaces():
    with pytest.raises(ValueError, match=r"Surface areas cannot be negative"):
        alphas = [1, 0]
        surfaces = [-1, 1]
        volume = 2*2*2
        ra.parametric.calculate_sabine_reverberation_time(surfaces, alphas, volume)

def test_sabine_negative_volume():
   with pytest.raises(ValueError, match=r"Volume cannot be negative."):
       alphas = [1, 0]
       surfaces = [1, 1]
       volume = -3
       ra.parametric.calculate_sabine_reverberation_time(surfaces, alphas, volume)
       
def test_sabine_zero_absorption_area():
   with pytest.raises(ZeroDivisionError):
       alphas = [1, 0]
       surfaces = [0, 1]
       volume = 3
       ra.parametric.calculate_sabine_reverberation_time(surfaces, alphas, volume)
