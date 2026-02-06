#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for reverberation time related things."""
import numpy as np
import numpy.testing as npt

import pyrato as ra
import pytest

@pytest.mark.parametrize('T',[0,20,1000])
def test_speed_of_sound(T):
    speed=ra.parametric.calculate_speed_of_sound(T)
    assert isinstance(T, (int, float))
    npt.assert_allclose(speed, 343.2 * np.sqrt((T + 273.15) / 293.15))
    with pytest.raises(ValueError,match=
                       "Temperature must be greater than absolute zero"):
        ra.parametric.calculate_speed_of_sound(-300)
