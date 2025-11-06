import numpy.testing as npt
import pytest
import pyrato.parametric as parametric

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
