"""
Test deprecations. For each deprecation two things must be tested:
1. Is a proper warning raised. This is done using
    import pytest
    from pyfar.classes.warnings import PyfarDeprecationWarning
    with pytest.warns(PyfarDeprecationWarning, match="some text"):
        call_of_function()
2. Was the function properly deprecated. This is done using:
    import pytest
    from packaging import version
    if version.parse(pf.__version__) >= version.parse('0.5.0'):
        with pytest.raises(AttributeError):
            # remove get_nearest_k() from pyfar 0.5.0!
            coords.get_nearest_k(1, 0, 0).
"""
import pytest
import pyrato
from packaging import version
import pyfar
from pyfar.classes.warnings import PyfarDeprecationWarning


# deprecate in 1.0.0 ----------------------------------------------------------
def test_deprecation_filter_fractional_octave_bands():
    """Test deprecation of pyrato.dsp.filter_fractional_octave_bands."""

    if version.parse(pyrato.__version__) >= version.parse('1.0.0'):
        with pytest.raises(AttributeError):
            _ = pyrato.dsp.filter_fractional_octave_bands


def test_deprecation_center_frequencies_third_octaves():
    """Test deprecation of pyrato.dsp.center_frequencies_third_octaves."""

    if version.parse(pyrato.__version__) >= version.parse('1.0.0'):
        with pytest.raises(AttributeError):
            _ = pyrato.dsp.center_frequencies_third_octaves


def test_deprecation_center_frequencies_octaves():
    """Test deprecation of pyrato.dsp.center_frequencies_octaves."""

    if version.parse(pyrato.__version__) >= version.parse('1.0.0'):
        with pytest.raises(AttributeError):
            _ = pyrato.dsp.center_frequencies_octaves


def test_deprecation_find_impulse_response_start():
    """Test deprecation of pyrato.dsp.find_impulse_response_start."""

    if version.parse(pyrato.__version__) >= version.parse('1.0.0'):
        with pytest.raises(AttributeError):
            _ = pyrato.dsp.find_impulse_response_start


# deprecate in 1.3.0 ----------------------------------------------------------
@pytest.mark.parametrize("function", [
     pyrato.edc.energy_decay_curve_truncation,
     pyrato.edc.energy_decay_curve_lundeby,
     pyrato.edc.energy_decay_curve_chu_lundeby,

])
def test_freq_parameter_deprecation(function):
    """Test deprecation of freq parameter in favor of smoothing_parameter."""

    rir = pyfar.signals.files.room_impulse_response(crop_noise_tail=False)

    # test deprecation warning
    message = ("'freq' will be deprecated in pyrato 1.3.0 in favor of "
               "'smoothing_parameter'")
    with pytest.warns(PyfarDeprecationWarning, match=message):
            _ = function(rir, freq='broadband')

    # test parameter deprecation
    message = "got an unexpected keyword argument 'freq'"
    if version.parse(pyrato.__version__) >= version.parse('1.3.0'):
            with pytest.raises(TypeError, match=message):
                _ = function
