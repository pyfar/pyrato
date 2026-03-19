import pytest
import pyrato
from packaging import version


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
