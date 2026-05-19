import pytest
import pyrato
import pyfar
import os
from packaging import version
from numpy import genfromtxt
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


# deprecate in 1.1.0 ----------------------------------------------------------
@pytest.mark.parametrize("function",
        [("pyrato.edc.energy_decay_curve_truncation(data=rir,freq='broadband')"),
        ("pyrato.edc.energy_decay_curve_chu_lundeby(data=rir,freq='broadband')"),
        ("pyrato.edc.energy_decay_curve_lundeby(data=rir,freq='broadband')"),
        ("pyrato.edc.intersection_time_lundeby(data=rir,freq='broadband')")])
def test_deprecation_edc_freq_parameter(function):
    """Test deprecation of the 'freq' parameter in the edc functions."""

    rir = pyfar.Signal(genfromtxt(
    os.path.join((os.path.join(os.path.dirname(__file__), 'test_data')),
                  'analytic_rir_psnr50_1D.csv'), delimiter=','), 3000)
    if version.parse(pyrato.__version__) >= version.parse('1.1.0'):
         with pytest.raises(AttributeError):
            eval(function)


@pytest.mark.parametrize("function",
        [("pyrato.edc.energy_decay_curve_truncation(data=rir,freq='broadband')"),
        ("pyrato.edc.energy_decay_curve_chu_lundeby(data=rir,freq='broadband')"),
        ("pyrato.edc.energy_decay_curve_lundeby(data=rir,freq='broadband')"),
        ("pyrato.edc.intersection_time_lundeby(data=rir,freq='broadband')")])
def test_deprecation_warning_edc_freq_parameter(function):
    """
    Test whether deprecation warnings are raised for the 'freq' parameter in
    the edc functions.
    """

    rir = pyfar.Signal(genfromtxt(
    os.path.join((os.path.join(os.path.dirname(__file__), 'test_data')),
                  'analytic_rir_psnr50_1D.csv'), delimiter=','), 3000)

    with pytest.warns(
        PyfarDeprecationWarning, match="'freq' will be deprecated in "
        "pyrato 1.1.0 in favor of 'smoothing_parameter'"):
            eval(function)
