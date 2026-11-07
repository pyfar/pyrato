from math import erfc
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest

from pyrato.parameters import mixing_time_abel, perceptual_mixing_time_lindau


def _noise_rir(n_samples=20000, onset=500, seed=0, rt=0.25):
    """Exponentially decaying Gaussian noise with a direct sound at ``onset``.

    The samples before ``onset`` are zero, so the onset detection returns
    ``onset`` rather than an early noise sample. The echo density of Gaussian
    noise crosses unity shortly after the onset, so the transition time is
    well defined.
    """
    rng = np.random.default_rng(seed)
    n_tail = n_samples - onset
    decay = np.exp(-np.arange(n_tail) / n_tail / rt)
    rir = np.zeros(n_samples)
    rir[onset:] = rng.standard_normal(n_tail) * decay
    rir[onset] += 10.0
    return rir


def _measured_rir():
    time = np.loadtxt('./tests/test_data/room_impulse_response_with_noise.csv')
    return pf.Signal(time, 48000)


@pytest.mark.parametrize("cshape", [(1,), (2,), (2, 3)])
def test_mixing_time_abel_returns_correct_shapes(cshape):
    """Return one mixing time and one echo density per channel."""
    n_channels = int(np.prod(cshape))
    data = np.stack([_noise_rir(seed=i) for i in range(n_channels)])
    rir = pf.Signal(data.reshape(*cshape, -1), 48000)

    mixing_time, echo_density, onset = mixing_time_abel(rir)

    assert isinstance(mixing_time, np.ndarray)
    assert mixing_time.shape == rir.cshape
    assert isinstance(echo_density, pf.TimeData)
    assert echo_density.cshape == rir.cshape
    assert isinstance(onset, float)


@pytest.mark.parametrize("cshape", [(1,), (2, 3)])
def test_perceptual_mixing_time_lindau_returns_correct_shape(cshape):
    """Return one perceptual mixing time per channel."""
    n_channels = int(np.prod(cshape))
    data = np.stack([_noise_rir(seed=i) for i in range(n_channels)])
    rir = pf.Signal(data.reshape(*cshape, -1), 48000)

    result = perceptual_mixing_time_lindau(rir)

    assert isinstance(result, np.ndarray)
    assert result.shape == rir.cshape


@pytest.mark.parametrize(
    "function", [mixing_time_abel, perceptual_mixing_time_lindau])
def test_mixing_time_rejects_non_signal_input(function):
    """TypeError is raised when input data is not a pyfar.Signal."""
    match = "Input data must be a pyfar.Signal."
    with pytest.raises(TypeError, match=match):
        function(_noise_rir())


def test_mixing_time_abel_rejects_silent_channel():
    """ValueError is raised when a channel carries no signal."""
    rir = pf.Signal(np.zeros(20000), 48000)
    match = ("Onset detection failed. At least one channel contained only "
             "zeros.")
    with pytest.raises(ValueError, match=match):
        mixing_time_abel(rir)


def test_mixing_time_abel_rejects_ir_shorter_than_window():
    """ValueError is raised when the IR is shorter than the analysis window."""
    rir = pf.Signal(_noise_rir(n_samples=512, onset=10), 48000)
    match = ("The impulse response is 512 samples long after onset detection, "
             "which is shorter than the analysis window length of 1024 "
             "samples.")
    with pytest.raises(ValueError, match=match):
        mixing_time_abel(rir)


@pytest.mark.parametrize("window_length", [0, -2, 1023, 1024.0])
def test_mixing_time_abel_rejects_invalid_window_length(window_length):
    """ValueError is raised when the window is not a positive even integer."""
    rir = pf.Signal(_noise_rir(), 48000)
    match = (f"window_length is {window_length} but must be a positive even "
             "integer.")
    with pytest.raises(ValueError, match=match):
        mixing_time_abel(rir, window_length=window_length)


@pytest.mark.parametrize("onset_threshold_db", [0, 3])
def test_mixing_time_abel_rejects_non_negative_onset_threshold(
        onset_threshold_db):
    """ValueError is raised when the threshold is not below the peak."""
    rir = pf.Signal(_noise_rir(), 48000)
    match = f"onset_threshold_db is {onset_threshold_db} but must be negative."
    with pytest.raises(ValueError, match=match):
        mixing_time_abel(rir, onset_threshold_db=onset_threshold_db)


@pytest.mark.parametrize("peak_secure_margin", [-1, 100.0])
def test_mixing_time_abel_rejects_invalid_peak_secure_margin(
        peak_secure_margin):
    """ValueError is raised when the margin is not a non-negative integer."""
    rir = pf.Signal(_noise_rir(), 48000)
    match = (f"peak_secure_margin is {peak_secure_margin} but must be a "
             "non-negative integer.")
    with pytest.raises(ValueError, match=match):
        mixing_time_abel(rir, peak_secure_margin=peak_secure_margin)


def test_perceptual_mixing_time_lindau_rejects_invalid_percentile():
    """ValueError is raised for percentiles without a regression."""
    rir = pf.Signal(_noise_rir(), 48000)
    match = "percentile is 90 but must be 50 or 95."
    with pytest.raises(ValueError, match=match):
        perceptual_mixing_time_lindau(rir, percentile=90)


def test_mixing_time_abel_returns_nan_without_diffuse_field():
    """NaN and a warning are returned when the echo density never exceeds
    unity.

    A Dirac in silence is maximally sparse: inside every window the sample
    standard deviation is set by the single non-zero sample, so the outlier
    fraction never reaches the Gaussian expectation.
    """
    rir = pf.signals.impulse(5000)
    match = "The echo density never exceeds unity in 1 channel"
    with pytest.warns(UserWarning, match=match):
        mixing_time, echo_density, _ = mixing_time_abel(rir)

    assert np.isnan(mixing_time[0])
    assert np.all(echo_density.time <= 1)


def test_mixing_time_matches_reference_values():
    """Transition time and tmp50 against the reference implementation.

    Cross-checked against AKcutIRmixingTime + AKmixingTimeAbel +
    AKdataBasedMixingTime (AKtools, TU Berlin) on the same impulse response.
    Both implementations agree, so the tolerance is tight.
    """
    rir = _measured_rir()

    mixing_time, _, onset = mixing_time_abel(rir)
    npt.assert_allclose(mixing_time, 72.8333333333e-3, atol=1e-9)
    npt.assert_allclose(onset, 1404 / 48000)
    npt.assert_allclose(
        perceptual_mixing_time_lindau(rir), 50.2666666667e-3, atol=1e-9)


@pytest.mark.parametrize(
    ("percentile", "slope", "intercept"),
    [(50, 0.8, -0.008), (95, 1.77, -0.038)])
def test_perceptual_mixing_time_lindau_maps_transition_time(
        percentile, slope, intercept):
    """t_mp50 and t_mp95 are affine maps of the transition time."""
    rir = _measured_rir()

    t_abel, _, _ = mixing_time_abel(rir)
    t_mp = perceptual_mixing_time_lindau(rir, percentile=percentile)

    assert np.all(slope * t_abel + intercept > 0)  # NaN branch not exercised
    npt.assert_allclose(t_mp, slope * t_abel + intercept, atol=1e-12)


@pytest.mark.parametrize(
    ("percentile", "sampling_rate"), [(50, 192000), (95, 48000)])
def test_perceptual_mixing_time_lindau_returns_nan_for_negative_prediction(
        percentile, sampling_rate):
    """A negative prediction returns NaN and a warning.

    The transition is found at the same sample regardless of the sampling
    rate, so a high rate drives the transition time below the range of the
    t_mp50 regression. The t_mp95 regression is negative below 21.5 ms,
    which the synthetic impulse response reaches at 48 kHz.
    """
    rir = pf.Signal(_noise_rir(onset=20, seed=3), sampling_rate)

    match = f"The predicted t_mp{percentile} is negative in 1 channel"
    with pytest.warns(UserWarning, match=match):
        result = perceptual_mixing_time_lindau(rir, percentile=percentile)

    assert np.isnan(result[0])


@pytest.mark.parametrize(
    ("onset_threshold_db", "expected_onset"), [(-30, 300), (-10, 500)])
def test_mixing_time_abel_onset_follows_threshold(
        onset_threshold_db, expected_onset):
    """A pre-echo 20 dB below the peak is the onset only for a threshold
    below -20 dB.
    """
    time = _noise_rir()
    time[300] = 1.0
    rir = pf.Signal(time, 48000)

    _, _, onset = mixing_time_abel(
        rir, onset_threshold_db=onset_threshold_db)

    npt.assert_allclose(onset, expected_onset / 48000)


@pytest.mark.parametrize("peak_secure_margin", [0, 50, 400])
def test_mixing_time_abel_margin_shifts_cut_but_not_mixing_time(
        peak_secure_margin):
    """The margin sets where the echo density starts, while the mixing time
    is still measured from the onset.
    """
    rir = _measured_rir()

    mixing_time, echo_density, onset = mixing_time_abel(
        rir, peak_secure_margin=peak_secure_margin)

    npt.assert_allclose(
        echo_density.times[0], onset - peak_secure_margin / 48000)
    npt.assert_allclose(mixing_time, 72.8333333333e-3, atol=1e-9)


def test_mixing_time_abel_cuts_all_channels_at_earliest_onset():
    """Multichannel input is cut at the earliest onset across channels.

    The channel holding the earliest onset keeps its single-channel reference
    value of 72.83 ms, while the later-onset channel is not aligned to its
    own direct sound. The delayed channel comes first so that the earliest
    onset is not simply the onset of channel zero.

    The 4.83 ms of the delayed channel is not a meaningful mixing time: the
    shared cut places its analysis window in the pre-onset noise floor of the
    measured impulse response, whose Gaussian statistics cross unity at once.
    It pins the cut behaviour, and matches AKtools on the same input.
    """
    time = _measured_rir().time[0]
    delayed = np.concatenate([np.zeros(1500), time[:-1500]])

    mixing_time, _, onset = mixing_time_abel(
        pf.Signal(np.stack([delayed, time]), 48000))

    npt.assert_allclose(onset, 1404 / 48000)
    npt.assert_allclose(
        mixing_time, [4.8333333333e-3, 72.8333333333e-3], atol=1e-9)


@pytest.mark.parametrize("onset", [0, 99, 100, 500])
def test_mixing_time_abel_is_independent_of_onset_position(onset):
    """The mixing time is measured from the onset wherever the onset lies.

    The same impulse response is preceded by a varying number of zeros. If
    fewer than ``peak_secure_margin`` samples precede the onset, the cut
    keeps all of them and subtracts only those from the transition time.
    This deliberately departs from AKtools, which subtracts the full margin
    even when it cut less, and returns 11.4583 ms for onsets below sample
    100 of this impulse response.
    """
    tail = _noise_rir(n_samples=19980, onset=0, seed=3)
    rir = pf.Signal(np.concatenate([np.zeros(onset), tail]), 48000)

    npt.assert_allclose(mixing_time_abel(rir)[0], 13.5416666667e-3, atol=1e-9)


def test_mixing_time_abel_echo_density_crosses_unity_at_mixing_time():
    """The echo density ends with the impulse response and first exceeds
    unity at the returned mixing time after the returned onset.

    The transition time counts the crossing sample 1-based, so it lies one
    sample after the 0-based sample of the crossing.
    """
    rir = _measured_rir()
    fs = rir.sampling_rate

    mixing_time, echo_density, onset = mixing_time_abel(rir)
    crossing = np.argmax(echo_density.time[0] > 1)

    npt.assert_allclose(echo_density.times[-1], rir.times[-1])
    npt.assert_allclose(
        echo_density.times[crossing] + 1 / fs, onset + mixing_time[0])


def test_mixing_time_abel_echo_density_of_gaussian_noise_is_unity():
    """The normalisation gives stationary Gaussian noise an echo density of
    one on average.
    """
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(48000)
    noise[0] = 10  # onset at the first sample

    _, echo_density, _ = mixing_time_abel(pf.Signal(noise, 48000))

    # exclude the growing and shrinking windows at both ends
    assert abs(np.mean(echo_density.time[0, 2048:-2048]) - 1) < 0.01


def test_mixing_time_abel_echo_density_matches_sliding_window_definition():
    """The echo density equals a direct sample-by-sample evaluation.

    The window around sample ``i`` spans ``x[max(0, i - N/2):
    min(L, i + N/2)]``, the count is divided by ``N`` at the edges too, and
    the standard deviation uses ``N - 1`` degrees of freedom.
    """
    window_length = 64
    half = window_length // 2
    rir = pf.Signal(_noise_rir(n_samples=3000, onset=10), 48000)

    _, echo_density, _ = mixing_time_abel(rir, window_length=window_length)

    x = rir.time[0]
    expected = np.empty(x.size)
    for i in range(x.size):
        window = x[max(0, i - half):min(x.size, i + half)]
        n_out = np.count_nonzero(np.abs(window) > np.std(window, ddof=1))
        expected[i] = n_out / window_length / erfc(1 / np.sqrt(2))

    npt.assert_allclose(echo_density.time[0], expected, atol=1e-12)


@pytest.mark.parametrize(
    "function", [mixing_time_abel, perceptual_mixing_time_lindau])
def test_mixing_time_plot(function):
    """The plot marks the mixing time of a measured impulse response."""
    function(_measured_rir(), plot=True)
    plt.close('all')


@pytest.mark.parametrize(
    "function", [mixing_time_abel, perceptual_mixing_time_lindau])
def test_mixing_time_plot_without_mixing_time(function):
    """The plot runs when no mixing time is found."""
    with pytest.warns(UserWarning, match="never exceeds unity"):
        function(pf.signals.impulse(5000), plot=True)
    plt.close('all')


@pytest.mark.parametrize(
    "function", [mixing_time_abel, perceptual_mixing_time_lindau])
def test_mixing_time_plot_accepts_single_channel_2d(function):
    """A single channel with a 2D cshape can be plotted."""
    rir = pf.Signal(_measured_rir().time[np.newaxis], 48000)
    function(rir, plot=True)
    plt.close('all')


@pytest.mark.parametrize(
    "function", [mixing_time_abel, perceptual_mixing_time_lindau])
def test_mixing_time_plot_rejects_multichannel_input(function):
    """ValueError is raised when plotting more than one channel.

    The channels are impulses that would warn if computed, so the test also
    pins that the check runs before the computation.
    """
    rir = pf.Signal(np.zeros((2, 5000)), 48000)
    rir.time[:, 0] = 1
    match = "The plot can only be done for single channel input."
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=match):
            function(rir, plot=True)
