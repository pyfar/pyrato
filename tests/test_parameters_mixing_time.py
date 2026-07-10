import numpy as np
import pytest
import pyfar as pf
import numpy.testing as npt

from pyrato.parameters import mixing_time


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


@pytest.mark.parametrize("cshape", [(1,), (2,), (2, 3)])
def test_mixing_time_returns_correct_shape(cshape):
    """Return one value per channel, matching the input cshape."""
    n_channels = int(np.prod(cshape))
    data = np.stack([_noise_rir(seed=i) for i in range(n_channels)])
    rir = pf.Signal(data.reshape(*cshape, -1), 48000)

    result = mixing_time(rir)

    assert isinstance(result, np.ndarray)
    assert result.shape == rir.cshape


def test_mixing_time_rejects_non_signal_input():
    """TypeError is raised when input data is not a pyfar.Signal."""
    match = "Input data must be a pyfar.Signal."
    with pytest.raises(TypeError, match=match):
        mixing_time(_noise_rir())


def test_mixing_time_rejects_silent_channel():
    """ValueError is raised when a channel carries no signal."""
    rir = pf.Signal(np.zeros(20000), 48000)
    match = "cannot detect an onset: a channel is silent"
    with pytest.raises(ValueError, match=match):
        mixing_time(rir)


def test_mixing_time_rejects_ir_shorter_than_window():
    """ValueError is raised when the IR is shorter than the analysis window."""
    rir = pf.Signal(_noise_rir(n_samples=512, onset=10), 48000)
    match = "IR shorter than analysis window length"
    with pytest.raises(ValueError, match=match):
        mixing_time(rir)


def test_mixing_time_rejects_ir_without_diffuse_field():
    """ValueError is raised when the echo density never exceeds unity.

    A Dirac in silence is maximally sparse: inside every window the sample
    standard deviation is set by the single non-zero sample, so the outlier
    fraction never reaches the Gaussian expectation.
    """
    rir = pf.signals.impulse(5000)
    match = "Mixing time not found within given temporal limits"
    with pytest.raises(ValueError, match=match):
        mixing_time(rir)


def test_mixing_time_matches_reference_values():
    """Transition time and tmp50 against the reference implementation.

    Cross-checked against AKcutIRmixingTime + AKmixingTimeAbel +
    AKdataBasedMixingTime (AKtools, TU Berlin) on the same impulse response.
    Both implementations agree, so the tolerance is tight.
    """
    time = np.loadtxt('./tests/test_data/room_impulse_response_with_noise.csv')
    rir = pf.Signal(time, 48000)

    npt.assert_allclose(mixing_time(rir), 72.8333333333e-3, atol=1e-9)
    npt.assert_allclose(
        mixing_time(rir, lindau_regression=True), 50.2666666667e-3, atol=1e-9)


def test_mixing_time_lindau_regression_maps_transition_time():
    """tmp50 is an affine map of the transition time: 0.8 * t - 8 ms."""
    rir = pf.Signal(_noise_rir(), 48000)

    t_abel = mixing_time(rir)
    t_mp50 = mixing_time(rir, lindau_regression=True)

    assert np.all(0.8 * t_abel - 0.008 > 0)  # clipping branch not exercised
    npt.assert_allclose(t_mp50, 0.8 * t_abel - 0.008, atol=1e-12)


def test_mixing_time_lindau_regression_clips_negative_to_one_ms():
    """A negative tmp50 is clipped to 1 ms."""
    # The transition is found at the same sample regardless of the sampling
    # rate, so a high rate drives the transition time low enough for the
    # regression to go negative.
    rir = pf.Signal(_noise_rir(onset=20, seed=3), 192000)

    assert 0.8 * mixing_time(rir)[0] - 0.008 < 0
    npt.assert_allclose(mixing_time(rir, lindau_regression=True), 0.001)


def test_mixing_time_rounds_window_down_to_even_samples():
    """An odd window length is rounded down to an even sample count.

    The analysis window spans ``2 * (N // 2)`` samples while the echo density
    normalises by ``N``, so an odd ``N`` would bias the density low.

    The measured impulse response is used rather than the synthetic one: the
    bias is small, and only a signal whose echo density crosses unity close to
    a sample boundary resolves 1024 from an unrounded 1025.
    """
    time = np.loadtxt('./tests/test_data/room_impulse_response_with_noise.csv')
    rir = pf.Signal(time, 48000)

    npt.assert_allclose(
        mixing_time(rir, window_length=1025 / 48000),
        mixing_time(rir, window_length=1024 / 48000))


def test_mixing_time_window_length_changes_result():
    """The window length is not silently ignored."""
    rir = pf.Signal(_noise_rir(), 48000)

    assert not np.isclose(
        mixing_time(rir, window_length=512 / 48000), mixing_time(rir))


def test_mixing_time_window_has_a_lower_bound_of_four_samples():
    """Window lengths below 4 samples are clipped to 4."""
    rir = pf.Signal(_noise_rir(), 48000)

    npt.assert_allclose(
        mixing_time(rir, window_length=1 / 48000),
        mixing_time(rir, window_length=4 / 48000))


def test_mixing_time_cuts_all_channels_at_earliest_onset():
    """Multichannel input is cut at the earliest onset across channels.

    The channel holding the earliest onset is unaffected by the presence of
    the other, while a later-onset channel is not aligned to its own direct
    sound. The delayed channel comes first so that the earliest onset is not
    simply the onset of channel zero.

    The 4.83 ms of the delayed channel is not a meaningful mixing time: the
    shared cut places its analysis window in the pre-onset noise floor of the
    measured impulse response, whose Gaussian statistics cross unity at once.
    It pins the cut behaviour, and matches AKtools on the same input.
    """
    time = np.loadtxt('./tests/test_data/room_impulse_response_with_noise.csv')
    delayed = np.concatenate([np.zeros(1500), time[:-1500]])

    both = mixing_time(pf.Signal(np.stack([delayed, time]), 48000))
    delayed_alone = mixing_time(pf.Signal(delayed, 48000))
    time_alone = mixing_time(pf.Signal(time, 48000))

    # the earliest onset belongs to the second channel, which is unaffected
    npt.assert_allclose(both[1], time_alone[0])
    assert not np.isclose(both[0], delayed_alone[0])

    npt.assert_allclose(both, [4.8333333333e-3, 72.8333333333e-3], atol=1e-9)


def test_mixing_time_keeps_full_margin_when_onset_precedes_it():
    """An onset inside the first ``peak_secure_margin`` samples drops the cut
    margin to zero, while the full margin is still subtracted from the
    transition time.

    This decoupling is inherited from the reference implementation, where the
    cut function zeroes only its local copy of the margin. Cross-checked
    against AKtools, which returns the same value.
    """
    rir = pf.Signal(_noise_rir(onset=20, seed=3), 48000)

    npt.assert_allclose(mixing_time(rir), 11.4583333333e-3, atol=1e-9)
