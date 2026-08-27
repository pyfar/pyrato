"""Tests for parametric related things."""

import numpy as np
import numpy.testing as npt
import pytest
import pyrato.parametric as parametric
from pyrato.parametric import mean_free_path
import pyrato as ra
import pyrato
import pyfar as pf
from scipy import stats


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


# ======================================================================
# _start_time_of_arrival_poisson_process
# ======================================================================

def test_start_time_of_arrival_correct_value():
    """Return value matching the closed-form expression."""
    volume = 100
    speed_of_sound = 343
    result = parametric._start_time_of_arrival_poisson_process(
        volume, speed_of_sound)
    expected = (
        2 * volume * np.log(2) / (4 * np.pi * speed_of_sound**3)
    ) ** (1 / 3)
    npt.assert_allclose(result, expected)


def test_start_time_of_arrival_default_speed_of_sound():
    """Omitting speed_of_sound uses the pyfar reference value."""
    volume = 100
    result_default = parametric._start_time_of_arrival_poisson_process(
        volume)
    result_explicit = parametric._start_time_of_arrival_poisson_process(
        volume, pf.constants.reference_speed_of_sound)
    npt.assert_allclose(result_default, result_explicit)


@pytest.mark.parametrize('volume', [0, -1])
def test_start_time_of_arrival_invalid_volume(volume):
    """Non-positive volume must raise a ValueError."""
    with pytest.raises(ValueError, match="'volume' must be positive"):
        parametric._start_time_of_arrival_poisson_process(volume, 343)


@pytest.mark.parametrize('speed_of_sound', [0, -343])
def test_start_time_of_arrival_invalid_speed_of_sound(speed_of_sound):
    """Non-positive speed of sound must raise a ValueError."""
    with pytest.raises(ValueError, match="speed_of_sound must be positive"):
        parametric._start_time_of_arrival_poisson_process(100, speed_of_sound)


# ======================================================================
# time_of_arrival_poisson_process
# ======================================================================

def test_poisson_process_toa_kolmogorov_smirnov_statistic():
    """
    Test if the time of arrival intervals are drawn according to the
    expected distribution of reflections in a room with a given volume and
    speed of sound.

    The test uses the Kolmogorov-Smirnov test to compare the empirical
    distribution of the time of arrival intervals with the expected cumulative
    distribution from room acoustics theory.
    """
    volume = 100
    speed_of_sound = 343

    times = np.linspace(0, 1, 100)
    toa = pyrato.parametric.time_of_arrival_poisson_process(
        volume,
        times,
        speed_of_sound,
        seed=42,
    )

    def cumulative_reflections_callable(x):
        """Normalized to fall in the range [0, 1].

        All constant parameters are not relevant after normalization,
        only the time dependency remains, which is cubic in time.
        """
        return (x / times[-1])**3

    ks_test = stats.kstest(
        toa,
        cumulative_reflections_callable,
        alternative='two-sided',
    )

    # p-value < 0.01 reject null hypothesis that the samples are drawn from
    # the expected distribution.
    # p-value > 0.01 fail to reject the null hypothesis that the data are not
    # drawn from the expected distribution.
    assert ks_test.pvalue > 0.01

def test_toa_poisson_seed_reproducibility():
    """Same seed must yield identical arrival arrays."""
    volume = 100
    times = np.linspace(0, 0.1, 100)
    toa1 = parametric.time_of_arrival_poisson_process(
        volume, times, seed=42)
    toa2 = parametric.time_of_arrival_poisson_process(
        volume, times, seed=42)
    npt.assert_array_equal(toa1, toa2)


def test_toa_poisson_arrivals_ge_t_start():
    """All returned arrivals must be >= the Poisson-process start time."""
    volume = 100
    speed_of_sound = 343
    times = np.linspace(0, 0.1, 100)
    t_start = parametric._start_time_of_arrival_poisson_process(
        volume, speed_of_sound)
    toa = parametric.time_of_arrival_poisson_process(
        volume, times, speed_of_sound, seed=0)
    assert len(toa) > 0
    assert np.all(toa >= t_start)


def test_toa_poisson_arrivals_within_time_range():
    """All returned arrivals must lie within the supplied time vector."""
    volume = 100
    times = np.linspace(0, 0.1, 100)
    toa = parametric.time_of_arrival_poisson_process(volume, times, seed=0)
    assert len(toa) > 0
    assert np.all(toa <= times[-1])


def test_toa_poisson_reflection_rate_limit_reduces_events():
    """A low reflection rate limit must produce fewer events than no limit."""
    volume = 100
    speed_of_sound = 343
    times = np.linspace(0, 0.1, 100)
    seed = 0
    toa_unlimited = parametric.time_of_arrival_poisson_process(
        volume, times, speed_of_sound, seed=seed)
    toa_limited = parametric.time_of_arrival_poisson_process(
        volume, times, speed_of_sound, reflection_rate_limit=1, seed=seed)
    assert len(toa_limited) < len(toa_unlimited)


@pytest.mark.parametrize('volume', [0, -1])
def test_toa_poisson_invalid_volume(volume):
    """Non-positive volume must raise a ValueError."""
    with pytest.raises(ValueError, match="'volume' must be positive"):
        parametric.time_of_arrival_poisson_process(
            volume, np.linspace(0, 1, 10))


@pytest.mark.parametrize('speed_of_sound', [0, -343])
def test_toa_poisson_invalid_speed_of_sound(speed_of_sound):
    """Non-positive speed of sound must raise a ValueError."""
    with pytest.raises(ValueError, match="speed_of_sound must be positive"):
        parametric.time_of_arrival_poisson_process(
            100, np.linspace(0, 1, 10), speed_of_sound=speed_of_sound)


# ======================================================================
# random_reflection_sequence
# ======================================================================

@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_returns_signal(distribution):
    """Return type must be a pyfar Signal."""
    arrivals = np.asarray([0.1, 0.3, 0.35])
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=50, sampling_rate=100, distribution=distribution)
    assert isinstance(seq, pf.Signal)


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_length(distribution):
    """Output signal must have exactly n_samples samples."""
    arrivals = np.asarray([0.1, 0.3, 0.35])
    n_samples = 50
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=n_samples, sampling_rate=100,
        distribution=distribution)
    assert seq.n_samples == n_samples


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_sampling_rate(distribution):
    """Output signal must carry the requested sampling rate."""
    arrivals = np.asarray([0.1, 0.3])
    sampling_rate = 44100
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=100, sampling_rate=sampling_rate,
        distribution=distribution)
    assert seq.sampling_rate == sampling_rate


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_seed_reproducibility(distribution):
    """Same seed must yield an identical output signal."""
    arrivals = np.asarray([0.1, 0.3, 0.35, 0.41])
    seq1 = parametric.random_reflection_sequence(
        arrivals, n_samples=50, sampling_rate=100, seed=42,
        distribution=distribution)
    seq2 = parametric.random_reflection_sequence(
        arrivals, n_samples=50, sampling_rate=100, seed=42,
        distribution=distribution)
    npt.assert_array_equal(seq1.time, seq2.time)


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_arrivals_out_of_range(distribution):
    """Arrivals whose sample index >= n_samples must be ignored."""
    # 1.0 * 100 = 100 == n_samples, so it must be excluded
    arrivals = np.asarray([0.1, 1.0])
    n_samples = 100
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=n_samples, sampling_rate=100, seed=0,
        distribution=distribution)
    signal = np.squeeze(seq.time)
    assert np.count_nonzero(signal) == 1
    assert signal[10] != 0


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_unique_samples(distribution):
    """Two arrivals mapping to the same sample yield exactly one non-zero."""
    # 0.1 and 0.1001 both round to sample index 10 at fs=100
    arrivals = np.asarray([0.1, 0.1001])
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=50, sampling_rate=100, seed=0,
        distribution=distribution)
    assert np.count_nonzero(np.squeeze(seq.time)) == 1


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_nonzero_positions(distribution):
    """Non-zero positions must equal the rounded arrival sample indices."""
    arrivals = np.asarray([0.1, 0.3, 0.35, 0.41])
    n_samples = 50
    sampling_rate = 100
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=n_samples, sampling_rate=sampling_rate, seed=0,
        distribution=distribution)
    expected_indices = np.round(arrivals * sampling_rate).astype(int)
    nonzero_indices = np.flatnonzero(np.squeeze(seq.time))
    npt.assert_array_equal(
        np.sort(nonzero_indices), np.sort(expected_indices))


def test_reflection_sequence_binary_values():
    """Binary distribution must produce values only in {-1, 0, 1}."""
    arrivals = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=100, sampling_rate=100,
        distribution='binary', seed=0, compensate_sparsity=False)
    assert np.all(np.isin(np.squeeze(seq.time), [-1, 0, 1]))


def test_reflection_sequence_normal_values():
    """Normal distribution must produce continuous (non-binary) amplitudes."""
    arrivals = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=100, sampling_rate=100,
        distribution='normal', seed=0)
    nonzero = np.squeeze(seq.time)[np.squeeze(seq.time) != 0]
    assert not np.all(np.abs(nonzero) == 1)


def test_reflection_sequence_uniform_values():
    """Uniform distribution must stay within [-sqrt(3), sqrt(3)]."""
    arrivals = np.linspace(0, 0.99, 50)
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=100, sampling_rate=100,
        distribution='uniform', seed=0, compensate_sparsity=False)
    nonzero = np.squeeze(seq.time)[np.squeeze(seq.time) != 0]
    assert np.all(np.abs(nonzero) <= np.sqrt(3))


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_compensate_sparsity_flag(distribution):
    """compensate_sparsity=False must disable the sqrt(delta_t*fs) weighting."""
    arrivals = np.asarray([0.1, 0.3, 0.35])
    sampling_rate = 100
    n_samples = 50

    seq_corrected = parametric.random_reflection_sequence(
        arrivals, n_samples=n_samples, sampling_rate=sampling_rate,
        distribution=distribution, seed=7, compensate_sparsity=True)
    seq_raw = parametric.random_reflection_sequence(
        arrivals, n_samples=n_samples, sampling_rate=sampling_rate,
        distribution=distribution, seed=7, compensate_sparsity=False)

    # Forward inter-arrival times for [0.1, 0.3, 0.35]: [0.2, 0.05, 0.05]
    weights = np.sqrt(np.array([0.2, 0.05, 0.05]) * sampling_rate)
    nonzero = np.squeeze(seq_raw.time) != 0
    npt.assert_allclose(
        np.squeeze(seq_corrected.time)[nonzero],
        np.squeeze(seq_raw.time)[nonzero] * weights,
    )


def test_reflection_sequence_invalid_distribution():
    """An unrecognised distribution name must raise a ValueError."""
    with pytest.raises(ValueError, match="Unknown distribution"):
        parametric.random_reflection_sequence(
            np.asarray([0.1]), n_samples=10, sampling_rate=100,
            distribution='invalid')


@pytest.mark.parametrize('distribution', ['normal', 'uniform', 'binary'])
def test_reflection_sequence_negative_arrivals_ignored(distribution):
    """Negative arrival times must not wrap-around array boundaries."""
    arrivals = np.asarray([-0.1, 0.1])
    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=50, sampling_rate=100, seed=0,
        distribution=distribution)
    signal = np.squeeze(seq.time)
    # Only the arrival at 0.1 s (sample 10) must be non-zero;
    # the negative arrival must be silently dropped, not written to sample -10.
    assert np.count_nonzero(signal) == 1
    assert signal[10] != 0
    assert signal[-10] == 0  # last-10th element must be untouched


def test_reflection_sequence_toa_weight():
    """Amplitudes are weighted by sqrt(delta_t * fs) from inter-arrival times."""
    arrivals = np.asarray([0.1, 0.3, 0.35])
    sampling_rate = 100
    n_samples = 50

    seq = parametric.random_reflection_sequence(
        arrivals, n_samples=n_samples, sampling_rate=sampling_rate, seed=7)

    # Replicate weight: forward inter-arrival times [0.2, 0.05, 0.05]
    # (last extrapolated from its predecessor)
    delta_t = np.array([0.2, 0.05, 0.05])
    weights = np.sqrt(delta_t * sampling_rate)

    # Replicate raw amplitude with same seed
    rng = np.random.default_rng(7)
    raw_amps = rng.normal(0, 1, size=len(arrivals))

    signal = np.squeeze(seq.time)
    npt.assert_allclose(signal[signal != 0], raw_amps * weights)
