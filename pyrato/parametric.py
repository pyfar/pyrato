"""Module for room acoustics related functions.

Parametric room acoustics calculations using simple geometric considerations
such as Sabine's theory of sound in rooms.
"""
import numpy as np
from typing import Literal, Union, List
import pyfar as pf
from scipy.integrate import cumulative_trapezoid


def energy_decay_curve(
        times : np.ndarray,
        reverberation_time : float | np.ndarray,
        energy : float | np.ndarray = 1,
    ) -> pf.TimeData:
    r"""Calculate the energy decay curve for the reverberation time and energy.

    The energy decay curve is calculated as

    .. math::
        E(t) = E_0 e^{-\frac{6 \ln(10)}{T_{60}} t}

    where :math:`E_0` is the initial energy, :math:`T_{60}` the reverberation
    time, and :math:`t` the time [#]_.

    Parameters
    ----------
    times : numpy.ndarray[float]
        The times at which the energy decay curve is evaluated.
    reverberation_time : float | numpy.ndarray[float]
        The reverberation time in seconds. If an array is passed, an energy
        decay curve is calculated for each reverberation time.
    energy : float | numpy.ndarray[float], optional
        The initial energy of the sound field, by default 1. If
        `reverberation_time` is an array, the shape of `energy` is required
        to match the shape or be broadcastable to the shape of
        `reverberation_time`.

    Returns
    -------
    energy_decay_curve : pyfar.TimeData
        The energy decay curve with a ``cshape`` equal to the shape of
        the passed ``reverberation_time``.

    Example
    -------
    Calculate and plot an energy decay curve with a reverberation time of
    2 seconds.

    .. plot::

        >>> import numpy as np
        >>> import pyrato
        >>> import pyfar as pf
        >>>
        >>> times = np.linspace(0, 3, 50)
        >>> T_60 = [2, 1]
        >>> edc = pyrato.parametric.energy_decay_curve(times, T_60)
        >>> pf.plot.time(edc, log_prefix=10, dB=True)


    References
    ----------

    .. [#] H. Kuttruff, Room acoustics, 4th Ed. Taylor & Francis, 2009.

    """
    reverberation_time = np.asarray(reverberation_time)
    energy = np.asarray(energy)
    times = np.asarray(times)

    if np.any(reverberation_time <= 0):
        raise ValueError("Reverberation time must be greater than zero.")

    if np.any(energy < 0):
        raise ValueError("Energy must be greater than or equal to zero.")

    if reverberation_time.shape != energy.shape:
        try:
            energy = np.broadcast_to(energy, reverberation_time.shape)
        except ValueError as error:
            raise ValueError(
                "Reverberation time and energy must be broadcastable to the "
                "same shape.",
            ) from error

    matching_shape = reverberation_time.shape
    reverberation_time = reverberation_time.flatten()
    energy = energy.flatten()

    reverberation_time = np.atleast_2d(reverberation_time)
    energy = np.atleast_2d(energy)

    damping_term = (3*np.log(10) / reverberation_time).T
    edc = energy.T * np.exp(-2*damping_term*times)

    return pf.TimeData(np.reshape(edc, (*matching_shape, times.size)), times)


def critical_distance(
                     volume,
                     reverberation_time):
    r"""Calculate the critical distance of a room with
    given volume and reverberation time.
    Assumes the source directivity is 1 (omnidirectional source).
    See [#kra]_.

    .. math::
        d_c = 0.057 \sqrt{\frac{V}{T_{60}}}

    Parameters
    ----------
    volume : double
        Volume of the room in cubic meters.
    reverberation_time : double
        Reverberation time of the room in seconds.

    Returns
    -------
    critical_dist : double
        The resulting critical distance in meters.

    References
    ----------
    .. [#kra] H. Kuttruff, Room acoustics, 4th Ed. Taylor & Francis, 2009.

    """
    if reverberation_time <= 0:
        raise ValueError("Reverberation time must be greater than zero.")
    if volume <= 0:
        raise ValueError("Volume must be greater than zero.")
    critical_dist = 0.057 * np.sqrt(volume / reverberation_time)
    return critical_dist


def mean_free_path(
        volume,
        surface_area):
    """Calculate the mean free path. Source https://ccrma.stanford.edu/~jos/smith-nam/Mean_Free_Path.html.

    Parameters
    ----------
    volume : double
        Room volume
    surface_area : double
        Total surface area

    Returns
    -------
    mean free path : double
        The calculated mean free path
    """

    if volume < 0:
        raise ValueError(f"Volume ({volume}) is smaller than 0.")
    if surface_area < 0:
        raise ValueError(f"Surface area ({surface_area}) is smaller than 0.")

    return 4 * volume / surface_area


def reverberation_time_eyring(
        volume: float,
        surface_area: float,
        mean_absorption: Union[float, np.ndarray],
        speed_of_sound: float = 343.4,
    ) -> np.ndarray:
    r"""
    Calculate the reverberation time in rooms as defined by Carl Eyring.

    The reverberation time is calculated according to Ref. [#]_ as

    .. math::
        T_{60} = -\frac{24 \cdot \ln(10)}{c}
        \cdot \frac{V}{S \ln(1 - \tilde{\alpha})}

    where :math:`V` is the room volume, :math:`S` is the total surface area
    of the room, :math:`\tilde{\alpha}` is the average absorption coefficient
    of the room surfaces, and :math:`c` is the speed of sound.

    Parameters
    ----------
    volume : float
        Room volume in :math:`\mathrm{m}^3`
    surface_area : float
        Total surface area of the room in :math:`\mathrm{m}^2`
    mean_absorption : float, numpy.ndarray
        Average absorption coefficient of room surfaces between 0 and 1. If
        an array is passed, the reverberation time is calculated for each value
        in the array.
    speed_of_sound : float
        Speed of sound in m/s. Default is 343.4 m/s, which corresponds to the
        speed of sound in air at 20 °C.

    Returns
    -------
    numpy.ndarray
        Reverberation time in seconds. The shape matches the shape of the input
        variable `mean_absorption`.

    Examples
    --------
    >>> from pyrato.parametric import reverberation_time_eyring
    >>> import numpy as np
    >>> volume = 64
    >>> surface_area = 96
    >>> mean_absorption = [0.1, 0.3, 0.4]
    >>> reverb_time = reverberation_time_eyring(
    ...     volume, surface_area, mean_absorption)
    >>> np.round(reverb_time, 2)
    ... array([1.02, 0.3 , 0.21])

    References
    ----------
    .. [#] Eyring, C.F., 1930. Reverberation time in "dead" rooms. The Journal
           of the Acoustical Society of America, 1(2A_Supplement), pp.168-168.

    """
    if speed_of_sound <= 0:
        raise ValueError("Speed of sound should be larger than 0")
    if volume <= 0:
        raise ValueError("Volume should be larger than 0")
    if surface_area <= 0:
        raise ValueError("Surface area should be larger than 0")

    mean_absorption = np.asarray(mean_absorption)
    if np.any(mean_absorption < 0) or np.any(mean_absorption > 1):
        raise ValueError("mean_absorption should be between 0 and 1")

    factor = 24 * np.log(10) / speed_of_sound

    with np.errstate(divide='ignore'):
        reverberation_time = -factor * (
            volume/(surface_area * np.log(1 - mean_absorption)))

    reverberation_time = np.where(
        np.isclose(mean_absorption, 0, atol=1e-10, rtol=1e-10),
        np.inf,
        reverberation_time)

    return reverberation_time


def reverberation_time_sabine(
        volume: float,
        surface_area: float,
        mean_absorption: Union[float, np.ndarray],
        speed_of_sound: float = 343.4,
    ) -> np.ndarray:
    r"""
    Calculate the reverberation time in rooms as defined by Wallace Sabine.

    The reverberation time is calculated according to Ref. [#]_ as

    .. math::
        T_{60} = \frac{24 \cdot \ln(10)}{c}
        \cdot \frac{V}{S\tilde{\alpha}}

    where :math:`V` is the room volume, :math:`S` is the total surface area
    of the room, :math:`\tilde{\alpha}` is the average absorption
    coefficient of the room surfaces, and :math:`c` is the speed of sound.

    Parameters
    ----------
    surface_area : float
        Total surface area of the room in :math:`\mathrm{m}^2`.
    mean_absorption : float, numpy.ndarray
        Average absorption coefficient of room surfaces between 0 and 1. If
        an array is passed, the reverberation time is calculated for each value
        in the array.
    volume : float
        Room volume in :math:`\mathrm{m}^3`.
    speed_of_sound : float
        Speed of sound in m/s. Default is 343.4 m/s, which corresponds to the
        speed of sound in air at 20 °C.

    Returns
    -------
    numpy.ndarray
        Reverberation time in seconds.

    References
    ----------
    .. [#] H. Kuttruff, Room acoustics, 4th Ed. Taylor & Francis, 2009.

    """

    if speed_of_sound <= 0:
        raise ValueError("Speed of sound should be larger than 0")
    if volume <= 0:
        raise ValueError("Volume should be larger than 0")
    if surface_area <= 0:
        raise ValueError("Surface area should be larger than 0")

    mean_absorption = np.asarray(mean_absorption)
    if np.any(mean_absorption < 0) or np.any(mean_absorption > 1):
        raise ValueError("mean_absorption should be between 0 and 1")

    factor = 24 * np.log(10) / speed_of_sound

    with np.errstate(divide='ignore'):
        reverberation_time = factor * volume / (surface_area * mean_absorption)

    return reverberation_time


def average_reflection_density(
        volume: float,
        times: np.ndarray | List,
        speed_of_sound: float | None = None,
    ) -> pf.TimeData:
    r"""Calculate the time dependent average reflection density in a room.

    The reflection density is calculated as the following ratio [#]_

    .. math::
        \frac{d N(t)}{dt} = \frac{4 \pi c^3 t^2}{V},

    where :math:`V` is the room volume in :math:`m^3`, :math:`c` is the
    speed of sound in the room, and :math:`t` is the time vector in seconds.

    Parameters
    ----------
    volume : float
        Volume of the room :math:`V` in :math:`m^3`.
    times : numpy.ndarray, list
        Time vector in seconds.
    speed_of_sound : float, None, optional
        Speed of sound in the room. By default, the
        :py:attr:`~pyfar.constants.reference_speed_of_sound` is used.

    Returns
    -------
    reflection_density : pyfar.TimeData
        The reflection density in :math:`1/s` as a function of time.

    Examples
    --------
    Calculate the reflection density for a room with a volume of 100 m³.

    .. plot::

        >>> import pyrato
        >>> import numpy as np
        >>> import pyfar as pf
        ...
        >>> n_samples = 2**10
        >>> sampling_rate = 16e3
        >>> times = np.arange(n_samples)/sampling_rate
        >>> density = pyrato.parametric.average_reflection_density(
        ...     volume=100, times=times)
        ...
        >>> plt.figure(figsize=(8, 4))
        >>> ax = pf.plot.time(density)
        >>> ax.set_yscale("log")
        >>> ax.set_ylabel("Reflection density in 1/s")

    References
    ----------
    .. [#] H. Kuttruff, Room acoustics, 7th Ed. Taylor & Francis, 2024.

    """

    times = np.asarray(times)

    if speed_of_sound is None:
        speed_of_sound = pf.constants.reference_speed_of_sound
    if speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be positive.")

    if np.any(times < 0):
        raise ValueError("'times' must be positive.")

    if volume <= 0:
        raise ValueError("'volume' must be positive.")

    density = 4 * np.pi * speed_of_sound**3 * times**2 / volume
    return pf.TimeData(density, times)


def average_number_of_reflections(
        volume: float,
        times: np.ndarray | List,
        speed_of_sound: float | None = None,
    ) -> pf.TimeData:
    r"""Calculate the time dependent average number of reflections in a room.

    The average number of reflections is calculated as the following ratio [#]_

    .. math::
        N(t) = \frac{4 \pi c^3 t^3}{3 V},

    where :math:`V` is the room volume in :math:`m^3`, :math:`c` is the
    speed of sound in the room, and :math:`t` is the time vector in seconds.

    Parameters
    ----------
    volume : float
        Volume of the room :math:`V` in :math:`m^3`.
    times : numpy.ndarray, list
        Time vector in seconds.
    speed_of_sound : float, None, optional
        Speed of sound in the room. By default, the
        :py:attr:`~pyfar.constants.reference_speed_of_sound` is used.

    Returns
    -------
    pyfar.TimeData
        The average number of reflections as a function of time.

    Examples
    --------
    Calculate the time dependent average number of reflections in a room
    with a volume of 100 :math:`m^3`.

    .. plot::

        >>> from pyrato.parametric import average_number_of_reflections
        >>> import numpy as np
        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        ...
        >>> n_samples = 2**10
        >>> sampling_rate = 16e3
        >>> times = np.arange(n_samples)/sampling_rate
        >>> number_of_reflections = average_number_of_reflections(
        ...     volume=100, times=times)
        ...
        >>> plt.figure(figsize=(8, 4))
        >>> ax = pf.plot.time(number_of_reflections)
        >>> ax.set_yscale("log")
        >>> ax.set_ylabel("Average number of reflections")

    References
    ----------
    .. [#] H. Kuttruff, Room acoustics, 7th Ed. Taylor & Francis, 2024.

    """

    density = average_reflection_density(volume, times, speed_of_sound)
    number_of_reflections = density.time * times / 3
    return pf.TimeData(number_of_reflections, times)


def _start_time_of_arrival_poisson_process(
        volume : float,
        speed_of_sound: float | None = None,
    ) -> float:
    """
    The earliest time of arrival approximated as a Poisson process.

    Calculated according to [#]_.

    Parameters
    ----------
    volume : float
        Volume of the room in m³.
    speed_of_sound : float, None, optional
        Speed of sound in the room. By default,
        the :py:attr:`~pyfar.constants.reference_speed_of_sound` is used
        which corresponds to the speed of sound in air at 20 °C.

    Returns
    -------
    float
        Earliest expected time of arrival in seconds.

    References
    ----------
    .. [#] D. Schröder, “Physically based real-time auralization of
           interactive virtual environments,” PhD Thesis, Logos-Verlag,
           Berlin, 2011. [Online].
           Available: https://publications.rwth-aachen.de/record/50580

    """

    if speed_of_sound is None:
        speed_of_sound = pf.constants.reference_speed_of_sound
    if speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be positive.")

    if volume <= 0:
        raise ValueError("'volume' must be positive.")

    return (2*volume*np.log(2)/ (4*np.pi*speed_of_sound**3))**(1/3)


def time_of_arrival_poisson_process(
        volume: float,
        times: np.ndarray,
        speed_of_sound: float | None = None,
        reflection_rate_limit: float = np.inf,
        seed: int | None = None,
    ) -> np.ndarray:
    """Generate a time of arrival sequence based on a Poisson process.

    The reflection rate is calculated using the average reflection density
    in a diffuse sound field.
    Note that the reflection rate increases with time, yielding a
    non-homogeneous Poisson process. Optionally, the reflection rate can
    be limited using the ``reflection_rate_limit`` parameter.
    In [#]_, a maximum of 10000 reflections per second is suggested.

    The implementation of the non-homogeneous Poisson process is based on
    the transform method described in chap 5 of [#]_.

    Parameters
    ----------
    volume : float
        Volume of the room in m³.
    times : numpy.ndarray
        Time vector in seconds.
    speed_of_sound : float, None, optional
        Speed of sound in the room. By default, the
        :py:attr:`~pyfar.constants.reference_speed_of_sound` is used.
    reflection_rate_limit : float, optional
        Maximum reflection rate in 1/s. If ``np.inf``, no limit is applied.
        Default is ``np.inf``.
    seed : int, None, optional
        Seed for the random number generator. If None, a random seed is used.
        Default is None.

    Returns
    -------
    numpy.ndarray
        Array of arrival times in seconds.

    Examples
    --------
    Simulate the time of arrival of reflections in a room with a volume
    of 100 m³ and compare the cumulative histogram to the model prediction.

    .. plot::

        >>> import pyrato
        >>> import numpy as np
        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        ...
        >>> volume = 100
        >>> times = np.linspace(0, .5, 200)
        >>> toa = pyrato.parametric.time_of_arrival_poisson_process(
        ...     volume, times)
        ...
        >>> plt.figure(figsize=(8, 4))
        >>> plt.hist(
        ...     toa, density=False, bins=50, cumulative=True, histtype='step',
        ...     linewidth=1.5, color='C0', label='Simulation')
        >>> ax = pf.plot.time(
        ...     pyrato.parametric.average_number_of_reflections(
        ...         volume, times),
        ...     label='Model', linestyle='--',
        ...     color='grey', linewidth=1.5)
        >>> ax.set_ylabel('Number of reflections')
        >>> ax.set_yscale('log')
        >>> ax.legend(loc='lower right')
        >>> ax.grid()

    References
    ----------
    .. [#] D. Schröder, “Physically based real-time auralization of
           interactive virtual environments,” PhD Thesis, Logos-Verlag,
           Berlin, 2011. [Online].
           Available: https://publications.rwth-aachen.de/record/50580
    .. [#] S. M. Ross, Simulation, Sixth edition. London, United Kingdom:
           Academic Press, 2023.

    """

    if speed_of_sound is None:
        speed_of_sound = pf.constants.reference_speed_of_sound

    if speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be positive.")

    if volume <= 0:
        raise ValueError("'volume' must be positive.")

    if (
        not np.isinf(reflection_rate_limit) and np.isnan(reflection_rate_limit)
    ) or reflection_rate_limit < 0:
        raise ValueError(
            "'reflection_rate_limit' must be non-negative and not NaN.",
        )
    rng = np.random.default_rng(seed=seed)

    reflection_density = average_reflection_density(
        volume, times, speed_of_sound,
    )

    mu_values = np.minimum(
        np.squeeze(reflection_density.time),
        reflection_rate_limit,
    )

    mu_times = reflection_density.times
    if np.any(np.diff(mu_times) <= 0):
        raise ValueError("'times' must be strictly increasing.")
    t_start = _start_time_of_arrival_poisson_process(volume, speed_of_sound)

    # Cumulative intensity F(t) via numerical integration
    cumulative_intensity = cumulative_trapezoid(mu_values, mu_times, initial=0)

    # Interpolate the cumulative intensity to find the warped time values
    F_start = np.interp(t_start, mu_times, cumulative_intensity)
    F_end = cumulative_intensity[-1]

    # expected number of arrivals
    total_events = F_end - F_start

    # Draw the total count, then place events uniformly in warped time
    n_events = rng.poisson(total_events)
    warped = rng.uniform(F_start, F_end, size=n_events)
    warped.sort()

    # Invert the warped time to get the arrival times by interpolation
    arrivals = np.interp(warped, cumulative_intensity, mu_times)

    return arrivals[arrivals >= t_start]


def random_reflection_sequence(
        arrivals : np.ndarray,
        n_samples : int,
        sampling_rate : float,
        distribution : Literal['uniform', 'binary', 'normal'] = "normal",
        seed : int | None = None,
    ) -> pf.Signal:
    r"""Generate reflection sequence from arrival times with random amplitudes.

    The amplitude is randomly sampled according to the chosen distribution
    function. `'normal'` and `'uniform'` yield continuous amplitude values and
    are suitable to encode random amplitude and phase of reflections.
    In contrast, `'binary'` yields only -1 and 1, and hence is only suitable
    to encode random phase.

    The final reflection sequence is generated by mapping the arrival times to
    uniform time samples. Duplicate time samples are removed, which results in
    a maximum of one reflection per time sample.

    Parameters
    ----------
    arrivals : numpy.ndarray
        Array of arrival times in seconds.
    n_samples : int
        Number of samples in the output sequence.
    sampling_rate : float
        Sampling rate in Hz.
    distribution : Literal['uniform', 'binary', 'normal'], optional
        Distribution of the reflection amplitudes. Default is 'normal'.
    seed : int, None, optional
        Seed for the random number generator. If None, a random seed is used.

    Returns
    -------
    pyfar.Signal
        Reflection sequence with a maximum of one reflection per time sample.
        The sequence has a length of ``n_samples`` and a sampling rate of
        ``sampling_rate``.

    Examples
    --------
    Create a reflection sequence from a set of arrival times
    and plot the result.

    .. plot::

        >>> import pyrato
        >>> import numpy as np
        >>> import pyfar as pf
        ...
        >>> times_of_arrival = np.asarray([.1, .3, .35, .41])
        >>> sequence = pyrato.parametric.random_reflection_sequence(
        ...     times_of_arrival, n_samples=50, sampling_rate=100,
        ...     distribution='normal', seed=10)
        >>> pf.plot.time(sequence, marker='o', linewidth=0.5)

    """

    rng = np.random.default_rng(seed=seed)

    sample_indices = np.round(arrivals * sampling_rate).astype(int)
    sample_indices = sample_indices[
        (sample_indices >= 0) & (sample_indices < n_samples)]

    if distribution == "normal":
        amplitude = rng.normal(
            loc=0,
            scale=1,
            size=len(sample_indices),
        )
    elif distribution == 'uniform':
        # min and max values are chosen to yield unit variance
        amplitude = rng.uniform(
            low=-np.sqrt(3),
            high=np.sqrt(3),
            size=len(sample_indices),
        )
    elif distribution == 'binary':
        amplitude = rng.choice(
            [-1, 1],
            p=[0.5, 0.5],
            size=len(sample_indices),
        )
    else:
        raise ValueError(
            "Unknown distribution type. "
            "Choose from 'uniform', 'binary', or 'normal'.")

    sequence = np.zeros(n_samples)
    unique_samples, unique_idx = np.unique(sample_indices, return_index=True)
    sequence[unique_samples] = amplitude[unique_idx]

    return pf.Signal(sequence, sampling_rate)
