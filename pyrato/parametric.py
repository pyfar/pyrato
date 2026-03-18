"""Module for room acoustics related functions.

Parametric room acoustics calculations using simple geometric considerations
such as Sabine's theory of sound in rooms.
"""
import numpy as np
from typing import Union
import pyfar as pf


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


def air_attenuation_coefficient(
        frequency,
        temperature=20,
        humidity=50,
        atmospheric_pressure=101325):
    """Calculate the attenuation coefficient m for the absorption caused
     by friction with the surrounding air.

    Parameters
    ----------
    frequency : double
        The frequency for which the attenuation coefficient is calculated.
        When processing in fractional octave bands use the center frequency.
    temperature : double
        Temperature in degrees Celsius.
    humidity : double
        Humidity in percent.
    atmospheric_pressure : double
        Atmospheric pressure.

    Returns
    -------
    attenuation_coefficient : double
        The resulting attenuation coefficient.

    """
    from pyfar.classes.warnings import PyfarDeprecationWarning
    import warnings

    warnings.warn(
        'Will be replaced by respective function in pyfar before v1.0.0',
        PyfarDeprecationWarning, stacklevel=2)

    # room temperature in Kelvin
    t_K = temperature + 273.16
    p_ref_kPa = 101.325
    p_kPa = atmospheric_pressure/1000.0

    # determine molar concentration of water vapor
    tmp = (
        (10.79586 * (1.0 - (273.16/t_K))) -
        (5.02808 * np.log10((t_K/273.16))) +
        (1.50474 * 0.0001 * (1.0 - 10.0 ** (-8.29692*((t_K/273.16) - 1.0)))) +
        (0.42873 * 0.001 * (-1.0 + 10.0 ** (-4.76955*(1.0 - (273.16/t_K))))) -
        2.2195983)

    # molar concentration water vapor in percent
    molar_water_vapor = (humidity * 10.0 ** tmp) / (p_kPa/p_ref_kPa)

    # determine relaxation frequencies of oxygen and nitrogen
    relax_oxygen = ((p_kPa/p_ref_kPa) * (24.0 + (
            4.04 * 10000.0 * molar_water_vapor * (
                (0.02 + molar_water_vapor) / (0.391 + molar_water_vapor)))))

    relax_nitrogen = ((p_kPa/p_ref_kPa) * (
        (t_K / 293.16) ** (-0.5)) *
        (9.0 + 280.0 * molar_water_vapor * np.exp(
            -4.17 * (((t_K / 293.16) ** (-0.3333333)) - 1.0))))

    # Neper/m -> dB/m
    air_abs_coeff = ((frequency**2 * (
        (1.84 * 10.0**(-11.0) * (p_ref_kPa / p_kPa) * (t_K/293.16)**0.5) +
        ((t_K/293.16)**(-2.5) * (
            (1.278 * 0.01 * np.exp(-2239.1/t_K) / (
                relax_oxygen + (frequency**2)/relax_oxygen)) +
            (1.068 * 0.1 * np.exp((-3352.0/t_K) / (
                relax_nitrogen + (frequency**2)/relax_nitrogen))))))
            ) * 20.0 / np.log(10.0) / (np.log10(np.exp(1.0)) * 10.0))

    return air_abs_coeff


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
