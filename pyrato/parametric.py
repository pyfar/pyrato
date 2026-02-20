"""Module for room acoustics related functions.

Parametric room acoustics calculations using simple geometric considerations
such as Sabine's theory of sound in rooms.
"""
import numpy as np
from typing import Union

def calculate_speed_of_sound(temperature):
    r"""Calculate the speed of sound in air depending on the temperature.

    Speed of sound is calculated as [#]_.

    .. math::
        c = c_0 \sqrt{\frac{T - T_0}{20 - T_0}}
    .. math::
        T_0=-273.15
    .. math::
        c_0 = 343.2

    Parameters
    ----------
    temperature : double
        Temperature in degrees Celsius.

    Returns
    -------
    speed_of_sound : double
        Speed of sound in m/s.

    References
    ----------
    .. [#] ISO 9613-1 (Formula A.5)
    """
    if temperature < -273.15:
        raise ValueError(
            "Temperature must be greater than absolute zero (-273.15 °C).")
    T_0 = -273.15
    c_0 = 343.2
    speed_of_sound = c_0 * np.sqrt((temperature - T_0)/(20 - T_0))
    return speed_of_sound


def schroeder_frequency(volume, reverberation_time):
    r"""
    Calculate the Schroeder cut-off frequency of a room.

    Calculation according to [#]_:

    .. math::

        f_s = 2000 \sqrt{\left(\frac{T}{V}\right)}

    Parameters
    ----------
    volume : float, np.ndarray
        room volume in m^3
    reverberation_time : float, np.ndarray
        reverberation time in s

    Returns
    -------
    schroeder_frequency : float, np.ndarray
        schroeder frequency in Hz

    Raises
    ------
    TypeError
        If inputs are not numeric or NumPy arrays.
    ValueError
        If inputs are non-positive or have incompatible shapes

    References
    ----------
    .. [#] H. Kuttruff, Room acoustics, 4th Ed. Taylor & Francis, 2009.

    Note
    ----
    this function still needs some tests ...

    """
    if volume is None or reverberation_time is None:
        raise TypeError("volume and reverberation_time cannot be None.")

    if isinstance(volume, str) or isinstance(reverberation_time, str):
        raise TypeError("volume and reverberation_time cannot be strings.")
    if isinstance(volume, np.ndarray) and volume.dtype.kind in {"U", "S", "O"}:
        raise TypeError("volume must contain only numeric values.")
    if (
        isinstance(reverberation_time, np.ndarray) and
        reverberation_time.dtype.kind in {"U", "S", "O"}
        ):
        raise TypeError("reverberation_time only numeric values.")

    volume = np.asarray(volume, dtype=float)
    reverberation_time = np.asarray(reverberation_time, dtype=float)
    if not np.issubdtype(volume.dtype, np.floating):
        raise TypeError("volume must be a float or a numeric array.")
    if not np.issubdtype(reverberation_time.dtype, np.floating):
        raise TypeError("reverberation_time must be float or numeric array.")
    if np.any(volume <= 0) | np.any(reverberation_time <= 0):
        raise ValueError("volume and reverberation_time " \
    "must be positiv")
    if volume.size != reverberation_time.size:
        raise ValueError("volume and reverberation_time must have" \
        " compatible shapes, either same shape or one is scalar.")
    schroeder_frequency = 2000*np.sqrt(reverberation_time / volume)

    return schroeder_frequency

def energy_decay_curve_analytic(
        surfaces, alphas, volume, times, source=None,
        receiver=None, method='eyring', c=343.4, frequency=None,
        air_absorption=True):
    """Calculate the energy decay curve analytically by using Eyring's or
    Sabine's equation.

    Calculation according to [#]_.

    Parameters
    ----------
    surfaces : ndarray, double
        Surface areas of all surfaces in the room
    alphas : ndarray, double
        Absorption coefficients corresponding to each surface
    volume : double
        Room volume
    times : ndarray, double
        Time vector for which the decay curve is calculated
    source : Coordinates
        Coordinate object with the source coordinates
    receiver : Coordinates
        Coordinate object with the receiver coordinates
    method : 'eyring', 'sabine'
        Use either Eyring's or Sabine's equation
    c : double
        Speed of sound
    frequency : double, optional
        Center frequency of the respective octave band. This is only used for
        the air absorption calculation.
    air_absorption : bool, optional
        If True, the air absorption is included in the calculation.
        Default is True.

    Returns
    -------
    energy_decay_curve : ndarray, double
        The energy decay curve

    References
    ----------
    .. [#] H. Kuttruff, Room acoustics, 4th Ed. Taylor & Francis, 2009.

    """

    alphas = np.asarray(alphas)
    surfaces = np.asarray(surfaces)
    surface_room = np.sum(surfaces)
    alpha_mean = np.sum(surfaces*alphas) / surface_room

    if air_absorption:
        m = air_attenuation_coefficient(frequency)
    else:
        m = 0

    if all([source, receiver]):
        dist_source_receiver = np.linalg.norm(
            source.cartesian - receiver.cartesian)
        delay_direct = dist_source_receiver / c
    else:
        delay_direct = 0

    if method == 'eyring':
        energy_decay_curve = np.exp(
            -c*(times - delay_direct) *
            ((-surface_room * np.log(1 - alpha_mean) / 4 / volume) + m))
    elif method == 'sabine':
        energy_decay_curve = np.exp(
            -c*(times - delay_direct) *
            ((surface_room * alpha_mean / 4 / volume) + m))
    else:
        raise ValueError("The method has to be either 'eyring' or 'sabine'.")

    return energy_decay_curve


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


def calculate_sabine_reverberation_time(surfaces, alphas, volume):
    """Calculate the reverberation time using Sabine's equation.

    Calculation according to [#]_.

    Parameters
    ----------
    surfaces : ndarray, double
        Surface areas of all surfaces in the room in square meters.
    alphas : ndarray, double
        Absorption coefficients corresponding to each surface
    volume : double
        Room volume in cubic meters

    The shape of `surfaces` and `alphas` must match.

    Returns
    -------
    reverberation_time_sabine :  double
        The value of calculated reverberation time in seconds

    References
    ----------
    .. [#] H. Kuttruff, Room acoustics, 4th Ed. Taylor & Francis, 2009.

    """
    surfaces = np.asarray(surfaces)
    alphas = np.asarray(alphas)

    if alphas.shape != surfaces.shape:
       raise ValueError("Size of alphas and surfaces " \
       "ndarray sizes must match.")

    if np.any(alphas) < 0 or np.any(alphas > 1):
       raise ValueError("Absorption coefficient values must "\
                        f"be in range [0, 1]. Got {alphas}.")
    if np.any(surfaces < 0):
       raise ValueError("Surface areas cannot "\
                        f"be negative. Got {surfaces}.")
    if volume < 0:
       raise ValueError(f"Volume cannot be negative. Got {volume}.")

    absorption_area = np.sum(surfaces*alphas)

    if absorption_area == 0:
       raise ZeroDivisionError("Absorption area should be positive.")

    reverberation_time_sabine = 0.161*volume/(absorption_area)

    return reverberation_time_sabine
