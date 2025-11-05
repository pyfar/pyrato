"""Module for room acoustics related functions.

Parametric room acoustics calculations using simple geometric considerations
such as Sabine's theory of sound in rooms.
"""
import numpy as np

def calculate_speed_of_sound(temperature):
    """Calculate the speed of sound in air depending on the temperature.

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
    ISO 9613-1 (Formula A.5)
    
    Equations
    ---------
    .. math::
        c=343.2*sqrt((temperature-t0)/(20-t0))\n
        t0=-273.15°C
    """
    speed_of_sound = 343.2 * np.sqrt((temperature + 273.15)/(20+ 273.15))
    return speed_of_sound

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
