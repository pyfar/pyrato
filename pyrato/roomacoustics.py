# -*- coding: utf-8 -*-

"""Main module."""

import re
import numpy as np
import matplotlib.pyplot as plt


def reverberation_time_energy_decay_curve(
        energy_decay_curve,
        times,
        T='T20',
        normalize=True,
        plot=False):
    """Estimate the reverberation time from a given energy decay curve
    according to the ISO standard 3382 [#]_.

    Parameters
    ----------
    energy_decay_curve : ndarray, double
        Energy decay curve. The time needs to be the arrays last dimension.
    times : ndarray, double
        Time vector corresponding to each sample of the EDC.
    T : 'T20', 'T30', 'T40', 'T50', 'T60', 'EDT', 'LDT'
        Decay interval to be used for the reverberation time extrapolation. EDT
        corresponds to the early decay time extrapolated from the interval
        [0, -10] dB, LDT corresponds to the late decay time extrapolated from
        the interval [-25, -35] dB.
    normalize : bool, True
        Normalize the EDC to the steady state energy level
    plot : bool, False
        Plot the estimated extrapolation line for visual inspection of the
        results.

    Returns
    -------
    reverberation_time : double
        The reverberation time

    References
    ----------
    .. [#]  ISO 3382, Acoustics - Measurement of the reverberation time of
            rooms with reference to other acoustical parameters.

    """
    intervals = [20, 30, 40, 50, 60]

    if T == 'EDT':
        upper = -0.1
        lower = -10.1
    elif T == 'LDT':
        upper = -25.
        lower = -35.
    else:
        try:
            (int(re.findall(r'\d+', T)[0]) in intervals)
        except IndexError:
            raise ValueError(
                "{} is not a valid interval for the regression.".format(T))

        upper = -5
        lower = -np.double(re.findall(r'\d+', T)) + upper

    if normalize:
        energy_decay_curve /= energy_decay_curve[0]

    edc_db = 10*np.log10(np.abs(energy_decay_curve))

    idx_upper = np.nanargmin(np.abs(upper - edc_db))
    idx_lower = np.nanargmin(np.abs(lower - edc_db))

    A = np.vstack(
        [times[idx_upper:idx_lower], np.ones(idx_lower - idx_upper)]).T
    gradient, const = np.linalg.lstsq(
        A, edc_db[..., idx_upper:idx_lower], rcond=None)[0]

    reverberation_time = -60 / gradient

    if plot:
        plt.figure()
        plt.plot(
            times,
            edc_db,
            label='edc')
        plt.plot(
            times,
            times * gradient + const,
            label='regression',
            linestyle='-.')
        ax = plt.gca()
        ax.set_ylim((-95, 5))

        reverberation_time = -60 / gradient

        ax.set_xlim((-0.05, 2*reverberation_time))
        plt.grid(True)
        plt.legend()
        ax.set_ylabel('EDC [dB]')
        ax.set_xlabel('Time [s]')

    return reverberation_time


def schroeder_integration(impulse_response, is_energy=False):
    r"""Calculate the Schroeder integral of a room impulse response [#]_. The
    result is the energy decay curve for the given room impulse response.

    .. math:

        \langle e^2(t) \rangle = N\cdot \int_{t}^{\infty} h^2(\tau)
        \mathrm{d} \tau

    Parameters
    ----------
    impulse_response : ndarray, double
        Room impulse response as array
    is_energy : boolean, optional
        Whether the input represents energy data or sound pressure values.

    Returns
    -------
    energy_decay_curve : ndarray, double
        The energy decay curve

    References
    ----------
    .. [#]  M. R. Schroeder, “New Method of Measuring Reverberation Time,”
            The Journal of the Acoustical Society of America, vol. 37, no. 6,
            pp. 1187–1187, 1965.

    """
    if not is_energy:
        data = np.abs(impulse_response)**2
    else:
        data = impulse_response.copy()

    ndim = data.ndim
    data = np.atleast_2d(data)
    energy_decay_curve = np.fliplr(np.nancumsum(np.fliplr(data), axis=-1))

    if ndim < energy_decay_curve.ndim:
        energy_decay_curve = np.squeeze(energy_decay_curve)

    return energy_decay_curve


def energy_decay_curve_analytic(
        surfaces, alphas, volume, times, source=None,
        receiver=None, method='eyring', c=343.4, frequency=None,
        air_absorption=True):
    """Calculate the energy decay curve analytically by using Eyring's or
    Sabine's equation [#]_.

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

    Returns
    -------
    energy_decay_curve : ndarray, double
        The energy decay curve

    References
    ----------
    .. [#]  H. Kuttruff, Room acoustics, 4th Ed. Taylor & Francis, 2009.

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
