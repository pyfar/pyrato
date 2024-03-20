#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The edc_noise_handling module provides various methods for noise
    compensation of room impulse responses.
"""

import numpy as np
from matplotlib import pyplot as plt
from pyrato import dsp
import warnings
import pyfar as pf


def subtract_noise_from_squared_rir(data, noise_level='auto'):
    """Subtract the noise power from a squared room impulse response.

    Parameters
    ----------
    data : pyfar.TimeData
        The squared room impulse response.
    noise_level : str or numpy.ndarray, float, optional
        The noise power for each channel. The default is 'auto', which will
        try to estimate the noise power from the room impulse response.

    Returns
    -------
    pyfar.TimeData
        The squared room impulse response after noise power subtraction.

    """
    subtracted = _subtract_noise_from_squared_rir(
        data.time, noise_level=noise_level)

    return pf.TimeData(subtracted, data.times, comment=data.comment)


def _subtract_noise_from_squared_rir(data, noise_level='auto'):
    """Subtract the noise power from a squared room impulse response.

    Parameters
    ----------
    data : ndarray, double
        The squared room impulse response with dimension [..., n_samples]

    Returns
    -------
    result : ndarray, double
        The noise-subtracted RIR.

    """
    if noise_level == "auto":
        noise_level = dsp._estimate_noise_energy(
            data,
            interval=[0.9, 1.0])
    return (data.T - noise_level).T


def schroeder_integration(room_impulse_response, is_energy=False):
    r"""Calculate the Schroeder integral of a room impulse response [#]_. The
    result is the energy decay curve for the given room impulse response.

    .. math:

        \langle e^2(t) \rangle = N\cdot \int_{t}^{\infty} h^2(\tau)
        \mathrm{d} \tau

    Parameters
    ----------
    room_impulse_response : pyfar.Signal
        Room impulse response as array
    is_energy : boolean, optional
        Whether the input represents energy data or sound pressure values.

    Returns
    -------
    energy_decay_curve : pyfar.TimeData
        The energy decay curve

    Note
    ----
    This function does not apply any compensation of measurement noise and
    integrates the full length of the input signal. It should only be used
    if no measurement noise or artifacts are present in the data.

    References
    ----------
    .. [#]  M. R. Schroeder, “New Method of Measuring Reverberation Time,”
            The Journal of the Acoustical Society of America, vol. 37, no. 6,
            pp. 1187-1187, 1965.

    Example
    -------
    Calculate the Schroeder integral of a simulated RIR and plot.

    .. plot::

        >>> import numpy as np
        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> from pyrato.analytic import rectangular_room_rigid_walls
        ...
        >>> L = np.array([8, 5, 3])/10
        >>> source_pos = np.array([5, 3, 1.2])/10
        >>> receiver_pos = np.array([1, 1, 1.2])/10
        >>> rir, _ = rectangular_room_rigid_walls(
        ...     L, source_pos, receiver_pos,
        ...     reverberation_time=1, max_freq=1e3, n_samples=2**16,
        ...     speed_of_sound=343.9)
        >>> edc = ra.schroeder_integration(rir)
        >>> pf.plot.time(rir/np.abs(rir.time).max(), dB=True, label='RIR')
        >>> ax = pf.plot.time(
        ...     edc/edc.time[..., 0], dB=True, log_prefix=10, label='EDC')
        >>> ax.set_ylim(-65, 5)
        >>> ax.legend()
    """

    edc = _schroeder_integration(
        room_impulse_response.time, is_energy=is_energy)

    return pf.TimeData(edc, room_impulse_response.times)


def _schroeder_integration(impulse_response, is_energy=False):
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
            pp. 1187-1187, 1965.

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


def energy_decay_curve_truncation(
        data,
        freq='broadband',
        noise_level='auto',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=15,
        plot=False):
    """ This function truncates a given room impulse response by the
    intersection time after Lundeby and calculates the energy decay curve.

    Parameters
    ----------
    data : pyfar.Signal
        The room impulse response.
    freq: integer OR string
        The frequency band. If set to 'broadband',
        the time window of the Lundeby-algorithm will not be set in dependence
        of frequency.
    noise_level: ndarray, double OR string
        If not specified, the noise level is calculated based on the last 10
        percent of the RIR. Otherwise specify manually for each channel
        as array.
    is_energy: boolean
        Defines, if the data is already squared.
    time_shift : boolean
        Defines, if the silence at beginning of the RIR should be removed.
    channel_independent : boolean
        Defines, if the time shift and normalization is done
        channel-independently or not.
    normalize : boolean
        Defines, if the energy decay curve should be normalized in the end
        or not.
    threshold : float
        Defines a peak-signal-to-noise ratio based threshold in dB for final
        truncation of the EDC. Values below the sum of the threshold level and
        the peak-signal-to-noise ratio in dB are discarded. The default is
        15 dB, which is in correspondence with ISO 3382-1:2009 [#]_.
    plot: Boolean
        Specifies, whether the results should be visualized or not.

    Returns
    -------
    pyfar.TimeData
        Returns the noise compensated EDC.

    Examples
    --------

    Plot the RIR and the EDC calculated truncating the integration at the
    intersection time.

    .. plot::

        >>> import numpy as np
        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> from pyrato.analytic import rectangular_room_rigid_walls
        ...
        >>> L = np.array([8, 5, 3])/10
        >>> source_pos = np.array([5, 3, 1.2])/10
        >>> receiver_pos = np.array([1, 1, 1.2])/10
        >>> rir, _ = rectangular_room_rigid_walls(
        ...     L, source_pos, receiver_pos,
        ...     reverberation_time=1, max_freq=1e3, n_samples=2**16,
        ...     speed_of_sound=343.9)
        >>> rir = rir/rir.time.max()
        ...
        >>> awgn = pf.signals.noise(
        ...     rir.n_samples, rms=rir.time.max()*10**(-50/20),
        ...     sampling_rate=rir.sampling_rate)
        >>> rir = rir + awgn
        >>> edc = ra.energy_decay_curve_truncation(rir)
        ...
        >>> ax = pf.plot.time(rir, dB=True, label='RIR')
        >>> pf.plot.time(edc, dB=True, log_prefix=10, label='EDC')
        >>> ax.set_ylim(-65, 5)
        >>> ax.legend()


    References
    ----------
    .. [#]  International Organization for Standardization, “EN ISO 3382-1:2009
            Acoustics - Measurement of room acoustic parameters,” 2009.

    """
    energy_data = dsp.preprocess_rir(
        data,
        is_energy=is_energy,
        shift=time_shift,
        channel_independent=channel_independent)
    n_samples = data.n_samples

    intersection_time = intersection_time_lundeby(
        energy_data,
        freq=freq,
        initial_noise_power=noise_level,
        is_energy=True,
        time_shift=False,
        channel_independent=False,
        plot=False)[0]

    intersection_time_idx = np.rint(intersection_time * data.sampling_rate)

    psnr = dsp.peak_signal_to_noise_ratio(
        data, noise_level, is_energy=is_energy)
    trunc_levels = 10*np.log10((psnr)) - threshold

    energy_decay_curve = np.zeros([*data.cshape, n_samples])
    for ch in np.ndindex(data.cshape):
        energy_decay_curve[
            ch, :int(intersection_time_idx[ch])] = \
                _schroeder_integration(
                    energy_data.time[
                        ch, :int(intersection_time_idx[ch])],
                    is_energy=True)

    energy_decay_curve = _truncate_energy_decay_curve(
        energy_decay_curve, trunc_levels)

    if normalize:
        # Normalize the EDC...
        if not channel_independent:
            # ...by the first element of each channel.
            energy_decay_curve = (energy_decay_curve.T /
                                  energy_decay_curve[..., 0]).T
        else:
            # ...by the maximum first element of each channel.
            max_start_value = np.amax(energy_decay_curve[..., 0])
            energy_decay_curve /= max_start_value

    edc = pf.TimeData(
        energy_decay_curve, data.times, comment=data.comment)

    if plot:
        ax = pf.plot.time(data, dB=True, label='RIR')
        pf.plot.time(edc, dB=True, log_prefix=10, label='EDC')
        ax.set_ylim(-65, 5)
        ax.legend()

    return edc


def energy_decay_curve_lundeby(
        data,
        freq='broadband',
        noise_level='auto',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False):
    """Lundeby et al. [#]_ proposed a correction term to prevent the truncation
    error. The missing signal energy from truncation time to infinity is
    estimated and added to the truncated integral.

    Parameters
    ----------
    data : pyfar.Signal
        The room impulse response.
    freq: integer OR string
        The frequency band. If set to 'broadband',
        the time window of the Lundeby-algorithm will not be set in dependence
        of frequency.
    noise_level: ndarray, double OR string
        If not specified, the noise level is calculated based on the last 10
        percent of the RIR. Otherwise specify manually for each channel
        as array.
    is_energy: boolean
        Defines, if the data is already squared.
    time_shift : boolean
        Defines, if the silence at beginning of the RIR should be removed.
    channel_independent : boolean
        Defines, if the time shift and normalizsation is done
        channel-independently or not.
    normalize : boolean
        Defines, if the energy decay curve should be normalized in the end
        or not.
    plot: Boolean
        Specifies, whether the results should be visualized or not.

    Returns
    -------
    pyfar.TimeData
        Returns the noise handeled edc.

    References
    ----------
    .. [#]  Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of
            Measurements in Room Acoustics - ACUSTICA Vol. 81 (1995)

    Examples
    --------

    Plot the RIR and the EDC calculated after Lundeby.

    .. plot::

        >>> import numpy as np
        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> from pyrato.analytic import rectangular_room_rigid_walls
        ...
        >>> L = np.array([8, 5, 3])/10
        >>> source_pos = np.array([5, 3, 1.2])/10
        >>> receiver_pos = np.array([1, 1, 1.2])/10
        >>> rir, _ = rectangular_room_rigid_walls(
        ...     L, source_pos, receiver_pos,
        ...     reverberation_time=1, max_freq=1e3, n_samples=2**16,
        ...     speed_of_sound=343.9)
        >>> rir = rir/rir.time.max()
        ...
        >>> awgn = pf.signals.noise(
        ...     rir.n_samples, rms=rir.time.max()*10**(-50/20),
        ...     sampling_rate=rir.sampling_rate)
        >>> rir = rir + awgn
        >>> edc = ra.energy_decay_curve_lundeby(rir)
        ...
        >>> ax = pf.plot.time(rir, dB=True, label='RIR')
        >>> pf.plot.time(edc, dB=True, log_prefix=10, label='EDC')
        >>> ax.set_ylim(-65, 5)
        >>> ax.legend()

    """

    energy_data = dsp.preprocess_rir(
        data,
        is_energy=is_energy,
        shift=time_shift,
        channel_independent=channel_independent)
    n_samples = energy_data.n_samples
    sampling_rate = data.sampling_rate

    intersection_time, late_reverberation_time, noise_estimation = \
        intersection_time_lundeby(
            energy_data,
            freq=freq,
            initial_noise_power=noise_level,
            is_energy=True,
            time_shift=False,
            channel_independent=False,
            plot=False)
    time_vector = data.times

    energy_decay_curve = np.zeros([*data.cshape, n_samples])

    for ch in np.ndindex(data.cshape):
        intersection_time_idx = np.argmin(
            np.abs(time_vector - intersection_time[ch]))
        p_square_at_intersection = noise_estimation[ch]

        # Calculate correction term according to DIN EN ISO 3382
        # TO-DO: check reference!
        correction = (p_square_at_intersection
                      * late_reverberation_time[ch]
                      * (1 / (6*np.log(10)))
                      * sampling_rate)

        energy_decay_curve[ch, :intersection_time_idx] = \
            _schroeder_integration(
                energy_data.time[ch, :intersection_time_idx],
                is_energy=True)
        energy_decay_curve[ch] += correction
        energy_decay_curve[ch, intersection_time_idx:] = np.nan

    if normalize:
        # Normalize the EDC...
        if not channel_independent:
            # ...by the first element of each channel.
            energy_decay_curve = (
                energy_decay_curve.T /
                energy_decay_curve[..., 0]).T
        else:
            # ...by the maximum first element of each channel.
            max_start_value = np.amax(energy_decay_curve[..., 0])
            energy_decay_curve /= max_start_value

    edc = pf.TimeData(
        energy_decay_curve, data.times, comment=data.comment)

    if plot:
        ax = pf.plot.time(data, dB=True, label='RIR')
        pf.plot.time(edc, dB=True, log_prefix=10, label='EDC')
        ax.set_ylim(-65, 5)
        ax.legend()

    return edc


def energy_decay_curve_chu(
        data,
        noise_level='auto',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        threshold=10,
        plot=False):
    """ Implementation of the "subtraction of noise"-method after Chu [#]
    The noise level is estimated and subtracted from the impulse response
    before backward integration.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    noise_level: ndarray, double OR string
        If not specified, the noise level is calculated based on the last 10
        percent of the RIR. Otherwise specify manually for each channel
        as array.
    is_energy: boolean
        Defines, if the data is already squared.
    time_shift : boolean
        Defines, if the silence at beginning of the RIR should be removed.
    channel_independent : boolean
        Defines, if the time shift and normalizsation is done
        channel-independently or not.
    normalize : boolean
        Defines, if the energy decay curve should be normalized in the end
        or not.
    threshold : float, None
        Defines a peak-signal-to-noise ratio based threshold in dB for final
        truncation of the EDC. Values below the sum of the threshold level and
        the peak-signal-to-noise ratio in dB are discarded. The default is
        10 dB. If `None`, the decay curve will not be truncated further.
    plot: Boolean
        Specifies, whether the results should be visualized or not.

    Returns
    -------
    energy_decay_curve: ndarray, double
        Returns the noise handeled edc.

    References
    ----------
    .. [#]  W. T. Chu. “Comparison of reverberation measurements using
            Schroeder’s impulse method and decay-curve averaging method”.
            In: Journal of the Acoustical Society of America 63.5 (1978),
            pp. 1444–1450.

    Examples
    --------

    .. plot::

        >>> import numpy as np
        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> from pyrato.analytic import rectangular_room_rigid_walls
        ...
        >>> L = np.array([8, 5, 3])/10
        >>> source_pos = np.array([5, 3, 1.2])/10
        >>> receiver_pos = np.array([1, 1, 1.2])/10
        >>> rir, _ = rectangular_room_rigid_walls(
        ...     L, source_pos, receiver_pos,
        ...     reverberation_time=1, max_freq=1e3, n_samples=2**16,
        ...     speed_of_sound=343.9)
        ...
        >>> awgn = pf.signals.noise(
        ...     rir.n_samples, rms=rir.time.max()*10**(-40/20),
        ...     sampling_rate=rir.sampling_rate)
        >>> rir = rir + awgn
        >>> edc = ra.energy_decay_curve_chu(rir)
        ...
        >>> pf.plot.time(rir/np.abs(rir.time).max(), dB=True, label='RIR')
        >>> ax = pf.plot.time(
        ...     edc/edc.time[..., 0], dB=True, log_prefix=10, label='EDC')
        >>> ax.set_ylim(-65, 5)
        >>> ax.legend()

    """
    energy_data = dsp.preprocess_rir(
        data,
        is_energy=is_energy,
        shift=time_shift,
        channel_independent=channel_independent)

    subtracted = subtract_noise_from_squared_rir(
        energy_data,
        noise_level=noise_level)

    edc = schroeder_integration(subtracted, is_energy=True)

    if normalize:
        # Normalize the EDC...
        if not channel_independent:
            # ...by the first element of each channel.
            edc.time = (edc.time.T / edc.time[..., 0]).T
        else:
            # ...by the maximum first element of all channels.
            max_start_value = np.amax(edc.time[..., 0])
            edc.time /= max_start_value

    mask = edc.time <= 2*np.finfo(float).eps
    if np.any(mask):
        first_zero = np.nanargmax(mask, axis=-1)
        for ch in np.ndindex(edc.cshape):
            edc.time[ch, first_zero[ch]:] = np.nan

    if threshold is not None:
        psnr = dsp.peak_signal_to_noise_ratio(
            data, noise_level, is_energy=is_energy)
        trunc_levels = 10*np.log10((psnr)) - threshold
        edc = truncate_energy_decay_curve(edc, trunc_levels)

    if plot:
        plt.figure(figsize=(15, 3))
        pf.plot.use('light')
        plt.subplot(131)
        pf.plot.time(energy_data, dB=True, log_prefix=10)
        plt.ylabel('Squared IR in dB')
        plt.subplot(132)
        pf.plot.time(subtracted, dB=True, log_prefix=10)
        plt.ylabel('Noise subtracted RIR in dB')
        plt.subplot(133)
        pf.plot.time(edc, dB=True, log_prefix=10)
        plt.ylabel('EDC in dB')

    return edc


def energy_decay_curve_chu_lundeby(
        data,
        freq='broadband',
        noise_level='auto',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False):
    """ This function combines Chu's and Lundeby's methods:
    The estimated noise level is subtracted before backward integration,
    the impulse response is truncated at the intersection time,
    and the correction for the truncation is applied [4]_, [5]_, [6]_

    Parameters
    ----------
    data : pyfar.Signal
        The room impulse response.
    freq: integer OR string
        The frequency band. If set to 'broadband',
        the time window of the Lundeby-algorithm will not be set in dependence
        of frequency.
    noise_level: ndarray, double OR string
        If not specified, the noise level is calculated based on the last 10
        percent of the RIR. Otherwise specify manually for each channel
        as array.
    is_energy: boolean
        Defines, if the data is already squared.
    time_shift : boolean
        Defines, if the silence at beginning of the RIR should be removed.
    channel_independent : boolean
        Defines, if the time shift and normalizsation is done
        channel-independently or not.
    normalize : boolean
        Defines, if the energy decay curve should be normalized in the end
        or not.
    plot: Boolean
        Specifies, whether the results should be visualized or not.

    Returns
    -------
    pyfar.TimeData
        Returns the noise handeled edc.

    References
    ----------
    .. [4]  Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of
            Measurements in Room Acoustics - ACUSTICA Vol. 81 (1995)
    .. [5]  W. T. Chu. “Comparison of reverberation measurements using
            Schroeder’s impulse method and decay-curve averaging method”. In:
            Journal of the Acoustical Society of America 63.5 (1978),
            pp. 1444–1450.
    .. [6]  M. Guski, “Influences of external error sources on measurements of
            room acoustic parameters,” 2015.

    Examples
    --------

    Calculate and plot the EDC using a combination of Chu's and Lundeby's
    methods.

    .. plot::

        >>> import numpy as np
        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> from pyrato.analytic import rectangular_room_rigid_walls
        ...
        >>> L = np.array([8, 5, 3])/10
        >>> source_pos = np.array([5, 3, 1.2])/10
        >>> receiver_pos = np.array([1, 1, 1.2])/10
        >>> rir, _ = rectangular_room_rigid_walls(
        ...     L, source_pos, receiver_pos,
        ...     reverberation_time=1, max_freq=1e3, n_samples=2**16,
        ...     speed_of_sound=343.9)
        >>> rir = rir/rir.time.max()
        ...
        >>> awgn = pf.signals.noise(
        ...     rir.n_samples, rms=rir.time.max()*10**(-50/20),
        ...     sampling_rate=rir.sampling_rate)
        >>> rir = rir + awgn
        >>> edc = ra.energy_decay_curve_chu_lundeby(rir)
        ...
        >>> ax = pf.plot.time(rir, dB=True, label='RIR')
        >>> pf.plot.time(edc, dB=True, log_prefix=10, label='EDC')
        >>> ax.set_ylim(-65, 5)
        >>> ax.legend()

    """

    energy_data = dsp.preprocess_rir(
        data,
        is_energy=is_energy,
        shift=time_shift,
        channel_independent=channel_independent)
    n_samples = energy_data.n_samples

    subtraction = subtract_noise_from_squared_rir(
        energy_data,
        noise_level=noise_level)

    intersection_time, late_reverberation_time, noise_level = \
        intersection_time_lundeby(
            energy_data,
            freq=freq,
            initial_noise_power=noise_level,
            is_energy=True,
            time_shift=False,
            channel_independent=False,
            plot=False)

    time_vector = data.times
    energy_decay_curve = np.zeros([*data.cshape, n_samples])

    for ch in np.ndindex(data.cshape):
        intersection_time_idx = np.argmin(np.abs(
            time_vector - intersection_time[ch]))
        if type(noise_level) is str and noise_level == 'auto':
            p_square_at_intersection = dsp.estimate_noise_energy(
                energy_data.time[ch], is_energy=True)
        else:
            p_square_at_intersection = noise_level[ch]

        # calculate correction term according to DIN EN ISO 3382
        correction = (p_square_at_intersection
                      * late_reverberation_time[ch]
                      * (1 / (6*np.log(10)))
                      * data.sampling_rate)

        energy_decay_curve[ch, :intersection_time_idx] = \
            _schroeder_integration(
                subtraction.time[ch, :intersection_time_idx],
                is_energy=True)
        energy_decay_curve[ch] += correction
        energy_decay_curve[ch, intersection_time_idx:] = np.nan

    if normalize:
        # Normalize the EDC...
        if not channel_independent:
            # ...by the first element of each channel.
            energy_decay_curve = (energy_decay_curve.T /
                                  energy_decay_curve[..., 0]).T
        else:
            # ...by the maximum first element of each channel.
            max_start_value = np.amax(energy_decay_curve[..., 0])
            energy_decay_curve /= max_start_value

    edc = pf.TimeData(
        energy_decay_curve, data.times, comment=data.comment)

    if plot:
        ax = pf.plot.time(data, dB=True, label='RIR')
        pf.plot.time(edc, dB=True, log_prefix=10, label='EDC')
        ax.set_ylim(-65, 5)
        ax.legend()

    return edc


def intersection_time_lundeby(
        data,
        freq='broadband',
        initial_noise_power='auto',
        is_energy=False,
        time_shift=False,
        channel_independent=False,
        plot=False):
    """Calculate the intersection time between impulse response and noise.

    This function uses the algorithm after Lundeby et al. [#]_ to calculate
    the intersection time, lundeby reverberation time, and noise level
    estimation.

    Parameters
    ----------
    data : pyfar.Signal
        The room impulse response
    freq: integer OR string
        The frequency band. If set to 'broadband',
        the time window of the Lundeby-algorithm will not be set in dependence
        of frequency.
    noise_level: ndarray, double OR string
        If not specified, the noise level is calculated based on the last 10
        percent of the RIR. Otherwise specify manually for each channel
        as array.
    is_energy: boolean
        Defines, if the data is already squared.
    time_shift : boolean
        Defines, if the silence at beginning of the RIR should be removed.
    channel_independent : boolean
        Defines, if the time shift and normalizsation is done
        channel-independently or not.
    plot: Boolean
        Specifies, whether the results should be visualized or not.

    Returns
    -------
    intersection_time: ndarray, float
        Returns the Lundeby intersection time for each channel.
    reverberation_time: ndarray, float
        Returns the Lundeby reverberation time for each channel.
    noise_level: ndarray, float
        Returns the noise level estimation for each channel.

    References
    ----------
    .. [#]  Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of
            Measurements in Room Acoustics - ACUSTICA Vol. 81 (1995)

    Examples
    --------

    Estimate the intersection time :math:`T_i` and plot the RIR and the
    estimated noise power.

    .. plot::

        >>> import numpy as np
        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> from pyrato.analytic import rectangular_room_rigid_walls
        ...
        >>> L = np.array([8, 5, 3])/10
        >>> source_pos = np.array([5, 3, 1.2])/10
        >>> receiver_pos = np.array([1, 1, 1.2])/10
        >>> rir, _ = rectangular_room_rigid_walls(
        ...     L, source_pos, receiver_pos,
        ...     reverberation_time=1, max_freq=1e3, n_samples=2**16,
        ...     speed_of_sound=343.9)
        >>> rir = rir/np.abs(rir.time).max()
        ...
        >>> awgn = pf.signals.noise(
        ...     rir.n_samples, rms=rir.time.max()*10**(-40/20),
        ...     sampling_rate=rir.sampling_rate)
        >>> rir = rir + awgn
        >>> inter_time, _, noise_power = ra.intersection_time_lundeby(rir)
        ...
        >>> ax = pf.plot.time(rir, dB=True, label='RIR')
        >>> ax.axvline(inter_time, c='k', linestyle='--', label='$T_i$')
        >>> ax.axhline(
        ...     10*np.log10(noise_power), c='k', linestyle=':', label='Noise')
        >>> ax.set_ylim(-65, 5)
        >>> ax.legend()

    """
    # Define constants:
    # time intervals per 10 dB decay. Lundeby: 3...10
    n_intervals_per_10dB = 5
    # end of regression 5 ... 10 dB
    dB_above_noise = 10
    # Dynamic range 10 ... 20 dB
    use_dyn_range_for_regression = 20

    energy_data = dsp.preprocess_rir(
        data,
        is_energy=is_energy,
        shift=time_shift,
        channel_independent=channel_independent)

    if isinstance(data, pf.Signal):
        sampling_rate = data.sampling_rate
    elif isinstance(data, pf.TimeData) and not isinstance(data, pf.Signal):
        sampling_rate = np.round(1/np.diff(data.times).mean(), decimals=4)
    energy_data = energy_data.time

    if freq == "broadband":
        # broadband: use 30 ms windows sizes
        freq_dependent_window_time = 0.03
    else:
        freq_dependent_window_time = (800/freq+10) / 1000

    # (1) SMOOTH
    time_window_data, time_vector_window, time_vector = dsp._smooth_rir(
        energy_data, sampling_rate, freq_dependent_window_time)

    # (2) ESTIMATE NOISE
    if initial_noise_power == 'auto':
        noise_estimation = dsp._estimate_noise_energy(energy_data)
    else:
        noise_estimation = initial_noise_power.copy()

    # (3) REGRESSION
    reverberation_time = np.zeros(data.cshape, data.time.dtype)
    noise_level = np.zeros(data.cshape, data.time.dtype)
    intersection_time = np.zeros(data.cshape, data.time.dtype)
    noise_peak_level = np.zeros(data.cshape, data.time.dtype)

    for ch in np.ndindex(data.cshape):
        time_window_data_current_channel = time_window_data[ch]
        start_idx = np.nanargmax(time_window_data_current_channel, axis=-1)
        try:
            stop_idx = (np.argwhere(10*np.log10(
                time_window_data_current_channel[start_idx+1:-1]) >
                    (10*np.log10(noise_estimation[ch]) +
                        dB_above_noise))[-1, 0] + start_idx)
        except IndexError:
            raise Exception(
                'Regression failed: Low SNR. Estimation terminated.')

        dyn_range = np.diff(10*np.log10(np.take(
            time_window_data_current_channel, [start_idx, stop_idx])))

        if (stop_idx == start_idx) or dyn_range > -5:
            raise Exception(
                'Regression failed: Low SNR. Estimation terminated.')

        # regression_matrix*slope = edc
        regression_matrix = np.vstack((np.ones(
            [stop_idx-start_idx]), time_vector_window[start_idx:stop_idx]))
        slope = np.linalg.lstsq(
            regression_matrix.T,
            10*np.log10(time_window_data_current_channel[start_idx:stop_idx]),
            rcond=None)[0]

        if slope[1] == 0 or np.any(np.isnan(slope)):
            raise Exception(
                'Regression failed: T would be Inf, setting to 0. \
                    Estimation terminated.')

        regression_time = np.array(
            [time_vector_window[start_idx], time_vector_window[stop_idx]])
        regression_values = np.array(
            [10*np.log10(time_window_data[0, start_idx]),
             (10*np.log10(time_window_data[0, start_idx])
                + slope[1]*time_vector_window[stop_idx])])

        # (4) PRELIMINARY CROSSING POINT
        crossing_point = \
            (10*np.log10(noise_estimation[ch]) - slope[0]) / slope[1]
        preliminary_crossing_point = crossing_point

        # (5) NEW LOCAL TIME INTERVAL LENGTH
        n_blocks_in_decay = (np.diff(
            10*np.log10(np.take(
                time_window_data_current_channel, [start_idx, stop_idx])))
            / -10 * n_intervals_per_10dB)

        n_samples_per_block = np.round(np.diff(np.take(
            time_vector_window,
            [start_idx, stop_idx])) / n_blocks_in_decay * sampling_rate)

        window_time = n_samples_per_block/sampling_rate

        # (6) AVERAGE
        time_window_data_current_channel, \
            time_vector_window_current_channel, \
            time_vector_current_channel = dsp._smooth_rir(
                energy_data[ch], sampling_rate, window_time)
        time_window_data_current_channel = np.squeeze(
            time_window_data_current_channel)
        idx_max = np.nanargmax(time_window_data_current_channel)

        # high start value to enter while-loop
        old_crossing_point = 11+crossing_point
        loop_counter = 0

        while True:
            # (7) ESTIMATE BACKGROUND LEVEL
            corresponding_decay = 10  # 5...10 dB
            idx_last_10_percent = np.round(
                time_window_data_current_channel.shape[-1]*0.9)

            t_block = n_samples_per_block / sampling_rate
            rel_decay = corresponding_decay / slope[1]
            idx_10dB_below_crosspoint = np.nanmax(
                np.r_[1, np.round(((crossing_point - rel_decay) / t_block))])

            noise_estimation_current_channel = np.nanmean(
                time_window_data_current_channel[int(np.nanmin(
                    [idx_last_10_percent, idx_10dB_below_crosspoint])):])

            # (8) ESTIMATE LATE DECAY SLOPE
            try:
                start_idx_loop = np.argwhere(10*np.log10(
                    time_window_data_current_channel[idx_max:]) < (
                        10*np.log10(noise_estimation_current_channel)
                        + dB_above_noise
                        + use_dyn_range_for_regression
                        + idx_max))[0, 0]
            except TypeError:
                start_idx_loop = 0

            try:
                stop_idx_loop = (np.argwhere(10*np.log10(
                    time_window_data_current_channel[start_idx_loop+1:]) < (
                        10*np.log10(noise_estimation_current_channel)
                        + dB_above_noise))[0, 0]
                                 + start_idx_loop)
            except IndexError:
                raise Exception(
                    'Regression failed: Low SNR. Estimation terminated.')

            # regression_matrix*slope = edc
            regression_matrix = np.vstack((np.ones(
                [stop_idx_loop-start_idx_loop]),
                time_vector_window_current_channel[
                    start_idx_loop:stop_idx_loop]))

            slope = np.linalg.lstsq(
                regression_matrix.T,
                (10*np.log10(time_window_data_current_channel[
                    start_idx_loop:stop_idx_loop])),
                rcond=None)[0]

            if slope[1] >= 0:
                raise Exception(
                    'Regression did not work due, T would be Inf, \
                        setting to 0. Estimation was terminated.')

            # (9) FIND CROSSPOINT
            old_crossing_point = crossing_point
            crossing_point = ((10*np.log10(noise_estimation_current_channel)
                               - slope[0]) / slope[1])

            loop_counter = loop_counter + 1

            if (np.abs(old_crossing_point-crossing_point) < 0.01):
                break
            if loop_counter > 30:
                # TO-DO: Paper says 5 iterations are sufficient in all cases!
                warnings.warn(
                    "Lundeby algorithm was terminated after 30 iterations.")
                break

        reverberation_time[ch] = -60/slope[1]
        noise_level[ch] = noise_estimation_current_channel
        intersection_time[ch] = crossing_point
        noise_peak_level[ch] = 10 * np.log10(
            np.nanmax(time_window_data_current_channel[int(np.nanmin(
                [idx_last_10_percent, idx_10dB_below_crosspoint])):]))

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        max_data_db = np.nanmax(10*np.log10(energy_data))
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared RIR [dB]')
        plt.ylim(bottom=max_data_db-80+5, top=max_data_db+5)
        plt.grid(True)

        plt.subplot(132)
        plt.plot(time_vector_window, 10*np.log10(time_window_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Smoothened RIR [dB]')
        plt.plot(
            time_vector_window[-int(np.round(time_window_data.shape[-1]/10))],
            10*np.log10(noise_estimation[0]),
            marker='o',
            label='noise estimation',
            linestyle='--',
            color='C1')
        plt.axhline(
                10*np.log10(noise_estimation[0]),
                color='C1',
                linestyle='--')
        plt.plot(
            regression_time,
            regression_values,
            marker='o',
            color='C2',
            label='regression')
        plt.plot(
            preliminary_crossing_point,
            10*np.log10(noise_estimation[ch]),
            marker='o',
            color='C3',
            label='preliminary crosspoint')
        plt.legend()
        plt.grid(True)

        plt.subplot(133)
        # TO-DO: plot all channels?
        plt.plot(
            time_vector_window_current_channel,
            10*np.log10(time_window_data_current_channel))
        plt.xlabel('Time [s]')
        plt.ylabel('Final EDC Estimation [dB]')
        plt.plot(
            time_vector_window_current_channel[int(idx_10dB_below_crosspoint)],
            10*np.log10(noise_estimation[0]),
            marker='o',
            color='C1',
            label='Noise')
        plt.axhline(
            10*np.log10(noise_estimation[0]),
            color='C1',
            linestyle='--')
        plt.plot(
            crossing_point,
            10*np.log10(noise_estimation[0]),
            marker='o',
            color='C3',
            label='Final crosspoint',
            linestyle='--')
        plt.axvline(
            reverberation_time[0],
            label=('$T_{60} = '+str(np.round(reverberation_time[0], 3)) + '$'),
            color='k')
        plt.axvline(
            intersection_time[0],
            label=('Lundeby intersection time'),
            color='C3',
            linestyle='--')
        plt.tight_layout()
        plt.legend()
        plt.grid(True)

    return intersection_time, reverberation_time, noise_level


def _truncate_energy_decay_curve(energy_decay_curve, threshold_level):

    edc = np.atleast_2d(energy_decay_curve)
    threshold_level = np.atleast_2d(threshold_level)
    e = edc.T[0]
    t = e / 10**(threshold_level/10)

    mask = edc.T < np.broadcast_to(t, edc.T.shape)
    edc[mask.T] = np.nan

    return edc


def truncate_energy_decay_curve(energy_decay_curve, threshold):
    """Truncate an energy decay curve, discarding values below the threshold.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        The energy decay curve
    threshold : float
        The threshold level in dB. The data below the threshold level are set
        to numpy.nan values.
    """
    return pf.TimeData(
        _truncate_energy_decay_curve(energy_decay_curve.time, threshold),
        energy_decay_curve.times,
        energy_decay_curve.comment)
