#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The edc_noise_handling module provides various methods for noise
    compensation of room impulse responses.
"""

import numpy as np
from matplotlib import pyplot as plt
from pyrato import dsp
import pyrato as ra
import warnings


def estimate_noise_energy(
        data,
        interval=[0.9, 1.0],
        is_energy=False):
    """ This function estimates the noise energy level of a given room impulse
    response. The noise is assumed to be Gaussian.

    Parameters
    ----------
    data: np.array
        The room impulse response with dimension [..., n_samples]
    interval : tuple, float, [0.9, 1.]
        Defines the interval of the RIR to be evaluated for the estimation.
        The interval is relative to the length of the RIR [0 = 0%, 1=100%)]
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    noise_energy: float
        The energy of the background noise.
    """

    energy_data = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=False,
        channel_independent=False)[0]

    if np.any(energy_data) < 0:
        raise ValueError("Energy is negative, check your input signal.")

    region_start_idx = int(energy_data.shape[-1]*interval[0])
    region_end_idx = int(energy_data.shape[-1]*interval[1])
    mask = np.arange(region_start_idx, region_end_idx)
    noise_energy = np.nanmean(np.take(energy_data, mask, axis=-1), axis=-1)

    return noise_energy


def preprocess_rir(
        data,
        is_energy=False,
        time_shift=False,
        channel_independent=False):
    """ Preprocess the room impulse response for further processing:
        - Square data
        - Shift the RIR to the first sample of the array, compensating for the
          delay of the time of arrival of the direct sound. The time shift is
          performed as a non-cyclic shift, adding numpy.nan values in the end
          of the RIR corresponding to the number of samples the data is
          shifted by.
        - The time shift can be done channel-independent or not.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    is_energy : boolean
        Defines, if the data is already squared.
    time_shift : boolean
        Defines, if the silence at beginning of the RIR should be removed.
    channel_independent : boolean
        Defines, if the time shift is done channel-independent or not.

    Returns
    -------
    energy_data : ndarray, double
        The preprocessed RIR
    n_channels : integer
        The number of channels of the RIR
    data_shape : list, integer
        The original data shape.

    """
    n_samples = data.shape[-1]
    data = np.atleast_2d(data)
    n_channels = np.prod(data.shape[:-1])

    data_shape = list(data.shape)
    data = np.reshape(data, (-1, n_samples))

    if time_shift:
        rir_start_idx = dsp.find_impulse_response_start(data)

        if channel_independent and not n_channels == 1:
            shift_samples = -rir_start_idx
        else:
            min_shift = np.amin(rir_start_idx)
            shift_samples = np.asarray(
                -min_shift * np.ones(n_channels), dtype=int)

        result = dsp.time_shift(
            data, shift_samples, circular_shift=False, keepdims=True)

    else:
        result = data
    if not is_energy:
        energy_data = np.abs(result)**2
    else:
        energy_data = result.copy()

    return energy_data, n_channels, data_shape


def smooth_rir(
        data,
        sampling_rate,
        smooth_block_length=0.075):
    """ Smoothens the RIR by averaging the data in an specified interval.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    sampling_rate: integer
        Defines the sampling rate of the room impulse response.
    smooth_block_length : double
        Defines the block-length of the smoothing algorithm in seconds.

    Returns
    -------
    time_window_data : ndarray, double
        The smoothed RIR.
    time_vector_window : ndarray, double
        The respective time vector fitting the smoothed data.
    time_vector : ndarray, double
        The time vector fitting the original data.

    """
    data = np.atleast_2d(data)
    n_samples = data.shape[-1]
    n_samples_nan = np.count_nonzero(np.isnan(data), axis=-1)

    n_samples_per_block = int(np.round(smooth_block_length * sampling_rate, 0))
    n_blocks = np.asarray(
        np.floor((n_samples-n_samples_nan)/n_samples_per_block),
        dtype=int)

    n_blocks_min = int(np.min(n_blocks))
    n_samples_actual = int(n_blocks_min*n_samples_per_block)
    reshaped_array = np.reshape(
        data[..., :n_samples_actual],
        (-1, n_blocks_min, n_samples_per_block))
    time_window_data = np.mean(reshaped_array, axis=-1)

    # Use average time instances corresponding to the average energy level
    # instead of time for the first sample of the block
    time_vector_window = \
        ((0.5+np.arange(0, n_blocks_min)) * n_samples_per_block/sampling_rate)

    # Use the time corresponding to the sampling of the original data
    time_vector = (np.arange(0, n_samples))/sampling_rate

    return time_window_data, time_vector_window, time_vector


def subtract_noise_from_squared_rir(data, noise_level='auto'):
    """ Subtracts the noise energy level from the squared RIR. Note, that the
        RIR has to be squared before.

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
        noise_level = estimate_noise_energy(
            data,
            is_energy=True,
            interval=[0.9, 1.0])
    return (data.T - noise_level).T


def energy_decay_curve_truncation(
        data,
        sampling_rate,
        freq='broadband',
        noise_level='auto',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False):
    """ This function truncates a given room impulse response by the
    intersection time after Lundeby and calculates the energy decay curve.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    sampling_rate: integer
        The sampling rate of the room impulse response.
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
    plot: Boolean
        Specifies, whether the results should be visualized or not.

    Returns
    -------
    energy_decay_curve: ndarray, double
        Returns the noise handeled edc.
    """
    energy_data, n_channels, data_shape = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=time_shift,
        channel_independent=channel_independent)
    n_samples = energy_data.shape[-1]

    intersection_time = intersection_time_lundeby(
        energy_data,
        sampling_rate=sampling_rate,
        freq=freq,
        initial_noise_power=noise_level,
        is_energy=True,
        time_shift=False,
        channel_independent=False,
        plot=False)[0]
    time_vector = smooth_rir(energy_data, sampling_rate)[2]

    intersection_time_idx = np.rint(intersection_time * sampling_rate)

    energy_decay_curve = np.zeros([n_channels, n_samples])
    for idx_channel in range(0, n_channels):
        energy_decay_curve[
            idx_channel, :int(intersection_time_idx[idx_channel])] = \
                ra.schroeder_integration(
                    energy_data[
                        idx_channel, :int(intersection_time_idx[idx_channel])],
                    is_energy=True)

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

    # Recover original data shape:
    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)
    energy_decay_curve = np.squeeze(energy_decay_curve)

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(121)
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(122)
        plt.plot(time_vector[0:energy_decay_curve.shape[-1]], 10*np.log10(
            energy_decay_curve.T))
        plt.xlabel('Time [s]')
        plt.ylabel('EDC: Truncation method [dB]')
        plt.tight_layout()

    return energy_decay_curve


def energy_decay_curve_lundeby(
        data,
        sampling_rate,
        freq='broadband',
        noise_level='auto',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False):
    """ Lundeby et al. [#]_ proposed a correction term to prevent the
    truncation error. The missing signal energy from truncation time to
    infinity is estimated and added to the truncated integral.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    sampling_rate: integer
        The sampling rate of the room impulse response.
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
    energy_decay_curve: ndarray, double
        Returns the noise handeled edc.

    References
    ----------
    .. [#]  Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of
            Measurements in Room Acoustics - ACUSTICA Vol. 81 (1995)
    """

    energy_data, n_channels, data_shape = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=time_shift,
        channel_independent=channel_independent)
    n_samples = energy_data.shape[-1]
    intersection_time, late_reverberation_time, noise_estimation = \
        intersection_time_lundeby(
            energy_data,
            sampling_rate=sampling_rate,
            freq=freq,
            initial_noise_power=noise_level,
            is_energy=True,
            time_shift=False,
            channel_independent=False,
            plot=False)
    time_vector = smooth_rir(energy_data, sampling_rate)[2]

    energy_decay_curve = np.zeros([n_channels, n_samples])

    for idx_channel in range(0, n_channels):
        intersection_time_idx = np.argmin(
            np.abs(time_vector - intersection_time[idx_channel]))
        p_square_at_intersection = noise_estimation[idx_channel]

        # Calculate correction term according to DIN EN ISO 3382
        # TO-DO: check reference!
        correction = (p_square_at_intersection
                      * late_reverberation_time[idx_channel]
                      * (1 / (6*np.log(10)))
                      * sampling_rate)

        energy_decay_curve[idx_channel, :intersection_time_idx] = \
            ra.schroeder_integration(
                energy_data[idx_channel, :intersection_time_idx],
                is_energy=True)
        energy_decay_curve[idx_channel] += correction

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

    energy_decay_curve[..., intersection_time_idx:] = np.nan

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(121)
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(122)
        plt.plot(time_vector[0:energy_decay_curve.shape[-1]], 10*np.log10(
            energy_decay_curve.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Tr. EDC with correction [dB]')
        plt.tight_layout()

    # Recover original data shape:
    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)
    energy_decay_curve = np.squeeze(energy_decay_curve)

    return energy_decay_curve


def energy_decay_curve_chu(
        data,
        sampling_rate,
        freq='broadband',
        noise_level='auto',
        is_energy=False,
        time_shift=True,
        channel_independent=False,
        normalize=True,
        plot=False):

    """ Implementation of the "subtraction of noise"-method after Chu [#]
    The noise level is estimated and subtracted from the impulse response
    before backward integration.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    sampling_rate: integer
        The sampling rate of the room impulse response.
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
    energy_decay_curve: ndarray, double
        Returns the noise handeled edc.

    References
    ----------
    .. [#]  W. T. Chu. “Comparison of reverberation measurements using
            Schroeder’s impulse method and decay-curve averaging method”.
            In: Journal of the Acoustical Society of America 63.5 (1978),
            pp. 1444–1450.
    """

    energy_data, n_channels, data_shape = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=time_shift,
        channel_independent=channel_independent)

    subtracted = subtract_noise_from_squared_rir(
        energy_data,
        noise_level=noise_level)

    energy_decay_curve = ra.schroeder_integration(
        subtracted,
        is_energy=True)

    energy_decay_curve = np.atleast_2d(energy_decay_curve)

    if normalize:
        # Normalize the EDC...
        if not channel_independent:
            # ...by the first element of each channel.
            energy_decay_curve = (energy_decay_curve.T
                                  / energy_decay_curve[..., 0]).T
        else:
            # ...by the maximum first element of all channels.
            max_start_value = np.amax(energy_decay_curve[..., 0])
            energy_decay_curve /= max_start_value

    mask = energy_decay_curve <= 2*np.finfo(float).eps
    if np.any(mask):
        first_zero = np.nanargmax(mask, axis=-1)
        for channel in range(n_channels):
            energy_decay_curve[channel, first_zero[channel]:] = np.nan

    if plot:
        time_vector = (0.5+np.arange(0, energy_data.shape[-1]))/sampling_rate
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.grid(True)
        plt.subplot(132)
        plt.plot(time_vector, 10*np.log10(subtracted.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Noise subtracted IR [dB]')
        plt.grid(True)
        plt.subplot(133)
        plt.plot(time_vector, 10*np.log10(energy_decay_curve.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Noise-handeled EDC [dB]')
        plt.grid(True)
        plt.tight_layout()

    # Recover original data shape:
    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)
    energy_decay_curve = np.squeeze(energy_decay_curve)

    return energy_decay_curve


def energy_decay_curve_chu_lundeby(
        data,
        sampling_rate,
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
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    sampling_rate: integer
        The sampling rate of the room impulse response.
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
    energy_decay_curve: ndarray, double
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
    """

    energy_data, n_channels, data_shape = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=time_shift,
        channel_independent=channel_independent)
    n_samples = energy_data.shape[-1]

    subtraction = subtract_noise_from_squared_rir(
        energy_data,
        noise_level=noise_level)

    intersection_time, late_reverberation_time, noise_level = \
        intersection_time_lundeby(
            energy_data,
            sampling_rate=sampling_rate,
            freq=freq,
            initial_noise_power=noise_level,
            is_energy=True,
            time_shift=False,
            channel_independent=False,
            plot=False)

    time_vector = smooth_rir(energy_data, sampling_rate)[2]
    energy_decay_curve = np.zeros([n_channels, n_samples])

    for idx_channel in range(0, n_channels):
        intersection_time_idx = np.argmin(np.abs(
            time_vector - intersection_time[idx_channel]))
        if noise_level == 'auto':
            p_square_at_intersection = estimate_noise_energy(
                energy_data[idx_channel], is_energy=True)
        else:
            p_square_at_intersection = noise_level[idx_channel]

        # calculate correction term according to DIN EN ISO 3382
        correction = (p_square_at_intersection
                      * late_reverberation_time[idx_channel]
                      * (1 / (6*np.log(10)))
                      * sampling_rate)

        energy_decay_curve[idx_channel, :intersection_time_idx] = \
            ra.schroeder_integration(
                subtraction[idx_channel, :intersection_time_idx],
                is_energy=True)
        energy_decay_curve[idx_channel] += correction

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

    energy_decay_curve[..., intersection_time_idx:] = np.nan

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(132)
        plt.plot(time_vector, 10*np.log10(subtraction.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Noise subtracted IR [dB]')
        plt.subplot(133)
        plt.plot(time_vector[0:energy_decay_curve.shape[-1]], 10*np.log10(
            energy_decay_curve.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Tr. EDC with corr. & subt. [dB]')
        plt.tight_layout()

    # Recover original data shape:
    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)
    energy_decay_curve = np.squeeze(energy_decay_curve)

    return energy_decay_curve


def intersection_time_lundeby(
        data,
        sampling_rate,
        freq='broadband',
        initial_noise_power='auto',
        is_energy=False,
        time_shift=False,
        channel_independent=False,
        plot=False):

    """ This function uses the algorithm after Lundeby et al. [#]_ to calculate
    the intersection time, lundeby reverberation time, and noise level
    estimation.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension [..., n_samples]
    sampling_rate: integer
        The sampling rate of the room impulse response.
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

    """
    # Define constants:
    # time intervals per 10 dB decay. Lundeby: 3...10
    n_intervals_per_10dB = 5
    # end of regression 5 ... 10 dB
    dB_above_noise = 10
    # Dynamic range 10 ... 20 dB
    use_dyn_range_for_regression = 20

    energy_data, n_channels, data_shape = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=time_shift,
        channel_independent=channel_independent)

    if freq == "broadband":
        # broadband: use 30 ms windows sizes
        freq_dependent_window_time = 0.03
    else:
        freq_dependent_window_time = (800/freq+10) / 1000

    # (1) SMOOTH
    time_window_data, time_vector_window, time_vector = smooth_rir(
        energy_data, sampling_rate, freq_dependent_window_time)

    # (2) ESTIMATE NOISE
    if initial_noise_power == 'auto':
        noise_estimation = estimate_noise_energy(
            energy_data, is_energy=True)
    else:
        noise_estimation = initial_noise_power.copy()

    # (3) REGRESSION
    reverberation_time = np.zeros(n_channels)
    noise_level = np.zeros(n_channels)
    intersection_time = np.zeros(n_channels)
    noise_peak_level = np.zeros(n_channels)

    for idx_channel in range(0, n_channels):
        time_window_data_current_channel = time_window_data[idx_channel]
        start_idx = np.nanargmax(time_window_data_current_channel, axis=-1)
        try:
            stop_idx = (np.argwhere(10*np.log10(
                time_window_data_current_channel[start_idx+1:-1]) >
                    (10*np.log10(noise_estimation[idx_channel]) +
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
            (10*np.log10(noise_estimation[idx_channel]) - slope[0]) / slope[1]
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
            time_vector_current_channel = smooth_rir(
                energy_data[idx_channel], sampling_rate, window_time)
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

        reverberation_time[idx_channel] = -60/slope[1]
        noise_level[idx_channel] = noise_estimation_current_channel
        intersection_time[idx_channel] = crossing_point
        noise_peak_level[idx_channel] = 10 * np.log10(
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
            10*np.log10(noise_estimation[idx_channel]),
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
