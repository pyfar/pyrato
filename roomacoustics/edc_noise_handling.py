#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The edc_noise_handling module provides various methods for noise
    compensation of room impulse responses.
"""

import numpy as np
from matplotlib import pyplot as plt
from roomacoustics import dsp
from roomacoustics import roomacoustics as ra
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

    region_start_idx = np.int(energy_data.shape[-1]*interval[0])
    region_end_idx = np.int(energy_data.shape[-1]*interval[1])
    mask = np.arange(region_start_idx, region_end_idx)
    noise_energy = np.mean(np.take(energy_data, mask, axis=-1), axis=-1)

    return noise_energy


def estimate_noise_energy_from_edc(
        energy_decay_curve,
        intersection_time,
        sampling_rate):
    """Estimate the noise energy from the differential of the energy decay
    curve. The interval used ranges from the intersection time to the end of
    the decay curve. The noise is assumed to be Gaussian.

    Parameters
    ----------
    energy_decay_curve : ndarray, double
        The energy decay curve
    intersection_time : double
        The intersection time between decay curve and noise
    sampling_rate : int
        The sampling rate

    Returns
    -------
    noise_energy : double
        The energy of the additive Gaussian noise

    """
    n_samples = energy_decay_curve.shape[-1]
    if energy_decay_curve.ndim < 2:
        n_channels = 1
        energy_decay_curve = energy_decay_curve[np.newaxis]
    else:
        n_channels = energy_decay_curve.shape[-2]
    noise_energy = np.zeros([n_channels])
    for idx_channel in range(0, n_channels):
        times = np.arange(0, n_samples) * sampling_rate
        mask_second = times > intersection_time[idx_channel]
        mask = times > intersection_time[idx_channel]
        mask_first = np.concatenate((mask[1:], [False]))
        intersection_sample = int(np.ceil(
            intersection_time[idx_channel]*sampling_rate))
        factor = 1/(n_samples - intersection_sample)
        noise_energy[idx_channel] = factor * np.nansum(
            energy_decay_curve[idx_channel, mask_first]
            - energy_decay_curve[idx_channel, mask_second])

    return noise_energy


def preprocess_rir(
        data,
        is_energy=False,
        time_shift=False,
        channel_independent=False):
    """ Preprocess the room impulse response for further processing:
        - square data
        - remove silence at the beginning, if necessary
        - the time shift can be done channel-independent or not.

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
        The preporcessed RIR
    n_channels : integer
        The number of channels of the RIR
    data_shape : list, integer
        The original data shape.

    """
    n_samples = data.shape[-1]
    if data.ndim < 2:
        n_channels = 1
        data = data[np.newaxis]
    else:
        n_channels = data.shape[-2]
    data_shape = list(data.shape)
    data = np.reshape(data, (-1, n_samples))

    if time_shift:
        rir_start_idx = dsp.find_impulse_response_start(data)
        min_shift = np.amin(rir_start_idx)
        result = np.zeros([n_channels, n_samples])

        if channel_independent and not n_channels == 1:
            # Shift each channel independently
            for idx_channel in range(0, n_channels):
                result[idx_channel, 0:-rir_start_idx[idx_channel]] = data[
                    idx_channel, rir_start_idx[idx_channel]:]
        else:
            # Shift each channel by the earliest impulse response start.
            result[:, :-min_shift] = data[:, min_shift:]

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
    n_samples = data.shape[-1]

    n_samples_per_block = int(np.round(smooth_block_length * sampling_rate, 0))
    n_blocks = int(np.floor(n_samples/n_samples_per_block))
    n_samples_actual = int(n_blocks*n_samples_per_block)
    reshaped_array = np.reshape(data[..., :n_samples_actual],
                                (-1, n_blocks, n_samples_per_block))
    time_window_data = np.mean(reshaped_array, axis=-1)

    time_vector_window = \
        ((0.5+np.arange(0, n_blocks)) * n_samples_per_block/sampling_rate)
    time_vector = (0.5+np.arange(0, n_samples))/sampling_rate

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
        noise_level = estimate_noise_energy(data,
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
        noise_level=noise_level,
        is_energy=True,
        time_shift=False,
        channel_independent=False,
        plot=False)[0]
    time_vector = smooth_rir(energy_data, sampling_rate)[2]

    intersection_time_idx = np.rint(intersection_time * sampling_rate)
    max_intersection_time_idx = np.amax(intersection_time_idx)
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
    """ Lundeby et al. [1]_ proposed a correction term to prevent the truncation
    error. The missing signal energy from truncation time to infinity is
    estimated and added to the truncated integral.

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
    ..  [1] Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of
        Measurements in Room Acoustics - ACUSTICA Vol. 81 (1995)
    """

    energy_data, n_channels, data_shape = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=time_shift,
        channel_independent=channel_independent)
    n_samples = energy_data.shape[-1]
    intersection_time, late_reveberation_time, noise_estimation = \
        intersection_time_lundeby(
            energy_data,
            sampling_rate=sampling_rate,
            freq=freq,
            noise_level=noise_level,
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
                      * late_reveberation_time[idx_channel]
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

    """ Implementation of the "subtraction of noise"-method after Chu [2]
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
    ..  [2] W. T. Chu. “Comparison of reverberation measurements using
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

    if normalize:
        # Normalize the EDC...
        if not channel_independent:
            # ...by the first element of each channel.
            energy_decay_curve = (energy_decay_curve.T
                                  / energy_decay_curve[..., 0]).T
        else:
            # ...by the maximum first element of each channel.
            max_start_value = np.amax(energy_decay_curve[..., 0])
            energy_decay_curve /= max_start_value

    mask = energy_decay_curve <= 0
    if np.any(mask):
        first_zero = np.argmax(mask, axis=-1)[0]
        energy_decay_curve[..., int(first_zero):] = np.nan

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
    and the correction for the truncation is applied [1, 2, 3]_

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
    ..  [1] Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of
            Measurements in Room Acoustics - ACUSTICA Vol. 81 (1995)
    ..  [2] W. T. Chu. “Comparison of reverberation measurements using
            Schroeder’s impulse method and decay-curve averaging method”. In:
            Journal of the Acoustical   Society of America 63.5 (1978),
            pp. 1444–1450.
    ..  [3] M. Guski, “Influences of external error sources on measurements of
            room acoustic parameters,” 2015.
    """

    energy_data, n_channels, data_shape = preprocess_rir(
        data,
        is_energy=is_energy,
        time_shift=time_shift,
        channel_independent=channel_independent)
    n_samples = energy_data.shape[-1]

    subtraction = subtract_noise_from_squared_rir(energy_data,
                                                  noise_level=noise_level)

    intersection_time, late_reveberation_time, noise_level = \
        intersection_time_lundeby(
            energy_data,
            sampling_rate=sampling_rate,
            freq=freq,
            noise_level=noise_level,
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
                      * late_reveberation_time[idx_channel]
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

    return energy_decay_curve


def intersection_time_lundeby(
        data,
        sampling_rate,
        freq='broadband',
        noise_level='auto',
        is_energy=False,
        time_shift=False,
        channel_independent=False,
        plot=False):

    """ This function uses the algorithm after Lundeby et al. [1] to calculate
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
    reverberation_time: ndarray, float
        Returns the Lundeby reverberation time for each channel.
    intersection_time: ndarray, float
        Returns the Lundeby intersection time for each channel.
    noise_level: ndarray, float
        Returns the noise level estimation for each channel.

    References
    ----------
    ..  [1] Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of
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

    n_samples = energy_data.shape[-1]

    # (1) SMOOTH
    time_window_data, time_vector_window, time_vector = smooth_rir(
        energy_data, sampling_rate, freq_dependent_window_time)

    # (2) ESTIMATE NOISE
    if noise_level == 'auto':
        noise_estimation = estimate_noise_energy(energy_data, is_energy=True)
    else:
        noise_estimation = noise_level

    # (3) REGRESSION
    reverberation_time = np.zeros(n_channels)
    noise_level = np.zeros(n_channels)
    intersection_time = np.zeros(n_channels)
    noise_peak_level = np.zeros(n_channels)

    for idx_channel in range(0, n_channels):
        time_window_data_current_channel = time_window_data[idx_channel]
        start_idx = np.argmax(time_window_data_current_channel, axis=-1)
        try:
            stop_idx = (np.argwhere(10*np.log10(
                time_window_data_current_channel[start_idx+1:-1]) > (10*np.log10(
                    noise_estimation[idx_channel]) + dB_above_noise))[-1, 0]
                        + start_idx)
        except:
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
        # TO-DO: least-squares solution necessary?
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

        preliminary_crossing_point = \
            (10*np.log10(noise_estimation[idx_channel]) - slope[0]) / slope[1]

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
        idx_max = np.argmax(time_window_data_current_channel)

        # high start value to enter while-loop
        old_crossing_point = 11+preliminary_crossing_point
        loop_counter = 0

        while(True):
            # (7) ESTIMATE BACKGROUND LEVEL
            corresponding_decay = 10  # 5...10 dB
            idx_last_10_percent = np.round(
                time_window_data_current_channel.shape[-1]*0.9)
            idx_10dB_below_crosspoint = np.amax([1, np.round(
                ((preliminary_crossing_point
                  - corresponding_decay / slope[1])
                 * sampling_rate / n_samples_per_block))])

            noise_estimation_current_channel = np.mean(
                time_window_data_current_channel[int(np.amin(
                    [idx_last_10_percent, idx_10dB_below_crosspoint])):])

            # (8) ESTIMATE LATE DECAY SLOPE
            try:
                start_idx_loop = np.argwhere(10*np.log10(
                    time_window_data_current_channel[idx_max:]) < (
                        10*np.log10(noise_estimation_current_channel)
                        + dB_above_noise
                        + (use_dyn_range_for_regression)[0, 0]
                        + idx_max))
            except:
                start_idx_loop = 0

            try:
                stop_idx_loop = (np.argwhere(10*np.log10(
                    time_window_data_current_channel[start_idx_loop+1:]) < (
                        10*np.log10(noise_estimation_current_channel)
                        + dB_above_noise))[0, 0]
                                 + start_idx_loop)
            except:
                raise Exception(
                    'Regression failed: Low SNR. Estimation terminated.')

            # regression_matrix*slope = edc
            # regression_matrix = 0

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
            old_crossing_point = preliminary_crossing_point
            crossing_point = ((10*np.log10(noise_estimation_current_channel)
                               - slope[0]) / slope[1])

            loop_counter = loop_counter + 1

            if (np.abs(old_crossing_point-preliminary_crossing_point) < 0.01):
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
            np.amax(time_window_data_current_channel[int(np.amin(
                [idx_last_10_percent, idx_10dB_below_crosspoint])):]))

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        max_data_db = np.max(10*np.log10(energy_data))
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
