import numpy as np
from matplotlib import pyplot as plt
import warnings
from roomacoustics import dsp
from roomacoustics import roomacoustics as ra

def estimate_noise_energy(
    data, interval=[0.9, 1.0], is_energy=False):
    """ This function estimates the noise energy level of a given room impulse
    response. The noise is assumed to be Gaussian.

    Parameters
    ----------
    data: np.array
        The room impulse response with dimension [..., n_samples]
    interval : tuple, float
        Defines the interval of the RIR to be evaluated for the estimation.
        The interval is relative to the length of the RIR [0 = 0%, 1=100%)]
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    noise_energy: float
        The energy of the background noise
    """

    if not is_energy:
        energy_data = np.abs(data)**2
    else:
        energy_data = data

    if np.any(energy_data) < 0:
        raise ValueError("Energy is negative, check your input signal.")

    region_start_idx = np.int(energy_data.shape[-1]*interval[0])
    region_end_idx = np.int(energy_data.shape[-1]*interval[1])
    mask = np.arange(region_start_idx, region_end_idx)
    noise_energy = np.mean(np.take(energy_data, mask, axis=-1), axis=-1)

    return noise_energy


def estimate_noise_energy_from_edc(
    energy_decay_curve, intersection_time, sampling_rate):
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
    times = np.arange(0, n_samples) * sampling_rate
    mask_second = times > intersection_time
    mask = times > intersection_time
    mask_first = np.concatenate((mask[1:], [False]))
    intersection_sample = int(np.ceil(intersection_time*sampling_rate))
    factor = 1/(n_samples - intersection_sample)

    noise_energy = factor * np.sum(
        energy_decay_curve[mask_first] - energy_decay_curve[mask_second])

    return noise_energy


def remove_silence_at_beginning_and_square_data(
    data, is_energy=False, channel_independent=False):

    rir_start_idx = dsp.find_impulse_response_start(data)
    n_samples = data.shape[-1]
    data_shape = list(data.shape)
    data = np.reshape(data, (-1,n_samples))
    min_shift = np.amin(rir_start_idx)
    if data.ndim < 2:
        n_channels = 1
    else:
        n_channels = data.shape[-2]
    result = np.zeros([n_channels, n_samples-min_shift])
    if channel_independent:
        for idx_channel in range(0, n_channels):
            result[idx_channel,0:-rir_start_idx[idx_channel]] = data[idx_channel, rir_start_idx[idx_channel]:]
    else:
        result[:,:] = data[:, min_shift:]
    data_shape[-1] = n_samples-min_shift
    result = np.reshape(result, data_shape)

    if not is_energy:
        energy_data = np.abs(result)**2
    else:
        energy_data = result
    return energy_data


def smooth_rir(data, sampling_rate, smooth_block_length=0.075):
    n_samples = data.shape[-1]

    n_samples_per_block = int(np.round(smooth_block_length * sampling_rate, 0))
    n_blocks = int(np.floor(n_samples/n_samples_per_block))
    n_samples_actual = int(n_blocks*n_samples_per_block)
    reshaped_array = np.reshape(data[..., :n_samples_actual],
                                (-1, n_blocks, n_samples_per_block))
    time_window_data = np.mean(reshaped_array, axis=-1)

    time_vector_window = ((0.5+np.arange(0, n_blocks)) *
                            n_samples_per_block/sampling_rate)
    time_vector = (0.5+np.arange(0, n_samples))/sampling_rate

    return time_window_data, time_vector_window, time_vector


def substract_noise_from_rir(data, is_energy=False):
    noise_level = estimate_noise_energy(data, is_energy=is_energy)
    return (data.T - noise_level).T


def energy_decay_curve_truncation(
    data, sampling_rate, freq='broadband', is_energy=False, normalize=True):
    """ This function trucates a given RIR by the intersection time after Lundeby and
    calculates the EDC.

    Parameters
    ----------
    data: np.array
        Contains the room impulse response or EDC.
    sampling_rate: integer
        Defines the sampling rate of the room impulse response.
    freq: integer OR string
        Defines the frequency band. If set to 'broadband',
        the time window will not be set in dependence of frequency
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    EDC: np.array
        Returns the noise handeled edc.
    """
    energy_data = remove_silence_at_beginning_and_square_data(data, is_energy)
    intersection_time = intersection_time_lundeby(
        data, sampling_rate, freq, False, is_energy)[0]
    if data.ndim < 2:
        n_channels = 1
    else:
        n_channels = data.shape[-2]

    n_samples = energy_data.shape[-1]
    data_shape = list(energy_data.shape)
    energy_data = np.reshape(energy_data, (-1, n_samples))

    intersection_time_idx = np.rint(intersection_time * sampling_rate)
    max_intersection_time_idx = np.amax(intersection_time_idx)
    energy_decay_curve = np.zeros([n_channels, n_samples])
    for idx_channel in range(0, n_channels):
        energy_decay_curve[idx_channel,:int(intersection_time_idx[idx_channel])] = (
            ra.schroeder_integration(
                energy_data[idx_channel,:int(intersection_time_idx[idx_channel])],
                is_energy=True))
    if normalize:
        energy_decay_curve = (energy_decay_curve.T / energy_decay_curve[..., 0]).T

    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)
    return energy_decay_curve


def energy_decay_curve_lundeby(
    data, sampling_rate, freq='broadband', plot=False, is_energy=False, normalize=True):
    """ Lundeby et al. [1] proposed a correction term to prevent the truncation error.
    The missing signal energy from truncation time to infinity is estimated and
    added to the truncated integral.

    Parameters
    ----------
    data: np.array
        Contains the room impulse response.
    sampling_rate: integer
        Defines the sampling rate of the room impulse response.
    freq: integer OR string
        Defines the frequency band. If set to 'broadband',
        the time window will not be set in dependence of frequency
    plot: Boolean
        Specifies, whether the results should be visualized or not.
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    edc_lundeby: np.array
        Returns the noise handeled edc.


    References
    ----------
        [1] Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of Measurements
        in Room Acoustics - ACUSTICA Vol. 81 (1995)

    """
    # Find start of RIR and perform a cyclic shift:
    energy_data = remove_silence_at_beginning_and_square_data(data, is_energy)
    if data.ndim < 2:
        n_channels = 1
    else:
        n_channels = data.shape[-2]
    n_samples = energy_data.shape[-1]
    data_shape = list(energy_data.shape)
    energy_data = np.reshape(energy_data, (-1,n_samples))

    intersection_time, late_reveberation_time, noise_level = intersection_time_lundeby(
        data, sampling_rate, freq, False, is_energy)

    time_vector = smooth_rir(energy_data, sampling_rate)[2]

    energy_decay_curve = np.zeros([n_channels, n_samples])

    for idx_channel in range(0, n_channels):
        intersection_time_idx = np.argmin(np.abs(time_vector - intersection_time[idx_channel]))

        p_square_at_intersection = estimate_noise_energy(
            energy_data[idx_channel], is_energy=True)

        # calculate correction term according to DIN EN ISO 3382
        correction = (p_square_at_intersection * late_reveberation_time[idx_channel] /
                      (6*np.log(10)) * sampling_rate)

        energy_decay_curve[idx_channel,:intersection_time_idx] = ra.schroeder_integration(
            energy_data[idx_channel,:intersection_time_idx], is_energy=True)
        energy_decay_curve[idx_channel] += correction
    if normalize:
        energy_decay_curve = (energy_decay_curve.T / energy_decay_curve[..., 0]).T
    energy_decay_curve[..., intersection_time_idx:] = np.nan
    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(121)
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(122)
        plt.plot(time_vector[0:energy_decay_curve.shape[-1]], 10*np.log10(energy_decay_curve.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Tr. EDC with correction [dB]')
        plt.tight_layout()

    return energy_decay_curve


def energy_decay_curve_chu(
    data, sampling_rate, freq='broadband', plot=False, is_energy=False,
    time_shift=True, normalize=True):
    """ Implementation of the "subtraction of noise"-method after Chu [1]
    The noise level is estimated and subtracted from the impulse response
    before backward integration.

    Parameters
    ----------
    data: np.array
        Contains the room impulse response.
    sampling_rate: integer
        Defines the sampling rate of the room impulse response.
    freq: integer OR string
        Defines the frequency band. If set to 'broadband',
        the time window will not be set in dependence of frequency
    plot: Boolean
        Specifies, whether the results should be visualized or not.
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    edc_lundeby: np.array
        Returns the noise handeled edc.


    References
    ----------
        [1] W. T. Chu. “Comparison of reverberation measurements using Schroeder’s
            impulse method and decay-curve averaging method”. In: Journal of the Acoustical
            Society of America 63.5 (1978), pp. 1444–1450.

    """

    # Find start of RIR and perform a cyclic shift:
    if time_shift:
        energy_data = remove_silence_at_beginning_and_square_data(data, is_energy)
    else:
        if is_energy:
            energy_data = data.copy()
        else:
            energy_data = np.abs(data)**2

    subtracted = substract_noise_from_rir(energy_data, is_energy=True)

    energy_decay_curve = ra.schroeder_integration(
        subtracted, is_energy=True)
    if normalize:
        energy_decay_curve = (energy_decay_curve.T / energy_decay_curve[..., 0]).T
    mask = energy_decay_curve <= 0
    if np.any(mask):
        first_zero = np.argmax(energy_decay_curve[mask], axis=-1) # Maybe too senstitive?
        energy_decay_curve[..., first_zero:] = np.nan

    if plot:
        time_vector = (0.5+np.arange(0, energy_data.shape[-1]))/sampling_rate
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data.T/energy_data.max(axis=-1)).T)
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.grid(True)
        plt.subplot(132)
        plt.plot(time_vector, 10*np.log10(subtracted.T/subtracted.max(axis=-1)).T)
        plt.xlabel('Time [s]')
        plt.ylabel('Noise substracted IR [dB]')
        plt.grid(True)
        plt.subplot(133)
        plt.plot(time_vector[0:energy_decay_curve.shape[-1]], 10*np.log10(energy_decay_curve.T))
        plt.xlabel('Time [s]')
        plt.ylabel('EDC, noise subtracted [dB]')
        plt.grid(True)
        plt.tight_layout()

    return energy_decay_curve


def energy_decay_curve_chu_lundeby(data, sampling_rate, freq='broadband', plot=False, is_energy=False, normalize=True):
    """ Description: This function combines Chu's and Lundeby's methods:
    The estimated noise level is subtracted before backward integration,
    the impulse response is truncated at the intersection time,
    and the correction for the truncation is applied. [1, 2, 3]

    Parameters
    ----------
    data: np.array
        Contains the room impulse response.
    sampling_rate: integer
        Defines the sampling rate of the room impulse response.
    freq: integer OR string
        Defines the frequency band. If set to 'broadband',
        the time window will not be set in dependence of frequency
    plot: Boolean
        Specifies, whether the results should be visualized or not.
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    edc_lundeby: np.array
        Returns the noise handeled edc.


    References
    ----------
        [1] Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of Measurements
        in Room Acoustics - ACUSTICA Vol. 81 (1995)
        [2] M. Guski, “Influences of external error sources on measurements of room acoustic parameters,” 2015.
        [3] W. T. Chu. “Comparison of reverberation measurements using Schroeder’s
            impulse method and decay-curve averaging method”. In: Journal of the Acoustical
            Society of America 63.5 (1978), pp. 1444–1450.

    """
    energy_data = remove_silence_at_beginning_and_square_data(data, is_energy)
    if data.ndim < 2:
        n_channels = 1
    else:
        n_channels = data.shape[-2]
    n_samples = energy_data.shape[-1]
    data_shape = list(energy_data.shape)
    energy_data = np.reshape(energy_data, (-1,n_samples))

    substraction = substract_noise_from_rir(energy_data, is_energy=True)

    intersection_time, late_reveberation_time, noise_level = (
        intersection_time_lundeby(data=data,
                                  sampling_rate=sampling_rate,
                                  freq=freq,
                                  plot=False,
                                  is_energy=is_energy))

    time_vector = smooth_rir(energy_data, sampling_rate)[2]
    energy_decay_curve = np.zeros([n_channels, n_samples])

    for idx_channel in range(0, n_channels):
        intersection_time_idx = np.argmin(np.abs(time_vector - intersection_time[idx_channel]))

        p_square_at_intersection = estimate_noise_energy(energy_data[idx_channel], is_energy=True)

        # calculate correction term according to DIN EN ISO 3382
        correction = p_square_at_intersection * late_reveberation_time[idx_channel] / (6*np.log(10)) * sampling_rate

        energy_decay_curve[idx_channel,:intersection_time_idx] = ra.schroeder_integration(substraction[idx_channel,:intersection_time_idx],
                                                      is_energy=True)
        energy_decay_curve[idx_channel] += correction

    if normalize:
        energy_decay_curve = (energy_decay_curve.T / energy_decay_curve[..., 0]).T
    mask = energy_decay_curve <= 0
    if np.any(mask):
        first_zero = np.argmax(energy_decay_curve[mask], axis=-1)
        energy_decay_curve[..., first_zero:] = np.nan
    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)
    energy_decay_curve[..., intersection_time_idx:] = np.nan
    energy_decay_curve = np.reshape(energy_decay_curve, data_shape)

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(132)
        plt.plot(time_vector, 10*np.log10(substraction.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Noise substracted IR [dB]')
        plt.subplot(133)
        plt.plot(time_vector[0:energy_decay_curve.shape[-1]], 10*np.log10(energy_decay_curve.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Tr. EDC with corr. & subst. [dB]')
        plt.tight_layout()

    return energy_decay_curve


def intersection_time_lundeby(data, sampling_rate, freq='broadband', plot=False, is_energy=False, time_shift=True):
    """ This function uses the algorithm after Lundeby et al. [1] to calculate the (late)
    reverberation time, intersection time and noise level estimation.

    TO-DO: write test function, ...

    Parameters
    ----------
    rir: np.array
        Contains the room impulse response.
    sampling_rate: integer
        Defines the sampling rate of the room impulse response.
    freq: integer OR string
        Defines the frequency band. If set to 'broadband',
        the time window will not be set in dependence of frequency
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    reverberation_time: float
        Returns the Lundeby reverberation time.
    intersection_time: float
        Returns the Lundeby intersection time.
    noise_level: float
        Returns the noise level estimation.

    References
    ----------
        [1] Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of Measurements
        in Room Acoustics - ACUSTICA Vol. 81 (1995)

    """

    # Define constants:
    n_intervals_per_10dB = 5   # time intervals per 10 dB decay. Lundeby: 3 ... 10
    dB_above_noise = 10   # end of regression 5 ... 10 dB
    use_dyn_range_for_regression = 20  # 10 ... 20 dB

    if time_shift:
        energy_data = remove_silence_at_beginning_and_square_data(
                data, is_energy=is_energy)
    else:
        if is_energy:
            energy_data = data.copy()
        else:
            energy_data = np.abs(data)**2
    is_energy = True


    if freq == 'broadband':
        freq_dependent_window_time = 0.03  # broadband: use 30 ms windows sizes
    else:
        freq_dependent_window_time = (800/freq+10) / 1000

    n_samples = energy_data.shape[-1]
    if data.ndim < 2:
        n_channels = 1
        energy_data = energy_data[np.newaxis]
    else:
        n_channels = data.shape[-2]

    # (1) SMOOTH
    time_window_data, time_vector_window, time_vector = smooth_rir(
        energy_data, sampling_rate, freq_dependent_window_time)

    # (2) ESTIMATE NOISE
    noise_estimation = estimate_noise_energy(energy_data, is_energy=True)

    # (3) REGRESSION
    reverberation_time = np.zeros(n_channels)
    noise_level = np.zeros(n_channels)
    intersection_time= np.zeros(n_channels)
    noise_peak_level = np.zeros(n_channels)

    for idx_channel in range(0, n_channels):
        time_window_data_current_channel = time_window_data[idx_channel]
        start_idx = np.argmax(time_window_data_current_channel, axis=-1)

        try:
            stop_idx = np.argwhere(10*np.log10(time_window_data_current_channel[start_idx+1:-1])
                                   > 10*np.log10(noise_estimation[idx_channel]) + dB_above_noise)[-1, 0] + start_idx
        except:
            raise Exception('Regression did not work due to low SNR. Estimation was terminated.')

        dyn_range = np.diff(10*np.log10(np.take(time_window_data_current_channel, [start_idx, stop_idx])))

        if (stop_idx == start_idx) or dyn_range > -5:
            raise Exception('Regression did not work due to low SNR. Estimation was terminated.')

        # regression_matrix*slope = edc
        regression_matrix = np.vstack(
            (np.ones([stop_idx-start_idx]), time_vector_window[start_idx:stop_idx]))
        # TO-DO: least-squares solution necessary?
        slope = np.linalg.lstsq(regression_matrix.T,
                                (10*np.log10(time_window_data_current_channel[start_idx:stop_idx])), rcond=None)[0]

        if slope[1] == 0 or np.any(np.isnan(slope)):
            raise Exception(
                'Regression did not work due, T would be Inf, setting to 0. Estimation was terminated.')

        regression_time = np.array([time_vector_window[start_idx],
                               time_vector_window[stop_idx]])
        regression_values = np.array([10*np.log10(time_window_data[0, start_idx]),
                            10*np.log10(time_window_data[0, start_idx])+slope[1]*time_vector_window[stop_idx]])

        # (4) PRELIMINARY CROSSING POINT

        preliminary_crossing_point = (10*np.log10(noise_estimation[idx_channel]) - slope[0]) / slope[1] # Why  - slope[0]?

        # (5) NEW LOCAL TIME INTERVAL LENGTH
        n_blocks_in_decay = np.diff(
            10*np.log10(np.take(time_window_data_current_channel, [start_idx, stop_idx]))) / -10 * n_intervals_per_10dB
        n_samples_per_block = np.round(
            np.diff(np.take(time_vector_window, [start_idx, stop_idx])) / n_blocks_in_decay * sampling_rate)
        window_time = n_samples_per_block/sampling_rate

        # (6) AVERAGE

        time_window_data_current_channel, time_vector_window_current_channel, time_vector_current_channel = smooth_rir(energy_data[idx_channel], sampling_rate, window_time)
        time_window_data_current_channel = np.squeeze(time_window_data_current_channel)
        idx_max = np.argmax(time_window_data_current_channel)

        old_crossing_point = 11+preliminary_crossing_point  # high start value to enter while-loop
        loop_counter = 0

        while(np.abs(old_crossing_point - preliminary_crossing_point) > 0.01):
            # (7) ESTIMATE BACKGROUND LEVEL
            corresponding_decay = 10  # 5...10 dB
            idx_last_10_percent = np.round(time_window_data_current_channel.shape[-1]*0.9)
            idx_10dB_below_crosspoint = np.amax(
                [1, np.round((preliminary_crossing_point - corresponding_decay / slope[1]) * sampling_rate / n_samples_per_block)])

            noise_estimation_current_channel = np.mean(time_window_data_current_channel[int(
                np.amin([idx_last_10_percent, idx_10dB_below_crosspoint])):])

            # (8) ESTIMATE LATE DECAY SLOPE
            try:
                start_idx_loop = np.argwhere(10*np.log10(time_window_data_current_channel[idx_max:]) < 10*np.log10(
                    noise_estimation_current_channel) + dB_above_noise + use_dyn_range_for_regression)[0, 0] + idx_max  # -1?
            except:
                start_idx_loop = 0

            try:
                stop_idx_loop = np.argwhere(10*np.log10(time_window_data_current_channel[start_idx_loop+1:]) < (
                    10*np.log10(noise_estimation_current_channel) + dB_above_noise))[0, 0] + start_idx_loop
            except:
                raise Exception('Regression did not work due to low SNR. Estimation was terminated.')

            # regression_matrix*slope = edc
            #regression_matrix = 0
            regression_matrix = np.vstack(
                (np.ones([stop_idx_loop-start_idx_loop]), time_vector_window_current_channel[start_idx_loop:stop_idx_loop]))
            slope = np.linalg.lstsq(regression_matrix.T,
                                    (10*np.log10(time_window_data_current_channel[start_idx_loop:stop_idx_loop])), rcond=None)[0]

            if slope[1] >= 0:
                raise Exception(
                    'Regression did not work due, T would be Inf, setting to 0. Estimation was terminated.')
                #slope[1] = np.inf

            # (9) FIND CROSSPOINT
            old_crossing_point = preliminary_crossing_point
            crossing_point = (10*np.log10(noise_estimation_current_channel) - slope[0]) / slope[1]

            loop_counter = loop_counter + 1
            if loop_counter > 30:
                # TO-DO: Paper says 5 iterations are sufficient in all cases!
                warnings.warn("Warning: Lundeby algorithm was canceled after 30 iterations.")
                break

        reverberation_time[idx_channel] = -60/slope[1]
        noise_level[idx_channel] = 10*np.log10(noise_estimation_current_channel)  # TO-DO: dB or absolute value?
        intersection_time[idx_channel] = crossing_point
        noise_peak_level[idx_channel] = 10 * \
            np.log10(
                np.amax(time_window_data_current_channel[int(np.amin([idx_last_10_percent, idx_10dB_below_crosspoint])):]))

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared RIR [dB]')
        plt.subplot(132)
        plt.plot(time_vector_window, 10*np.log10(time_window_data.T))
        plt.xlabel('Time [s]')
        plt.ylabel('Smoothened RIR [dB]')
        plt.plot(time_vector_window[-int(np.round(time_window_data.shape[-1]/10))],
                 10*np.log10(noise_estimation[0]), marker='o', label='noise estimation')
        plt.plot(regression_time, regression_values, marker='o', label='regression')
        plt.plot(preliminary_crossing_point, 10*np.log10(noise_estimation[idx_channel]),
                 marker='o', label='preliminary crosspoint')
        plt.legend()
        plt.subplot(133)
        # TO-DO: plot all channels?
        plt.plot(time_vector_window_current_channel, 10*np.log10(time_window_data_current_channel))
        plt.xlabel('Time [s]')
        plt.ylabel('Final EDC Estimation [dB]')
        plt.plot(time_vector_window_current_channel[int(idx_10dB_below_crosspoint)],
                 10*np.log10(noise_estimation[0]), marker='o', label='noise')
        plt.plot(crossing_point, 10*np.log10(noise_estimation[0]),
                 marker='o', label='final crosspoint')
        plt.axvline(reverberation_time[0], label=(
            '$T_{60} = ' + str(np.round(reverberation_time[0], 3)) + '$'))
        #plt.axhline(noise_level[0], label=('final noise estimation'))
        plt.axvline(intersection_time[0], label=('lundeby intersection time'), color='r')
        plt.tight_layout()
        plt.legend()

    return intersection_time, reverberation_time, noise_level
