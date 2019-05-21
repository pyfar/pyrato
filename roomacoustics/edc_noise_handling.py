import numpy as np
from matplotlib import pyplot as plt
import warnings
from roomacoustics import dsp
from roomacoustics import roomacoustics as ra

np.seterr(divide='ignore')

# TO-DO: Write test functions, check multidimensional compatibility.


def estimate_noise_energy(data,
                          region_start=0.9,
                          region_end=1,
                          is_energy=True):

    """ This function estimates the noise energy level of a given room impulse response.

    Parameters
    ----------
    data: np.array
        Contains the room impulse response.
    region_start/end: float
        Defines the region of the RIR to be evaluated for estimation (0 = 0%, 1=100%).
    is_energy: Boolean
        Defines if the data is already squared.

    Returns
    -------
    noise_energy: float
        Returns the noise level estimation.
    """

    if not is_energy:
        data = np.abs(data)**2

    region_start_idx = int(np.round(data.shape[-1]*region_start))
    region_end_idx = int(np.round(data.shape[-1]*region_end))

    noise_energy = np.mean(data[region_start_idx:region_end_idx]) # + np.finfo(np.double).tiny      necessary?

    return noise_energy


def remove_silence_at_beginning_and_square_data(data, is_energy=False):
    rir_start_idx = dsp.find_impulse_response_start(data)
    # data = dsp.time_shift(data, -rir_start_idx)
    data = data[rir_start_idx+1:]
    if not is_energy:
        energy_data = np.abs(data)**2
    else:
        energy_data = data
    return energy_data

def smooth_edc(data, sampling_rate, smooth_block_length=0.075):
    n_samples = data.shape[0]
    n_samples_per_block = int(np.round(smooth_block_length * sampling_rate, 0))
    n_blocks = int(np.floor(n_samples/n_samples_per_block))
    n_samples_actual = int(n_blocks*n_samples_per_block)

    reshaped_array = np.reshape(data[0:n_samples_actual], (n_blocks, n_samples_per_block))
    time_window_data = np.squeeze(np.sum(reshaped_array,1)/n_samples_per_block)

    time_vector_window = ((0.5+np.arange(0, n_blocks)) *
                            n_samples_per_block/sampling_rate)

    time_vector = (0.5+np.arange(0, n_samples))/sampling_rate

    return time_window_data, time_vector_window, time_vector


def substract_noise_from_edc(data):
    noise_level = estimate_noise_energy(data)
    return (data - noise_level)


def energy_decay_curve_truncation(data, sampling_rate, freq='broadband', is_energy=False):
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

    intersection_time = intersection_time_lundeby(data,
                                                  sampling_rate,
                                                  freq,
                                                  False,
                                                  is_energy)[0]

    intersection_time_idx = int(np.round(intersection_time * sampling_rate))

    energy_decay_curve = ra.schroeder_integration(energy_data[:intersection_time_idx],
                                                  is_energy=True)
    energy_decay_curve = energy_decay_curve / energy_decay_curve[0]
    return energy_decay_curve


def energy_decay_curve_lundeby(data, sampling_rate, freq='broadband', plot=False, is_energy=False):
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

    intersection_time, late_reveberation_time, noise_level = intersection_time_lundeby(
        data, sampling_rate, freq, False, is_energy)
    time_window_data, time_vector_window, time_vector = smooth_edc(energy_data, sampling_rate)

    intersection_time_window_idx = np.argmin(np.abs(time_vector_window - intersection_time))
    intersection_time_idx = np.argmin(np.abs(time_vector - intersection_time))

    p_square_at_intersection = time_window_data[intersection_time_window_idx]

    # calculate correction term according to DIN EN ISO 3382
    correction = p_square_at_intersection * late_reveberation_time / (6*np.log(10)) * sampling_rate

    energy_decay_curve = ra.schroeder_integration(energy_data[:intersection_time_idx],
                                                  is_energy=True)
    edc_with_correction = energy_decay_curve + correction
    energy_decay_curve /= energy_decay_curve[0]
    edc_with_correction /= edc_with_correction[0]

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(132)
        plt.plot(time_vector[0:energy_decay_curve.shape[0]], 10*np.log10(energy_decay_curve))
        plt.xlabel('Time [s]')
        plt.ylabel('Truncated EDC [dB]')
        plt.subplot(133)
        plt.plot(time_vector[0:edc_with_correction.shape[0]], 10*np.log10(edc_with_correction))
        plt.xlabel('Time [s]')
        plt.ylabel('Tr. EDC with correction [dB]')
        plt.tight_layout()

    return edc_with_correction


def energy_decay_curve_chu(data, sampling_rate, freq='broadband', plot=False, is_energy=False):
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
    energy_data = remove_silence_at_beginning_and_square_data(data, is_energy)
    substraction = substract_noise_from_edc(energy_data)

    energy_decay_curve = ra.schroeder_integration(substraction, is_energy=True)
    energy_decay_curve /= energy_decay_curve[0]
    energy_decay_curve[energy_decay_curve <= 0] = np.nan

    if plot:
        time_vector = (0.5+np.arange(0, energy_data.shape[0]))/sampling_rate
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(132)
        plt.plot(time_vector, 10*np.log10(substraction))
        plt.xlabel('Time [s]')
        plt.ylabel('Noise substracted IR [dB]')
        plt.subplot(133)
        plt.plot(time_vector[0:energy_decay_curve.shape[0]], 10*np.log10(energy_decay_curve))
        plt.xlabel('Time [s]')
        plt.ylabel('EDC, noise substracted [dB]')
        plt.tight_layout()

    return energy_decay_curve


def energy_decay_curve_chu_lundeby(data, sampling_rate, freq='broadband', plot=False, is_energy=False):
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
    # Find start of RIR and perform a cyclic shift:
    energy_data = remove_silence_at_beginning_and_square_data(data, is_energy)

    substraction = substract_noise_from_edc(energy_data)

    intersection_time, late_reveberation_time, noise_level = (
        intersection_time_lundeby(data,
                                  sampling_rate,
                                  freq,
                                  False,
                                  is_energy))

    time_window_data, time_vector_window, time_vector = (
        smooth_edc(energy_data, sampling_rate))

    intersection_time_window_idx = np.argmin(np.abs(time_vector_window - intersection_time))
    intersection_time_idx = np.argmin(np.abs(time_vector - intersection_time))

    p_square_at_intersection = time_window_data[intersection_time_window_idx]

    # calculate correction term according to DIN EN ISO 3382
    correction = p_square_at_intersection * late_reveberation_time / (6*np.log(10)) * sampling_rate

    energy_decay_curve = np.cumsum(substraction[intersection_time_idx::-1])[::-1]
    edc_with_correction = energy_decay_curve + correction
    energy_decay_curve /= energy_decay_curve[0]
    edc_with_correction /= edc_with_correction[0]

    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(141)
        plt.plot(time_vector, 10*np.log10(energy_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Squared IR [dB]')
        plt.subplot(142)
        plt.plot(time_vector, 10*np.log10(substraction))
        plt.xlabel('Time [s]')
        plt.ylabel('Noise substracted IR [dB]')
        plt.subplot(143)
        plt.plot(time_vector[0:energy_decay_curve.shape[0]], 10*np.log10(energy_decay_curve))
        plt.xlabel('Time [s]')
        plt.ylabel('Truncated EDC [dB]')
        plt.subplot(144)
        plt.plot(time_vector[0:edc_with_correction.shape[0]], 10*np.log10(edc_with_correction))
        plt.xlabel('Time [s]')
        plt.ylabel('Tr. EDC with corr. & subst. [dB]')
        plt.tight_layout()

    return edc_with_correction


def intersection_time_lundeby(data, sampling_rate, freq='broadband', plot=False, is_energy=False):
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

    energy_data = remove_silence_at_beginning_and_square_data(data, is_energy)

    if freq == 'broadband':
        freq_dependent_window_time = 0.03  # broadband: use 30 ms windows sizes
    else:
        freq_dependent_window_time = (800/freq+10) / 1000

    # if shortRevTimeMode: # TO-DO?
    #    freq_dependent_window_time = freq_dependent_window_time / 5;

    n_samples = energy_data.shape[0]

    # (1) SMOOTH
    time_window_data, time_vector_window, time_vector = smooth_edc(energy_data,
                                                                   sampling_rate,
                                                                   freq_dependent_window_time)
    #time_window_data = time_window_data/time_window_data[0]
    # (2) ESTIMATE NOISE
    noise_estimation = estimate_noise_energy(time_window_data)

    # (...) PLOT FOR VISUALIZSATION
    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector, 10*np.log10(energy_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Original EDC [dB]')
        plt.subplot(132)
        plt.plot(time_vector_window, 10*np.log10(time_window_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Smoothened EDC [dB]')
        plt.plot(time_vector_window[-int(np.round(time_window_data.shape[0]/10))],
                 10*np.log10(noise_estimation), marker='o', label='noise estimation')

    # (3) REGRESSION

    start_idx = np.argmax(time_window_data)
    try:
        stop_idx = np.argwhere(10*np.log10(time_window_data[start_idx+1:-1])
                               > 10*np.log10(noise_estimation) + dB_above_noise)[-1, 0] + start_idx
    except:
        raise Exception('Regression did not work due to low SNR. Estimation was terminated.')

    dyn_range = np.diff(10*np.log10(np.take(time_window_data, [start_idx, stop_idx])))

    if (stop_idx == start_idx) or dyn_range > -5:
        raise Exception('Regression did not work due to low SNR. Estimation was terminated.')

    # regression_matrix*slope = edc
    regression_matrix = np.vstack(
        (np.ones([stop_idx-start_idx]), time_vector_window[start_idx:stop_idx]))
    # TO-DO: least-squares solution necessary? # kann weg
    slope = np.linalg.lstsq(regression_matrix.T,
                            (10*np.log10(time_window_data[start_idx:stop_idx])), rcond=None)[0]

    if slope[1] == 0 or np.any(np.isnan(slope)):
        raise Exception(
            'Regression did not work due, T would be Inf, setting to 0. Estimation was terminated.')

    # (4) PRELIMINARY CROSSING POINT
    crossing_point = (10*np.log10(noise_estimation) - slope[0]) / slope[1]

    if plot:
        plt.plot([time_vector_window[start_idx], time_vector_window[stop_idx]], [10*np.log10(time_window_data[start_idx]),
                                                                                 10*np.log10(time_window_data[start_idx])+slope[1]*time_vector_window[stop_idx]], marker='o', label='regression')
        plt.plot(crossing_point, 10*np.log10(noise_estimation),
                 marker='o', label='preliminary crosspoint')
        plt.legend()

    # if crossing_point > ((input.trackLength + timeShifted(iChannel))  * 2 ):
        # continue

    # (5) NEW LOCAL TIME INTERVAL LENGTH
    n_blocks_in_decay = np.diff(
        10*np.log10(np.take(time_window_data, [start_idx, stop_idx]))) / -10 * n_intervals_per_10dB
    n_samples_per_block = np.round(
        np.diff(np.take(time_vector_window, [start_idx, stop_idx])) / n_blocks_in_decay * sampling_rate)

    # (6) AVERAGE
    time_window_data = np.squeeze(np.sum(np.reshape(
        energy_data[0:int(np.floor(n_samples/n_samples_per_block)*n_samples_per_block)],
        (int(np.floor(n_samples/n_samples_per_block)), int(n_samples_per_block))), 1)/n_samples_per_block)  # np.squeeze notwendig?
    #time_window_data = time_window_data/time_window_data[0]

    time_vector_window = np.arange(0, time_window_data.shape[0])*n_samples_per_block/sampling_rate
    idx_max = np.argmax(time_window_data)

    old_crossing_point = 11+crossing_point  # high start value to enter while-loop
    loop_counter = 0
    if plot:
        plt.subplot(133)
        plt.plot(time_vector_window, 10*np.log10(time_window_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Final EDC Estimation [dB]')

    while(np.abs(old_crossing_point-crossing_point) > 0.01):
        # (7) ESTIMATE BACKGROUND LEVEL
        corresponding_decay = 10  # 5...10 dB
        idx_last_10_percent = np.round(time_window_data.shape[0]*0.9)
        idx_10dB_below_crosspoint = np.amax(
            [1, np.round((crossing_point - corresponding_decay / slope[1]) * sampling_rate / n_samples_per_block)])
        noise_estimation = np.mean(time_window_data[int(
            np.amin([idx_last_10_percent, idx_10dB_below_crosspoint])):]) + np.finfo(np.double).tiny

        if plot:
            plt.plot(time_vector_window, 10*np.log10(time_window_data))
            plt.plot(time_vector_window[int(idx_10dB_below_crosspoint)],
                     10*np.log10(noise_estimation), marker='o', label='noise')

        # (8) ESTIMATE LATE DECAY SLOPE
        start_idx = np.argwhere(10*np.log10(time_window_data[idx_max:]) < 10*np.log10(
            noise_estimation) + dB_above_noise + use_dyn_range_for_regression)[0, 0] + idx_max  # -1?

        if stop_idx.size == 0:
            start_idx = 1
        try:
            stop_idx = np.argwhere(10*np.log10(time_window_data[start_idx+1:]) < (
                10*np.log10(noise_estimation) + dB_above_noise))[0, 0] + start_idx
        except:
            raise Exception('Regression did not work due to low SNR. Estimation was terminated.')

        # regression_matrix*slope = edc
        regression_matrix = np.vstack(
            (np.ones([stop_idx-start_idx]), time_vector_window[start_idx:stop_idx]))
        slope = np.linalg.lstsq(regression_matrix.T,
                                (10*np.log10(time_window_data[start_idx:stop_idx])), rcond=None)[0]

        if slope[1] >= 0:
            raise Exception(
                'Regression did not work due, T would be Inf, setting to 0. Estimation was terminated.')
            #slope[1] = np.inf

        # (9) FIND CROSSPOINT
        old_crossing_point = crossing_point
        crossing_point = (10*np.log10(noise_estimation) - slope[0]) / slope[1]

        loop_counter = loop_counter + 1
        if loop_counter > 30:
            # TO-DO: Paper says 5 iterations are sufficient in all cases!
            warnings.warn("Warning: Lundeby algorithm was canceled after 30 iterations.")
            break

    reverberation_time = -60/slope[1]
    noise_level = 10*np.log10(noise_estimation)  # TO-DO: dB or absolute value?
    intersection_time = crossing_point
    noise_peak_level = 10 * \
        np.log10(
            np.amax(time_window_data[int(np.amin([idx_last_10_percent, idx_10dB_below_crosspoint])):]))

    if plot:
        plt.plot(crossing_point, 10*np.log10(noise_estimation),
                 marker='o', label='final crosspoint')
        plt.axvline(reverberation_time, label=(
            '$T_{60} = ' + str(np.round(reverberation_time, 3)) + '$'))
        #plt.axhline(noise_level, label=('final noise estimation'))
        plt.axvline(intersection_time, label=('lundeby intersection time'), color='r')
        #plt.axhline(noise_peak_level, label=('final noise peak level'))
        plt.tight_layout()
        plt.legend()

    return intersection_time, reverberation_time, noise_level
