%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np
from matplotlib import pyplot as plt

def intersection_time_lundeby(rir, sampling_rate, freq='broadband', plot=False):
    """ This function uses the algorithm after Lundeby et al. [1] to calculate the (late)
    reverberation time, intersection time and noise level estimation.

    TO-DO: write test function, ...

    References
    [1] Lundeby, Virgran, Bietz and Vorlaender - Uncertainties of Measurements
        in Room Acoustics - ACUSTICA Vol. 81 (1995)


    Parameters
    ----------
    rir: np.array
        Contains the room impulse response.
    sampling_rate: integer
        Defines the sampling rate of the room impulse response.
    freq: integer OR string
        Defines the frequency band. If set to 'broadband',
        the time window will not be set in dependence of frequency

    Returns
    -------
    reverberation_time: float
        Returns the Lundeby reverberation time.
    intersection_time: float
        Returns the Lundeby intersection time.
    noise_level: float
        Returns the noise level estimation.
    """

    n_intervals_per_10dB = 5   # time intervals per 10 dB decay. Lundeby: 3 ... 10
    dB_above_noise = 10   # end of regression 5 ... 10 dB
    use_dyn_range_for_regression = 20  # 10 ... 20 dB

    if freq == 'broadband':
        freq_dependent_window_time = 0.03  # broadband: use 30 ms windows sizes
    else:
        freq_dependent_window_time = (800/freq+10) / 1000

    # if shortRevTimeMode: # TODO!
    #    freq_dependent_window_time = freq_dependent_window_time / 5;

    raw_time_data = rir**2  # squared room impulse response
    n_samples = raw_time_data.shape[0]
    #n_samples = n_samples - n_samples2cut;

    # TODO: Implement time shift!

    # (1) SMOOTH
    n_samples_per_block = np.round(freq_dependent_window_time * sampling_rate, 0)
    time_window_data = np.squeeze(np.sum(np.reshape(
        raw_time_data[0:int(np.floor(n_samples/n_samples_per_block)*n_samples_per_block)],
        (int(np.floor(n_samples/n_samples_per_block)), int(n_samples_per_block))), 1))/n_samples_per_block
    time_vector_window = np.arange(0, time_window_data.shape[0])*n_samples_per_block/sampling_rate
    time_vector_window2 = np.arange(0, raw_time_data.shape[0])/sampling_rate

    # (...) PLOT FOR DEMONSTRATION
    if plot:
        plt.figure(figsize=(15, 3))
        plt.subplot(131)
        plt.plot(time_vector_window2, 10*np.log10(raw_time_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Original EDC [dB]')
        plt.subplot(132)
        plt.plot(time_vector_window, 10*np.log10(time_window_data))
        plt.xlabel('Time [s]')
        plt.ylabel('Smoothened EDC [dB]')

    # (2) ESTIMATE NOISE
    noise_estimation = np.mean(
        time_window_data[-int(np.round(time_window_data.shape[0]/10)):])+np.finfo(np.double).tiny
    if plot:
        plt.plot(time_vector_window[-int(np.round(time_window_data.shape[0]/10))],
                 10*np.log10(noise_estimation), marker='o', label='noise estimation')

    # (3) REGRESSION

    start_idx = np.argmax(time_window_data)
    stop_idx = np.argwhere(10*np.log10(time_window_data[start_idx+1:-1])
                           > 10*np.log10(noise_estimation) + dB_above_noise)[-1, 0] + start_idx
    dyn_range = np.diff(10*np.log10(np.take(time_window_data, [start_idx, stop_idx])))

    if stop_idx.size == 0 or (stop_idx == start_idx) or dyn_range > -5:
        print('Regression did not work due to low SNR, continuing with next channel/band')

    # regression_matrix*slope = edc
    regression_matrix = np.vstack(
        (np.ones([stop_idx-start_idx]), time_vector_window[start_idx:stop_idx]))
    # TODO: least-squares solution necessary?
    slope = np.linalg.lstsq(regression_matrix.T,
                            (10*np.log10(time_window_data[start_idx:stop_idx])), rcond=None)[0]
    if plot:
        plt.plot([time_vector_window[start_idx], time_vector_window[stop_idx]], [10*np.log10(time_window_data[start_idx]),
                                                                                 10*np.log10(time_window_data[start_idx])+slope[1]*time_vector_window[stop_idx]], marker='o', label='regression')

    if slope[1] == 0 or np.any(np.isnan(slope)):
        print('Regression did not work due, T would be Inf, setting to 0, continuing with next channel/band')
        # TODO: throw error

    # (4) PRELIMINARY CROSSING POINT
    crossing_point = (10*np.log10(noise_estimation) - slope[0]) / slope[1]

    if plot:
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
        raw_time_data[0:int(np.floor(n_samples/n_samples_per_block)*n_samples_per_block)],
        (int(np.floor(n_samples/n_samples_per_block)), int(n_samples_per_block))), 1)/n_samples_per_block)  # np.squeeze notwendig?

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

        stop_idx = np.argwhere(10*np.log10(time_window_data[start_idx+1:]) < (
            10*np.log10(noise_estimation) + dB_above_noise))[0, 0] + start_idx
        if stop_idx.size == 0:
            print('Regression did not work due to low SNR, continuing with next channel/band')
            break

        # regression_matrix*slope = edc
        regression_matrix = np.vstack(
            (np.ones([stop_idx-start_idx]), time_vector_window[start_idx:stop_idx]))
        slope = np.linalg.lstsq(regression_matrix.T,
                                (10*np.log10(time_window_data[start_idx:stop_idx])), rcond=None)[0]

        if slope[1] >= 0:
            print('Regression did not work due, T would be Inf, setting to 0, continuing with next channel/band')
            slope[1] = np.inf
            break

        # (9) FIND CROSSPOINT
        old_crossing_point = crossing_point
        crossing_point = (10*np.log10(noise_estimation) - slope[0]) / slope[1]

        loop_counter = loop_counter + 1
        if loop_counter > 30:
            print('30 iterations => cancel')
            break

    reverberation_time = -60/slope[1]
    noise_level = 10*np.log10(noise_estimation)
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

    return reverberation_time, intersection_time, noise_level
