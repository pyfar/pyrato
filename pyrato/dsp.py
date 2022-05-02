# -*- coding: utf-8 -*-

"""Signal processing related functions."""

import warnings

import pyfar as pf
import numpy as np
import scipy.signal as spsignal


def find_impulse_response_start(
        impulse_response,
        threshold=20,
        noise_energy='auto'):
    """Find the first sample of an impulse response in a accordance with the
    ISO standard ISO 3382 [#]_. The start sample is identified as the first
    sample that varies significantly from the noise floor but still has a level
    of at least 20 dB below the maximum of the impulse response. The function
    further tries to consider oscillations before the time below the threshold
    value.

    Parameters
    ----------
    impulse_response : pyfar.Signal
        The impulse response
    threshold : double, optional
        Threshold according to ISO3382 in dB

    Returns
    -------
    start_sample : numpy.ndarray, int
        Sample at which the impulse response starts

    Note
    ----
    The function tries to estimate the SNR in the IR based on the signal energy
    in the last 10 percent of the IR.

    References
    ----------
    .. [#]  ISO 3382-1:2009-10, Acoustics - Measurement of the reverberation
            time of rooms with reference to other acoustical parameters. pp. 22
    """
    ir_squared = np.abs(impulse_response.time)**2

    mask_start = np.int(0.9*ir_squared.shape[-1])

    if noise_energy == 'auto':
        mask = np.arange(mask_start, ir_squared.shape[-1])
        noise = np.mean(np.take(ir_squared, mask, axis=-1), axis=-1)
    else:
        noise = noise_energy

    max_sample = np.argmax(ir_squared, axis=-1)
    max_value = np.max(ir_squared, axis=-1)

    if np.any(max_value < 10**(threshold/10) * noise) or \
            np.any(max_sample > mask_start):
        raise ValueError(
            "The SNR is lower than the defined threshold. Check "
            "if this is a valid impulse response with sufficient SNR.")

    start_sample_shape = max_sample.shape
    n_samples = ir_squared.shape[-1]
    ir_squared = np.reshape(ir_squared, (-1, n_samples))
    n_channels = ir_squared.shape[0]
    max_sample = np.reshape(max_sample, n_channels)
    max_value = np.reshape(max_value, n_channels)

    start_sample = max_sample.copy()
    for idx in range(0, n_channels):
        # Only look for the start sample if the maximum index is bigger than 0
        if start_sample[idx] > 0:
            ir_before_max = ir_squared[idx, :max_sample[idx]+1] \
                / max_value[idx]
            # Last value before peak lower than the peak/threshold
            idx_last_below_thresh = np.argwhere(
                ir_before_max < 10**(-threshold/10))
            if idx_last_below_thresh.size > 0:
                start_sample[idx] = idx_last_below_thresh[-1]
            else:
                start_sample[idx] = 0
                warnings.warn(
                    'No values below threshold found before the maximum value,\
                    defaulting to 0')

            idx_6dB_above_threshold = np.argwhere(
                ir_before_max[:start_sample[idx]+1] >
                10**((-threshold+6)/10))
            if idx_6dB_above_threshold.size > 0:
                idx_6dB_above_threshold = int(idx_6dB_above_threshold[0])
                tmp = np.argwhere(
                    ir_before_max[:idx_6dB_above_threshold+1] <
                    10**(-threshold/10))
                if tmp.size == 0:
                    start_sample[idx] = 0
                    warnings.warn(
                        'Oscillations detected in the impulse response. \
                        No clear starting sample found, defaulting to 0')
                else:
                    start_sample[idx] = tmp[-1]

    start_sample = np.reshape(start_sample, start_sample_shape)

    return np.squeeze(start_sample)


def find_impulse_response_maximum(
        impulse_response,
        threshold=20,
        noise_energy='auto'):
    """Find the maximum of an impulse response as argmax(h(t)).
    Performs an initial SNR check according to a defined threshold level in dB.

    Parameters
    ----------
    impulse_response : ndarray, double
        The impulse response
    threshold : double, optional
        Threshold SNR value in dB

    Returns
    -------
    max_sample : int
        Sample at which the impulse response starts

    Note
    ----
    The function tries to estimate the SNR in the IR based on the signal energy
    in the last 10 percent of the IR.

    """
    ir_squared = np.abs(impulse_response.time)**2

    mask_start = int(0.9*ir_squared.shape[-1])
    if noise_energy == 'auto':
        mask = np.arange(mask_start, ir_squared.shape[-1])
        noise = np.mean(np.take(ir_squared, mask, axis=-1), axis=-1)
    else:
        noise = noise_energy

    max_sample = np.argmax(ir_squared, axis=-1)
    max_value = np.max(ir_squared, axis=-1)

    if np.any(max_value < 10**(threshold/10) * noise) or \
            np.any(max_sample > mask_start):
        raise ValueError("The SNR is lower than the defined threshold. Check \
                if this is a valid impulse response with sufficient SNR.")

    return np.squeeze(max_sample)


def time_shift(signal, shift, circular_shift=True, unit='samples'):
    """Apply a time-shift to a signal.

    By default, the shift is performed as a cyclic shift on the time axis,
    potentially resulting in non-causal signals for negative shift values.
    Use the option ``circular_shift=False`` to pad with nan values instead,
    note that in this case the return type will be a ``pyfar.TimeData``.

    Parameters
    ----------
    signal : Signal
        The signal to be shifted
    shift : int, float
        The time-shift value. A positive value will result in right shift on
        the time axis (delaying of the signal), whereas a negative value
        yields a left shift on the time axis (non-causal shift to a earlier
        time). If a single value is given, the same time shift will be applied
        to each channel of the signal. Individual time shifts for each channel
        can be performed by passing an array matching the signals channel
        dimensions ``cshape``.
    unit : str, optional
        Unit of the shift variable, this can be either ``'samples'`` or ``'s'``
        for seconds. By default ``'samples'`` is used. Note that in the case
        of specifying the shift time in seconds, the value is rounded to the
        next integer sample value to perform the shift.
    circular_shift : bool, True
        Perform a circular or non-circular shift. If a non-circular shift is
        performed, the data will be padded with nan values at the respective
        beginning or ending of the data, corresponding to the number of samples
        the data is shifted. In this case, a ``pyfar.TimeData`` object is
        returned.


    Returns
    -------
    pyfar.Signal, pyfar.TimeData
        The time-shifted signal. If a circular shift is performed, the return
        value will be a ``pyfar.Signal``, in case of a non-circular shift, its
        type will be ``pyfar.TimeData``.

    Notes
    -----
    This function is primarily intended to be used when processing room impulse
    responses. When ``circular_shift=True``, the function input is passed into
    ``pyfar.dsp.time_shift``.

    Examples
    --------
    Perform a circular shift a set of ideal impulses stored in three different
    channels and plot the resulting signals

    .. plot::

        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> import matplotlib.pyplot as plt
        >>> impulse = pf.signals.impulse(
        ...     32, amplitude=(1, 1.5, 1), delay=(14, 15, 16))
        >>> shifted = ra.time_shift(impulse, [-2, 0, 2])
        >>> pf.plot.use('light')
        >>> _, axs = plt.subplots(2, 1)
        >>> pf.plot.time(impulse, ax=axs[0])
        >>> pf.plot.time(shifted, ax=axs[1])
        >>> axs[0].set_title('Original signals')
        >>> axs[1].set_title('Shifted signals')
        >>> plt.tight_layout()

    Perform a non-circular shift a single impulse and plot the results.

    .. plot::

        >>> import pyfar as pf
        >>> import pyrato as ra
        >>> import matplotlib.pyplot as plt
        >>> impulse = pf.signals.impulse(32, delay=15)
        >>> shifted = ra.time_shift(impulse, -10, circular_shift=False)
        >>> pf.plot.use('light')
        >>> _, axs = plt.subplots(2, 1)
        >>> pf.plot.time(impulse, ax=axs[0])
        >>> pf.plot.time(shifted, ax=axs[1])
        >>> axs[0].set_title('Original signal')
        >>> axs[1].set_title('Shifted signal')
        >>> plt.tight_layout()

    """
    shift = np.atleast_1d(shift)
    if shift.size == 1:
        shift = np.ones(signal.cshape) * shift

    if unit == 's':
        shift_samples = np.round(shift*signal.sampling_rate).astype(int)
    elif unit == 'samples':
        shift_samples = shift.astype(int)
    else:
        raise ValueError(
            f"Unit is: {unit}, but has to be 'samples' or 's'.")

    shifted = pf.dsp.time_shift(signal, shift_samples, unit='samples')

    if circular_shift is False:
        # Convert to TimeData, as filling with nans will break Fourier trafos
        shifted = pf.TimeData(
            shifted.time,
            shifted.times,
            comment=shifted.comment,
            dtype=shifted.dtype)

        shifted = shifted.flatten()
        shift_samples = shift_samples.flatten()
        for ch in range(shifted.cshape[0]):
            if shift[ch] < 0:
                shifted.time[ch, shift_samples[ch]:] = np.nan
            else:
                shifted.time[ch, :shift_samples[ch]] = np.nan

        shifted = shifted.reshape(signal.cshape)

    return shifted


def center_frequencies_octaves():
    """Return the octave center frequencies according to the IEC 61260:1:2014
    standard.

    Returns
    -------
    frequencies : ndarray, float
        Octave center frequencies

    """
    nominal = np.array([31.5, 63, 125, 250, 500, 1e3,
                        2e3, 4e3, 8e3, 16e3], dtype=np.float)
    indices = _frequency_indices(nominal, 1)
    exact = exact_center_frequencies_fractional_octaves(indices, 1)

    return nominal, exact


def center_frequencies_third_octaves():
    """Return the third octave center frequencies according
    to the ICE 61260:1:2014 standard.

    Returns
    -------
    frequencies : ndarray, float
        third octave center frequencies

    """
    nominal = np.array([
        25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000,
        6300, 8000, 10000, 12500, 16000, 20000], dtype=np.float)

    indices = _frequency_indices(nominal, 3)
    exact = exact_center_frequencies_fractional_octaves(indices, 3)

    return nominal, exact


def exact_center_frequencies_fractional_octaves(indices, num_fractions):
    """Returns the exact center frequencies for fractional octave bands
    according to the IEC 61260:1:2014 standard.

    octave ratio
    .. G = 10^{3/10}

    center frequencies
    .. f_m = f_r G^{x/b}
    .. f_m = f_e G^{(2x+1)/(2b)}

    where b is the number of octave fractions, f_r is the reference frequency
    chosen as 1000Hz and x is the index of the frequency band.

    Parameters
    ----------
    indices : array
        The indices for which the center frequencies are calculated.
    num_fractions : 1, 3
        The number of octave fractions. 1 returns octave center frequencies,
        3 returns third octave center frequencies.

    Returns
    -------
    frequencies : ndarray, float
        center frequencies of the fractional octave bands

    """

    reference_freq = 1e3
    octave_ratio = 10**(3/10)

    iseven = np.mod(num_fractions, 2) == 0
    if ~iseven:
        exponent = (indices/num_fractions)
    else:
        exponent = ((2*indices + 1) / num_fractions / 2)

    return reference_freq * octave_ratio**exponent


def _frequency_indices(frequencies, num_fractions):
    """Return the indices for fractional octave filters.

    Parameters
    ----------
    frequencies : array
        The nominal frequencies for which the indices for exact center
        frequency calculation are to be calculated.
    num_fractions : 1, 3
        Number of fractional bands

    Returns
    -------
    indices : array
        The indices for exact center frequency calculation.

    """
    reference_freq = 1e3
    octave_ratio = 10**(3/10)

    iseven = np.mod(num_fractions, 2) == 0
    if ~iseven:
        indices = np.around(
            num_fractions * np.log(frequencies/reference_freq)
            / np.log(octave_ratio))
    else:
        indices = np.around(
            2.0*num_fractions *
            np.log(frequencies/reference_freq) / np.log(octave_ratio) - 1)/2

    return indices


def filter_fractional_octave_bands(
        signal, samplingrate, num_fractions,
        freq_range=(20.0, 20e3), order=6):
    """Apply a fractional octave filter to a signal.
    Filter bank implementation using second order sections of butterworth
    filters for increased numeric accuracy and stability.

    Parameters
    ----------
    signal : ndarray
        input signal to be filtered
    samplingrate : integer
        samplingrate of the signal
    num_fractions : integer
        number of octave fractions
    order : integer, optional
        order of the butterworth filter

    Returns
    -------
    signal_filtered : ndarray
        Signal filtered into fractional octave bands. The array has a new axis
        with dimension corresponding to the number of frequency bands:
        [num_fractions, *signal.shape]

    """

    if num_fractions not in (1, 3):
        raise ValueError("This currently supports only octave and third \
                octave band filters.")

    octave_ratio = 10**(3/10)

    if num_fractions == 1:
        nominal, exact = center_frequencies_octaves()
    else:
        nominal, exact = center_frequencies_third_octaves()

    mask_frequencies = (nominal > freq_range[0]) & (nominal < freq_range[1])

    nominal = nominal[mask_frequencies]
    exact = exact[mask_frequencies]

    signal_out_shape = (exact.size,) + signal.shape
    signal_out = np.broadcast_to(signal, signal_out_shape).copy()

    for band in range(exact.size):
        freq_upper = exact[band] * octave_ratio**(1/2/num_fractions)
        freq_lower = exact[band] * octave_ratio**(-1/2/num_fractions)

        # normalize interval such that the Nyquist frequency is 1
        Wn = np.array([freq_lower, freq_upper]) / samplingrate * 2
        # in case the upper frequency limit is above Nyquist, use a highpass
        if Wn[-1] > 1:
            warnings.warn('Your upper frequency limit [{}] is above the \
                Nyquist frequency. Using a highpass filter instead of a \
                bandpass'.format(freq_upper))
            Wn = Wn[0]
            btype = 'highpass'
        else:
            btype = 'bandpass'
        sos = spsignal.butter(order, Wn, btype=btype, output='sos')
        signal_out[band, :] = spsignal.sosfilt(sos, signal)

    return signal_out


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
        shift=False,
        channel_independent=False)[0]

    return _estimate_noise_energy(energy_data.time, interval=interval)


def _estimate_noise_energy(
        energy_data,
        interval=[0.9, 1.0]):
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

    if np.any(energy_data) < 0:
        raise ValueError("Energy is negative, check your input signal.")

    region_start_idx = int(energy_data.shape[-1]*interval[0])
    region_end_idx = int(energy_data.shape[-1]*interval[1])
    mask = np.arange(region_start_idx, region_end_idx)
    noise_energy = np.nanmean(np.take(energy_data, mask, axis=-1), axis=-1)

    return noise_energy


def _smooth_rir(
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


def preprocess_rir(
        data,
        is_energy=False,
        shift=False,
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
    shift : boolean
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
    times = data.times
    n_channels = np.prod(data.cshape)

    data_shape = list(data.cshape)
    data = data.reshape((-1,))

    if shift:
        rir_start_idx = find_impulse_response_start(data)

        if channel_independent and not n_channels == 1:
            shift_samples = -rir_start_idx
        else:
            min_shift = np.amin(rir_start_idx)
            shift_samples = np.asarray(
                -min_shift * np.ones(n_channels), dtype=int)

        result = time_shift(
            data, shift_samples, circular_shift=False)
    else:
        result = data

    if not is_energy:
        energy_data = np.abs(result.time)**2
    else:
        energy_data = result.time.copy()

    energy_data = pf.TimeData(energy_data, times)

    return energy_data, n_channels, data_shape
