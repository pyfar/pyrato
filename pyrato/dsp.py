# -*- coding: utf-8 -*-

"""Signal processing related functions."""

import warnings

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
    impulse_response : ndarray, double
        The impulse response
    threshold : double, optional
        Threshold according to ISO3382 in dB

    Returns
    -------
    start_sample : int
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
    ir_squared = np.abs(impulse_response)**2

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

    start_sample_shape = max_sample.shape
    n_samples = ir_squared.shape[-1]
    ir_squared = np.reshape(ir_squared, (-1, n_samples))
    n_channels = ir_squared.shape[0]
    max_sample = np.reshape(max_sample, n_channels)
    max_value = np.reshape(max_value, n_channels)

    start_sample = max_sample.copy()
    for idx in range(n_channels):
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
    ir_squared = np.abs(impulse_response)**2

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


def time_shift(signal, n_samples_shift, circular_shift=True, keepdims=False):
    """Shift a signal in the time domain by n samples.
    This function will perform a circular shift by default, inherently
    assuming that the signal is periodic. Use the option `circular_shift=False`
    to pad with nan values instead.

    Notes
    -----
    This function is primarily intended to be used when processing impulse
    responses.

    Parameters
    ----------
    signal : ndarray, float
        Signal to be shifted
    n_samples_shift : integer
        Number of samples by which the signal should be shifted. A negative
        number of samples will result in a left-shift, while a positive
        number of samples will result in a right shift of the signal.
    circular_shift : bool, True
        Perform a circular or non-circular shift. If a non-circular shift is
        performed, the data will be padded with nan values at the respective
        beginning or ending of the data, corresponding to the number of samples
        the data is shifted.
    keepdims : bool, False
        Do not squeeze the data before returning.

    Returns
    -------
    shifted_signal : ndarray, float
        Shifted input signal

    """
    n_samples_shift = np.asarray(n_samples_shift, dtype=int)
    if np.any(signal.shape[-1] < n_samples_shift):
        msg = "Shifting by more samples than length of the signal."
        if circular_shift:
            warnings.warn(msg, UserWarning)
        else:
            raise ValueError(msg)

    signal = np.atleast_2d(signal)
    n_samples = signal.shape[-1]
    signal_shape = signal.shape
    signal = np.reshape(signal, (-1, n_samples))
    n_channels = np.prod(signal.shape[:-1])

    if n_samples_shift.size == 1:
        n_samples_shift = np.broadcast_to(n_samples_shift, n_channels)
    elif n_samples_shift.size == n_channels:
        n_samples_shift = np.reshape(n_samples_shift, n_channels)
    else:
        raise ValueError("The number of shift samples has to match the number \
            of signal channels.")

    shifted_signal = signal.copy()
    for channel in range(n_channels):
        shifted_signal[channel, :] = \
            np.roll(
                shifted_signal[channel, :],
                n_samples_shift[channel],
                axis=-1)
        if not circular_shift:
            if n_samples_shift[channel] < 0:
                # index is negative, so index will reference from the
                # end of the array
                shifted_signal[channel, n_samples_shift[channel]:] = np.nan
            else:
                # index is positive, so index will reference from the
                # start of the array
                shifted_signal[channel, :n_samples_shift[channel]] = np.nan

    shifted_signal = np.reshape(shifted_signal, signal_shape)
    if not keepdims:
        shifted_signal = np.squeeze(shifted_signal)

    return shifted_signal


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
