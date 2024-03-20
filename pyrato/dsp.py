# -*- coding: utf-8 -*-

"""Signal processing related functions."""

import warnings

import pyfar as pf
import numpy as np


def find_impulse_response_start(
        impulse_response,
        threshold=20):
    """Find the start sample of an impulse response.
    The start sample is identified as the first sample which is below the
    ``threshold`` level relative to the maximum level of the impulse response.
    For room impulse responses, ISO 3382 [#]_ specifies a threshold of 20 dB.
    This function is primary intended to be used when processing room impulse
    responses.

    Parameters
    ----------
    impulse_response : pyfar.Signal
        The impulse response
    threshold : float, optional
        The threshold level in dB, by default 20, which complies with ISO 3382.
    Returns
    -------
    start_sample : numpy.ndarray, int
        Sample at which the impulse response starts
    Notes
    -----
    The function tries to estimate the PSNR in the IR based on the signal
    power in the last 10 percent of the IR. The automatic estimation may fail
    if the noise spectrum is not white or the impulse response contains
    non-linear distortions. If the PSNR is lower than the specified threshold,
    the function will issue a warning.
    References
    ----------
    .. [#]  ISO 3382-1:2009-10, Acoustics - Measurement of the reverberation
            time of rooms with reference to other acoustical parameters. pp. 22
    Examples
    --------
    Create a band-limited impulse shifted by 0.5 samples and estimate the
    starting sample of the impulse and plot.
    .. plot::
        >>> import pyfar as pf
        >>> import numpy as np
        >>> n_samples = 256
        >>> delay_samples = n_samples // 2 + 1/2
        >>> ir = pf.signals.impulse(n_samples)
        >>> ir = pf.dsp.linear_phase(ir, delay_samples, unit='samples')
        >>> start_samples = pf.dsp.find_impulse_response_start(ir)
        >>> ax = pf.plot.time(ir, unit='ms', label='impulse response', dB=True)
        >>> ax.axvline(
        ...     start_samples/ir.sampling_rate*1e3,
        ...     color='k', linestyle='-.', label='start sample')
        >>> ax.axhline(
        ...     20*np.log10(np.max(np.abs(ir.time)))-20,
        ...     color='k', linestyle=':', label='threshold')
        >>> ax.legend()
    Create a train of weighted impulses with levels below and above the
    threshold, serving as a very abstract room impulse response. The starting
    sample is identified as the last sample below the threshold relative to the
    maximum of the impulse response.
    .. plot::
        >>> import pyfar as pf
        >>> import numpy as np
        >>> n_samples = 64
        >>> delays = np.array([14, 22, 26, 30, 33])
        >>> amplitudes = np.array([-35, -22, -6, 0, -9], dtype=float)
        >>> ir = pf.signals.impulse(n_samples, delays, 10**(amplitudes/20))
        >>> ir.time = np.sum(ir.time, axis=0)
        >>> start_sample_est = pf.dsp.find_impulse_response_start(
        ...     ir, threshold=20)
        >>> ax = pf.plot.time(
        ...     ir, dB=True, unit='samples',
        ...     label=f'peak samples: {delays}')
        >>> ax.axvline(
        ...     start_sample_est, linestyle='-.', color='k',
        ...     label=f'ir start sample: {start_sample_est}')
        >>> ax.axhline(
        ...     20*np.log10(np.max(np.abs(ir.time)))-20,
        ...     color='k', linestyle=':', label='threshold')
        >>> ax.legend()
    """
    warnings.warn(
        "This function will be deprecated in version 0.5.0 "
        "Use pyfar.dsp.find_impulse_response_start instead",
        DeprecationWarning)

    return pf.dsp.find_impulse_response_start(impulse_response, threshold)


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
        warnings.warn(
            "The SNR seems lower than the specified threshold value. Check "
            "if this is a valid impulse response with sufficient SNR.")

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
            comment=shifted.comment)

        for ch in np.ndindex(shifted.cshape):
            if shift[ch] < 0:
                shifted.time[ch, shift_samples[ch]:] = np.nan
            else:
                shifted.time[ch, :shift_samples[ch]] = np.nan

    return shifted


def center_frequencies_octaves():
    """Return the octave center frequencies according to the IEC 61260:1:2014
    standard.
    Returns
    -------
    frequencies : ndarray, float
        Octave center frequencies
    """
    warnings.warn(
        "This function will be deprecated in version 0.5.0 "
        "Use pyfar.dsp.filter.fractional_octave_frequencies instead",
        DeprecationWarning)

    nominal, exact = pf.dsp.filter.fractional_octave_frequencies(
        1, (20, 20e3), return_cutoff=False)

    return nominal, exact


def center_frequencies_third_octaves():
    """Return the third octave center frequencies according
    to the ICE 61260:1:2014 standard.
    Returns
    -------
    frequencies : ndarray, float
        third octave center frequencies
    """
    warnings.warn(
        "This function will be deprecated in version 0.5.0 "
        "Use pyfar.dsp.filter.fractional_octave_frequencies instead",
        DeprecationWarning)

    nominal, exact = pf.dsp.filter.fractional_octave_frequencies(
        3, (20, 20e3), return_cutoff=False)

    return nominal, exact


def filter_fractional_octave_bands(
        signal, num_fractions,
        freq_range=(20.0, 20e3), order=6):
    """Apply a fractional octave filter to a signal.
    Filter bank implementation using second order sections of butterworth
    filters for increased numeric accuracy and stability.

    Parameters
    ----------
    signal : ndarray
        input signal to be filtered
    num_fractions : integer
        number of octave fractions
    order : integer, optional
        order of the butterworth filter

    Returns
    -------
    signal_filtered : ndarray
        Signal filtered into fractional octave bands.
    """
    warnings.warn(
        "This function will be deprecated in version 0.5.0 "
        "Use pyfar.dsp.filter.fractional_octave_bands instead",
        DeprecationWarning)

    return pf.dsp.filter.fractional_octave_bands(
        signal, num_fractions, freq_range=freq_range, order=order)


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
        channel_independent=False)

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

    """
    times = data.times
    n_channels = np.prod(data.cshape)

    if shift:
        rir_start_idx = find_impulse_response_start(data)

        if channel_independent and not n_channels == 1:
            shift_samples = -rir_start_idx
        else:
            min_shift = np.amin(rir_start_idx)
            shift_samples = np.asarray(
                -min_shift * np.ones(data.cshape), dtype=int)

        result = time_shift(
            data, shift_samples, circular_shift=False)
    else:
        result = data

    if not is_energy:
        energy_data = np.abs(result.time)**2
    else:
        energy_data = result.time.copy()

    energy_data = pf.TimeData(energy_data, times)

    return energy_data


def peak_signal_to_noise_ratio(
        impulse_response,
        noise_power='auto',
        is_energy=False):
    """Calculate the peak-signal-to-noise-ratio of an impulse response.

    Parameters
    ----------
    impulse_response : pyfar.Signal
        The impulse response
    noise_power : float, str, optional
        The noise power. The default is 'auto', in which case the noise power
        is estimated from the last 10 % of the impulse response.

    Returns
    -------
    float, array-like
        The estimated peak-signal-to-noise-ratio for each channel of the
        impulse response.

    """
    data = impulse_response.time
    if is_energy is False:
        data = data**2

    if noise_power == 'auto':
        noise_power = _estimate_noise_energy(data)

    return np.max(data, axis=-1) / noise_power
