# -*- coding: utf-8 -*-

"""Signal processing related functions."""

import warnings

import pyfar as pf
import numpy as np


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
    noise_energy : float, str, optional
        If ``'auto'``, the noise level is calculated based on the last 10
        percent of the RIR. Otherwise specify manually for each channel
        as array.

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
            "if this is a valid impulse response with sufficient SNR.",
            stacklevel=2)

    return max_sample


def estimate_noise_energy(
        data,
        interval=[0.9, 1.0],
        is_energy=False):
    """Estimate the noise power of additive noise in impulse responses.

    The noise power is distributed from an interval in which the additive
    noise is assumed to be larger than the impulse response data.

    Parameters
    ----------
    data : pyfar.Signal
        The impulse response.
    interval : tuple, float
        Defines the interval of the RIR to be evaluated for the estimation.
        The interval is relative to the length of the RIR ``0 = 0%, 1=100%``.
        By default ``(0.9, 1.0)``.
    is_energy : bool
        Defines if the data is already squared.

    Returns
    -------
    noise_energy : numpy.ndarray[float]

        The energy of the background noise,shaped according to the channel
        shape of the input Signal.
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
    """Estimate the noise power of additive noise in impulse responses.

    Private function for use with numpy arrays.

    Parameters
    ----------
    energy_data: np.array
        The room impulse response with shape ``(..., n_samples)``.
    interval : tuple, float
        Defines the interval of the RIR to be evaluated for the estimation.
        The interval is relative to the length of the RIR ``0 = 0%, 1=100%``.
        By default ``(0.9, 1.0)``.

    Returns
    -------
    noise_energy : float
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
    """Smoothens the RIR by averaging the data in an specified interval.

    Parameters
    ----------
    data : ndarray, double
        The room impulse response with dimension ``(..., n_samples)``.
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
    """Preprocess the room impulse response for further processing.

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
        The room impulse response with dimension (..., n_samples).
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
        rir_start_idx = pf.dsp.find_impulse_response_start(data)

        if channel_independent and not n_channels == 1:
            shift_samples = -rir_start_idx
        else:
            min_shift = np.amin(rir_start_idx)
            shift_samples = np.asarray(
                -min_shift * np.ones(data.cshape), dtype=int)

        result = pf.dsp.time_shift(
            data, shift_samples, mode='linear', pad_value=np.nan)
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
    is_energy : bool, optional
        Defines if the impulse response is already squared. If set to True,
        the function assumes that the input is already energy data, otherwise
        it squares the input data. The default is False.

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
