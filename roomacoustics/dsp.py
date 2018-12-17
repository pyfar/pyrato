# -*- coding: utf-8 -*-

"""Main module."""

import re
import numpy as np
import scipy.signal as ssignal


def find_impulse_response_start(impulse_response, threshold=20):
    """Find the first sample of an impulse response in a accordance with the
    ISO standard ISO 3382.

    Parameters
    ----------
    impulse_response : ndarray, double
        The impulse response
    threshold : double, optional

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
    .. [1]  ISO 3382, Acoustics - Measurement of the reverberation time of rooms
            with reference to other acoustical parameters.

    """
    ir_squared = np.abs(impulse_response)**2

    mask_start = np.int(0.9*ir_squared.shape[-1])
    mask = np.arange(mask_start, ir_squared.shape[-1])
    noise = np.mean(np.take(ir_squared, mask, axis=-1), axis=-1)

    max_sample = np.argmax(ir_squared, axis=-1)
    max_value = np.max(ir_squared, axis=-1)

    if np.any(max_value < 10**(threshold/10) * noise) or np.any(max_sample > mask_start):
        raise ValueError("The SNR is lower than the defined threshold. Check \
                if this is a valid impulse resonse with sufficient SNR.")

    ir_shape = ir_squared.shape
    start_sample_shape = max_sample.shape
    n_samples = ir_squared.shape[-1]
    ir_squared = np.reshape(ir_squared, (-1, n_samples))
    n_channels = ir_squared.shape[0]
    max_sample = np.reshape(max_sample, n_channels)
    max_value = np.reshape(max_value, n_channels)

    start_sample = max_sample
    for idx in range(0, n_channels):
        if start_sample[idx] != 0:
            mask_before_max = np.arange(0, max_sample[idx])
            ir_before_max = ir_squared[idx, :max_sample[idx]+1] / max_value[idx]
            start_sample[idx] = np.argwhere(ir_before_max < 10**(threshold/10))[-1]

    start_sample = np.reshape(start_sample, start_sample_shape)

    return np.squeeze(start_sample)
