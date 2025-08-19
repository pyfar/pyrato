"""
Sub-module implementing calculations of room acoustic parameters from
simulated or experimental data.
"""
import re
import numpy as np
import pyfar.signals as pysi
from . import dsp
import warnings


def reverberation_time_linear_regression(
        energy_decay_curve, T='T20', return_intercept=False):
    """Estimate the reverberation time from a given energy decay curve.

    The linear regression is performed using least squares error minimization
    according to the ISO standard 3382 [#]_.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve. The time needs to be the arrays last dimension.
    T : 'T15', 'T20', 'T30', 'T40', 'T50', 'T60', 'EDT', 'LDT'
        Decay interval to be used for the reverberation time extrapolation. EDT
        corresponds to the early decay time extrapolated from the interval
        ``[0, -10]`` dB, LDT corresponds to the late decay time extrapolated
        from the interval ``[-25, -35]`` dB.
    return_intercept : bool
        If True, the intercept of the linear regression is returned in addition
        to the reverberation time. The default is False.

    Returns
    -------
    reverberation_time : double
        The reverberation time

    References
    ----------
    .. [#] ISO 3382, Acoustics - Measurement of the reverberation time of
           rooms with reference to other acoustical parameters.

    Examples
    --------
    Estimate the reverberation time from an energy decay curve.

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyrato as ra
    >>> from pyrato.analytic import rectangular_room_rigid_walls
    ...
    >>> L = np.array([8, 5, 3])/10
    >>> source_pos = np.array([5, 3, 1.2])/10
    >>> receiver_pos = np.array([1, 1, 1.2])/10
    >>> rir, _ = rectangular_room_rigid_walls(
    ...     L, source_pos, receiver_pos,
    ...     reverberation_time=1, max_freq=1.5e3, n_samples=2**12,
    ...     speed_of_sound=343.9, samplingrate=3e3)
    >>> rir = rir/rir.time.max()
    ...
    >>> awgn = pf.signals.noise(
    ...     rir.n_samples, rms=10**(-50/20),
    ...     sampling_rate=rir.sampling_rate)
    >>> rir = rir + awgn
    ...
    >>> edc = ra.energy_decay_curve_chu_lundeby(rir)
    >>> t_20 = ra.parameters.reverberation_time_linear_regression(edc, 'T20')
    >>> t_20
    ...     array([0.99526253])

    """

    if T == 'EDT':
        upper = -0.1
        lower = -10.1
    elif T == 'LDT':
        upper = -25.
        lower = -35.
    else:
        if T not in ['T15', 'T20', 'T30', 'T40', 'T50', 'T60']:
            raise ValueError(
                f"{T} is not a valid interval for the regression.")
        upper = -5
        lower = -np.double(re.findall(r'\d+', T)) + upper

    edcs_db = 10*np.log10(np.abs(energy_decay_curve.time))
    times = energy_decay_curve.times

    reverberation_times = np.zeros(energy_decay_curve.cshape, dtype=float)
    intercepts = np.zeros(energy_decay_curve.cshape, dtype=float)

    for ch in np.ndindex(energy_decay_curve.cshape):
        edc_db = edcs_db[ch]
        idx_upper = np.nanargmin(np.abs(upper - edc_db))
        idx_lower = np.nanargmin(np.abs(lower - edc_db))

        A = np.vstack(
            [times[idx_upper:idx_lower], np.ones(idx_lower - idx_upper)]).T
        gradient, const = np.linalg.lstsq(
            A, edc_db[..., idx_upper:idx_lower], rcond=None)[0]

        reverberation_times[ch] = -60 / gradient
        intercepts[ch] = 10**(const/10)

    if return_intercept is True:
        return reverberation_times, intercepts
    else:
        return reverberation_times





import numpy as np
import pyfar as pf
import pyfar.dsp as dsp

def clarity(RIR, early_time_limit=80):
    """Calculate the clarity of a signal in a room. 
    
    The clarity parameter is calculated with the early-to-late index at 50 ms or 80 ms and describes how 
    clearly someone can hear sound and music in a room

    Parameters
    ----------
    RIR : pyfar.Signal
        Room impulse response (or energy decay curve)
    early_time_limit : float [s]
        Early time limit to calculate the clarity as a scalar in seconds
        Typically 0.05 (C50) or 0.08 (C80).

    Returns
    -------
    clarity : ndarray [dB]
        Clarity index (early-to-late energy ratio) in decibel,
        shaped according to the channel structure of RIR.

    Reference
    ---------
    ISO3382-1 : Annex A
    """
    if not hasattr(RIR, "cshape") or not hasattr(RIR, "sampling_rate"):
        raise AttributeError("clarity() requires a Signal object as input.")

    # warnign for unusual early_time_limit
    if early_time_limit not in (50, 80):
        warnings.warn(
            f"early_time_limit={early_time_limit}s is unusual. "
            "Typically 50ms (C50) or 80ms (C80) are used.",
            UserWarning
        )

    # get channel shape & flatten audio object
    channel_shape = RIR.cshape
    RIR_flat = RIR.flatten()

    clarity_vals = []

    # iterate over flattended channels
    for rir in RIR_flat:

        # start-index
        start_index = dsp.find_impulse_response_start(rir)[0]

        # early_time_limit-index
        early_time_limit_index = int(rir.find_nearest_time(early_time_limit/1000))

        # calculate edc
        if rir.signal_type == "energy":
            energy_decay = rir.time
        else:
            energy_decay = rir.time**2

        # late- and early energy
        energy_decay_early = np.sum(energy_decay[:,start_index:early_time_limit_index])
        energy_decay_late = np.sum(energy_decay[:,early_time_limit_index:])

        # clarity fraction incl. edge case handling
        if energy_decay_early == 0 and energy_decay_late == 0:
            val = np.nan
        elif energy_decay_early == 0:
            val = -np.inf
        elif energy_decay_late == 0:
            val = np.inf
        else:
            val = 10 * np.log10(energy_decay_early / energy_decay_late)

        clarity_vals.append(val)

    # reshape array to channel_shape
    clarity_vals = np.array(clarity_vals).reshape(channel_shape)

    return clarity_vals


