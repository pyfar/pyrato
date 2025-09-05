"""
Sub-module implementing calculations of room acoustic parameters from
simulated or experimental data.
"""
import re
import numpy as np
import pyfar as pf
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
    >>> t_20 = ra.reverberation_time_linear_regression(edc, 'T20')
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






def clarity(RIR, early_time_limit=80):
    """
    Calculate the clarity of a room impulse response.

    The clarity parameter (C50 or C80) is defined as the ratio of early-to-late
    arriving energy in an impulse response and describes how clearly speech or
    music can be perceived in a room. The early-to-late boundary is typically
    set at 50 ms (C50) or 80 ms (C80).

    Parameters
    ----------
    RIR : pyfar.Signal
        Room impulse response (time-domain signal).
    early_time_limit : float, optional
        Early time limit in milliseconds. Defaults to 80 (C80). Typical values
        are 50 ms (C50) or 80 ms (C80).
    frequencies : None or ndarray, optional
        Placeholder for octave/third-octave band center frequencies if the
        result should be returned as a frequency-domain representation. The
        filtering must be applied by the user prior to calling this function.

    Returns
    -------
    clarity : ndarray of float
        Clarity index (early-to-late energy ratio) in decibels, shaped according
        to the channel structure of ``RIR``.

    References
    ----------
    ISO 3382-1 : Annex A

    Examples
    --------

    Estimate the clarity from a real room impulse response and octave-band
    filtering:

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyrato as ra
    >>> RIR = pf.signals.files.room_impulse_response(sampling_rate=44100)
    >>> RIR = pf.dsp.filter.fractional_octave_bands(RIR, bands_per_octave=3)
    >>> C80 = ra.parameters.clarity(RIR, early_time_limit=80)

    """

    if not hasattr(RIR, "cshape") or not hasattr(RIR, "sampling_rate"):
        raise AttributeError("clarity() requires a signal object as input.")

    # warnign for unusual early_time_limit
    if early_time_limit not in (50, 80):
        warnings.warn(
            f"early_time_limit={early_time_limit}ms is unusual. "
            "Typically 50ms (C50) or 80ms (C80) are chosen.",
            UserWarning
        )
    signal_length_ms = (RIR.signal_length) * 1000
    if early_time_limit > signal_length_ms:
        raise ValueError("early_time_limit cannot be larger than signal length.")
    if early_time_limit <= 0:
        raise ValueError("early_time_limit must be positive.")

    if RIR.complex:
        warnings.warn(
            "Complex-valued input detected. Clarity is only defined for real "
            "signals and will be computed using |x(t)|^2.",
            UserWarning,
        )
    
    # convert milliseconds to seconds for index lookup
    early_time_limit_sec = early_time_limit / 1000

    channel_shape = RIR.cshape
    RIR_flat = RIR.flatten()

    clarity_vals = []

    for rir in RIR_flat:
        start_index = pf.dsp.find_impulse_response_start(rir)[0]
        early_time_limit_index = int(rir.find_nearest_time(early_time_limit_sec))

        # energy from squared amplitude
        energy_decay = np.abs(rir.time) ** 2

        early_energy = np.sum(energy_decay[:, start_index:early_time_limit_index])
        late_energy = np.sum(energy_decay[:, early_time_limit_index:])

        if early_energy == 0 and late_energy == 0:
            val = np.nan
        elif early_energy == 0:
            val = -np.inf
        elif late_energy == 0:
            val = np.inf
        else:
            val = 10 * np.log10(early_energy / late_energy)

        clarity_vals.append(val)

    clarity = np.array(clarity_vals).reshape(channel_shape)
    return clarity


