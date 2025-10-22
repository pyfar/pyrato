"""
Sub-module implementing calculations of room acoustic parameters from
simulated or experimental data.
"""
import re
import numpy as np
import pyfar as pf


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


def clarity(energy_decay_curve, early_time_limit=80):
    r"""
    Calculate the clarity from the energy decay curve (EDC).

    The clarity parameter (C50 or C80) is defined as the ratio of early-to-late
    arriving energy in an impulse response and is a measure for how clearly
    speech or music can be perceived in a room. The early-to-late boundary is
    typically set at 50 ms (C50) or 80 ms (C80) [#iso]_.

    Clarity is calculated as:

    .. math::

        C_{t_e} = 10 \log_{10} \frac{
            \displaystyle \int_0^{t_e} p^2(t) \, dt
        }{
            \displaystyle \int_{t_e}^{\infty} p^2(t) \, dt
        }

    where :math:`t_e` is the early time limit and :math:`p(t)` is the pressure
    of a room impulse response. Here, the clarity is efficiently computed
    from the EDC :math:`e(t)` directly by:

    .. math::

        C_{t_e} = 10 \log_{10} \left( \frac{e(0)}{e(t_e)} - 1 \right).

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve (EDC) of the room impulse response
        (time-domain signal). The EDC must start at time zero.
    early_time_limit : float, optional
        Early time limit (:math:`t_e`) in milliseconds. Defaults to 80 (C80).
        Typical values are 50 ms (C50) or 80 ms (C80) [#iso]_.

    Returns
    -------
    clarity : numpy.ndarray[float]
        Clarity index (early-to-late energy ratio) in decibels,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#iso] ISO 3382, Acoustics — Measurement of the reverberation time of
        rooms with reference to other acoustical parameters.

    Examples
    --------
    Estimate the clarity from a real room impulse response filtered in
    octave bands:

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyrato as ra
    >>> rir = pf.signals.files.room_impulse_response(sampling_rate=44100)
    >>> rir = pf.dsp.filter.fractional_octave_bands(rir)
    >>> edc = ra.edc.energy_decay_curve_lundeby(rir)
    >>> C80 = clarity(edc, early_time_limit=80)
    """

    # Check input type
    if not isinstance(energy_decay_curve, pf.TimeData):
        raise TypeError("Input must be a pyfar.TimeData object.")
    if not isinstance(early_time_limit, (int, float)):
        raise TypeError('early_time_limit must be a number.')

    # Validate time range
    if (early_time_limit > energy_decay_curve.signal_length * 1000) or (
            early_time_limit <= 0):
        raise ValueError(
            "early_time_limit must be in the range of 0"
            f"and {energy_decay_curve.signal_length * 1000}.",
            )

    # Raise error if TimeData is complex
    if energy_decay_curve.complex:
        raise ValueError(
            "Complex-valued input detected. Clarity is"
            "only defined for real TimeData.",
        )

    # Convert milliseconds to seconds
    early_time_limit_sec = early_time_limit / 1000.0

    start_vals_energy_decay_curve = energy_decay_curve.time[..., 0]

    idx_early_time_limit = int(
        energy_decay_curve.find_nearest_time(early_time_limit_sec))
    vals_energy_decay_curve = \
        energy_decay_curve.time[..., idx_early_time_limit]
    vals_energy_decay_curve[vals_energy_decay_curve == 0] = np.nan

    clarity = start_vals_energy_decay_curve / vals_energy_decay_curve - 1
    clarity_db = 10 * np.log10(clarity)

    return clarity_db



def clarity_from_energy_balance(energy_decay_curve, early_time_limit=80):
    r"""
    Calculate the clarity from the energy decay curve (EDC).

    The clarity parameter (C50 or C80) is defined as the ratio of early-to-late
    arriving energy in an impulse response and is a measure for how clearly
    speech or music can be perceived in a room. The early-to-late boundary is
    typically set at 50 ms (C50) or 80 ms (C80) [#iso]_.

    Clarity is calculated as:

    .. math::

        C_{t_e} = 10 \log_{10} \frac{
            \displaystyle \int_0^{t_e} p^2(t) \, dt
        }{
            \displaystyle \int_{t_e}^{\infty} p^2(t) \, dt
        }

    where :math:`t_e` is the early time limit and :math:`p(t)` is the pressure
    of a room impulse response. Here, the clarity is efficiently computed
    from the EDC :math:`e(t)` directly by:

    .. math::

        C_{t_e} = 10 \log_{10} \left( \frac{e(0)}{e(t_e)} - 1 \right).

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve (EDC) of the room impulse response
        (time-domain signal). The EDC must start at time zero.
    early_time_limit : float, optional
        Early time limit (:math:`t_e`) in milliseconds. Defaults to 80 (C80).
        Typical values are 50 ms (C50) or 80 ms (C80) [#iso]_.

    Returns
    -------
    clarity : numpy.ndarray[float]
        Clarity index (early-to-late energy ratio) in decibels,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#iso] ISO 3382, Acoustics — Measurement of the reverberation time of
        rooms with reference to other acoustical parameters.

    Examples
    --------
    Estimate the clarity from a real room impulse response filtered in
    octave bands:

    >>> import numpy as np
    >>> import pyfar as pf
    >>> import pyrato as ra
    >>> rir = pf.signals.files.room_impulse_response(sampling_rate=44100)
    >>> rir = pf.dsp.filter.fractional_octave_bands(rir)
    >>> edc = ra.edc.energy_decay_curve_lundeby(rir)
    >>> C80 = clarity(edc, early_time_limit=80)
    """
    # Check input type
    if not isinstance(energy_decay_curve, pf.TimeData):
        raise TypeError("Input must be a pyfar.TimeData object.")

    if not isinstance(early_time_limit, (int, float)):
        raise TypeError('early_time_limit must be a number.')

    # Validate time range
    if (early_time_limit > energy_decay_curve.signal_length * 1000) or (
            early_time_limit <= 0):
        raise ValueError(
            "early_time_limit must be in the range of 0"
            f"and {energy_decay_curve.signal_length * 1000}.",
            )

    # Convert milliseconds to seconds
    early_time_limit_sec = early_time_limit / 1000

    # calculate lim1 - lim4 for each channel
    lim1, lim2 = early_time_limit_sec, energy_decay_curve.time[..., -1]
    lim3, lim4 = 0.0, early_time_limit_sec

    return __energy_balance(energy_decay_curve, lim1, lim2, lim3, lim4)


def __energy_balance(energy_decay_curve, lim1, lim2, lim3, lim4):
    r"""
    Calculate the energy balance for the time limits from the energy
    decay curve (EDC).

    A collection of roomacoustic parameters are defined by their
    time-respective energy balance, where the differentiation is made by
    the four given time limits [#iso]_.

    Energy-Balance is calculated as:

    .. math::

        EB(p) = 10 \log_{10} \frac{
            \displaystyle \int_{lim3}^{lim4} p^2(t) \, dt
        }{
            \displaystyle \int_{lim1}^{lim2} p^2(t) \, dt
        }

    where :math:`lim1 - lim4` are the time limits and :math:`p(t)` is the
    pressure of a room impulse response. Here, the energy balance is
    efficiently computed from the EDC :math:`e(t)` directly by:

    .. math::

        EB(e) = 10 \log_{10} \left( \frac{e(lim3) -
        e(lim4)}{e(lim1) - e(lim2)} \right).

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve (EDC) of the room impulse response
        (time-domain signal). The EDC must start at time zero.
    lim1, lim2, lim3, lim4 : float
        Time limits (:math:`t_e`) in seconds.

    Returns
    -------
    energy balance : numpy.ndarray[float]
        energy-balance index (early-to-late energy ratio) in decibels,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#iso] ISO 3382, Acoustics — Measurement of the reverberation time of
        rooms with reference to other acoustical parameters.
    """

    # Check input type
    if not isinstance(energy_decay_curve, pf.TimeData):
        raise TypeError("Input must be a pyfar.TimeData object.")
    if energy_decay_curve.complex:
        raise ValueError(
            "Complex-valued input detected. Roomacoustic Parameters are"
            "only defined for real TimeData.",
        )

    # for name, val in zip(("lim1","lim2","lim3","lim4"), (lim1, lim2, lim3, lim4)):
    #     if not isinstance(val, numbers.Real):
    #         raise TypeError(f"{name} must be numeric.")

    # # Order validation
    # if lim2 <= lim1:
    #     raise ValueError("lim2 must be greater than lim1.")
    # if lim4 <= lim3:
    #     raise ValueError("lim4 must be greater than lim3.")

    lim1_idx = energy_decay_curve.find_nearest_time(lim1)
    lim2_idx = energy_decay_curve.find_nearest_time(lim2)
    lim3_idx = energy_decay_curve.find_nearest_time(lim3)
    lim4_idx = energy_decay_curve.find_nearest_time(lim4)

    lim1_vals = energy_decay_curve.time[..., lim1_idx]
    lim2_vals = energy_decay_curve.time[..., lim2_idx]
    lim3_vals = energy_decay_curve.time[..., lim3_idx]
    lim4_vals = energy_decay_curve.time[..., lim4_idx]

    numerator = lim3_vals - lim4_vals
    denominator = lim1_vals - lim2_vals

    energy_balance = numerator / denominator
    return 10 * np.log10(energy_balance)
