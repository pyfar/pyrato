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
    typically set at 50 ms (C50) or 80 ms (C80) [#isoClarity]_.

    Clarity is calculated as:

    .. math::

        C_{t_e} = 10 \log_{10} \frac{
            \displaystyle \int_0^{t_e} p^2(t) \, dt
        }{
            \displaystyle \int_{t_e}^{\infty} p^2(t) \, dt
        }

    where :math:`t_e` is the early time limit and :math:`p(t)` is the pressure
    of a room impulse response. Here, the clarity is efficiently computed
    from the EDC :math:`e(t)`

    .. math::

        C_{t_e} = 10 \log_{10} \frac{e(0) - e(t_e)}{e(t_e) - e(\infty)}
                = 10 \log_{10} \left( \frac{e(0)}{e(t_e)} - 1 \right),

    where :math:`e(\infty) = 0` by definition of the EDC.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve (EDC) of the room impulse response
        (time-domain signal). The EDC must start at time zero.
    early_time_limit : float, optional
        Early time limit (:math:`t_e`) in milliseconds. Defaults to 80 (C80).
        Typical values are 50 ms (C50) or 80 ms (C80) [#isoClarity]_.

    Returns
    -------
    clarity : numpy.ndarray[float]
        Clarity index (early-to-late energy ratio) in decibels,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#isoClarity] ISO 3382, Acoustics — Measurement of the reverberation
        time of rooms with reference to other acoustical parameters.

    Examples
    --------
    Estimate the clarity from a real room impulse response filtered in
    octave bands:

    >>> import pyfar as pf
    >>> import pyrato as ra
    ...
    >>> rir = pf.signals.files.room_impulse_response(sampling_rate=44100)
    >>> rir = pf.dsp.filter.fractional_octave_bands(rir, num_fractions=1)
    >>> edc = ra.edc.energy_decay_curve_lundeby(rir)
    ...
    >>> C80 = ra.parameters.clarity(edc, early_time_limit=80)
    >>> C80
    ...     [[-55.57140506]
    ...     [-11.75657677]
    ...     [ -3.21150787]
    ...     [  2.76276817]
    ...     [  4.70786211]
    ...     [  5.98148157]
    ...     [  9.66764094]
    ...     [  9.08687417]
    ...     [ 14.14550646]
    ...     [ 21.60048332]]
    """

    if not isinstance(early_time_limit, (int, float)):
        raise TypeError('early_time_limit must be a number.')

    # Convert milliseconds to seconds
    early_time_limit_sec = early_time_limit / 1000

    limits = np.array([early_time_limit_sec,
                       np.inf,
                       0.0,
                       early_time_limit_sec])

    return 10*np.log10(_energy_ratio(limits,
                                     energy_decay_curve,
                                     energy_decay_curve))


def early_lateral_energy_fraction(energy_decay_curve_omni,
                                  energy_decay_curve_lateral):
    r"""
    Calculate the early lateral energy fraction.

    The early lateral energy fraction :math:`J_\mathrm{LF}`
    according to [#isoEarlyLat]_ is defined as the ratio between the
    lateral sound energy captured with a figure of eight microphone
    arriving between 5 ms and 80 ms and the total sound energy
    captured with an omnidirectional microphone arriving within
    the first 80 ms after the direct sound. It is a measure of the
    apparent source width.

    The parameter is defined as

    .. math::

        J_\mathrm{LF} =
        \frac{
            \displaystyle \int_{0.005}^{0.08} p_\mathrm{L}^2(t)\,\mathrm{d}t
        }{
            \displaystyle \int_{0}^{0.08} p^2(t)\,\mathrm{d}t
        }

    where :math:`p_\mathrm{L}(t)` is the lateral sound pressure measured with a
    figure-eight microphone whose zero axis is oriented towards the source,
    and :math:`p(t)` is the sound pressure measured at the same position
    with an omnidirectional microphone.

    Using the energy decay curves of the omnidirectional response
    :math:`e(t)` and the lateral response :math:`e_\mathrm{L}(t)`, the
    parameter can be computed efficiently as

    .. math::

        J_\mathrm{LF} =
        \frac{
            e_\mathrm{L}(0.005) - e_\mathrm{L}(0.08)
        }{
            e(0) - e(0.08)
        }.

    Parameters
    ----------
    energy_decay_curve_omni : pyfar.TimeData
        Energy decay curve of the room impulse response measured with an
        omnidirectional microphone. The EDC must start at time zero.

    energy_decay_curve_lateral : pyfar.TimeData
        Energy decay curve of the room impulse response measured with a
        figure-eight microphone oriented according to [#isoEarlyLat]_
        (zero axis pointing towards the source). The EDC must start at
        time zero.
        Both EDCs must have identical ``signal.cshape``.

    Returns
    -------
    Early Lateral Energy Fraction: numpy.ndarray
        Early lateral energy fraction (:math:`J_\mathrm{LF}`) in decibels,
        shaped according to the channel shape of the input EDCs.

    References
    ----------
    .. [#isoEarlyLat] ISO 3382, Acoustics — Measurement of the reverberation
        time of rooms with reference to other acoustical parameters.
    """

    limits = np.array([0.0, 0.08, 0.005, 0.08])

    return _energy_ratio(limits,
                         energy_decay_curve_omni,
                         energy_decay_curve_lateral)



def definition(energy_decay_curve, early_time_limit=50):
    r"""
    Calculate the definition from the energy decay curve (EDC).

    The definition parameter (D50) is defined as the ratio of early-to-total
    arriving energy in an impulse response and is a measure for how defined
    speech or music can be perceived in a room. The early-to-total boundary is
    typically set at 50 ms (D50) [#isoDefinition]_.

    Definition is calculated as:

    .. math::

        D_{t_\mathrm{e}} = \frac{
            \displaystyle \int_0^{t_\mathrm{e}} p^2(t) \, dt
        }{
            \displaystyle \int_{0}^{\infty} p^2(t) \, dt
        }

    where :math:`t_e` is the early time limit and :math:`p(t)` is the pressure
    of a room impulse response. Here, the definition is efficiently computed
    from the EDC :math:`e(t)` directly by:

    .. math::

        D_{t_\mathrm{e}} = \frac{e(0) - e(t_\mathrm{e})}{e(0) - e(\infty)}
                = 1 - \left( \frac{e(t_\mathrm{e})}{e(0)} \right),

    where :math:`e(\infty) = 0` by definition of the EDC.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve (EDC) of the room impulse response
        (time-domain signal). The EDC must start at time zero.
    early_time_limit : float, optional
        Early time limit (:math:`t_\mathrm{e}`) in milliseconds. Defaults to
        typical value 50 (D50) [#isoDefinition]_.

    Returns
    -------
    definition : numpy.ndarray[float]
        Definition index (early-to-total energy ratio),
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#isoDefinition] ISO 3382, Acoustics — Measurement of the reverberation
        time of rooms with reference to other acoustical parameters.

    Examples
    --------
    Estimate the defintion from a real room impulse response filtered in
    octave bands:

    >>> import pyfar as pf
    >>> import pyrato
    ...
    >>> rir = pf.signals.files.room_impulse_response(sampling_rate=44100)
    >>> rir = pf.dsp.filter.fractional_octave_bands(
    >>>     rir, num_fractions=1, frequency_range=(125, 20e3))
    >>> edc = pyrato.edc.energy_decay_curve_lundeby(rir)
    ...
    >>> D50 = pyrato.parameters.definition(edc, early_time_limit=50)
    >>> D50
    ...     [[0.25984852]
    ...     [0.50208742]
    ...     [0.66722359]
    ...     [0.73528532]
    ...     [0.87801455]
    ...     [0.82757594]
    ...     [0.86536142]
    ...     [0.87374988]]
    """

    if not isinstance(early_time_limit, (int, float)):
        raise TypeError('early_time_limit must be a number.')

    # Convert milliseconds to seconds
    early_time_limit_sec = early_time_limit / 1000

    limits = np.array([0.0, np.inf, 0.0, early_time_limit_sec])

    return _energy_ratio(limits,
                         energy_decay_curve,
                         energy_decay_curve)

def _energy_ratio(limits, energy_decay_curve1, energy_decay_curve2):
    r"""
    Calculate the energy ratio for the time limits from two energy
    decay curves (EDC).

    A variety of room-acoustic parameters are defined by energy ratios derived
    from one or two time-domain Energy Decay Curves (EDCs). These parameters
    distinguish between time regions using the four provided limits, and some,
    such as Strength (:math:`G`), Early lateral sound (:math:`J_\mathrm{LF}`),
    and Late lateral sound (:math:`L_J`), require EDCs obtained from different
    impulse-response measurements [#iso2]_.

    Energy-Ratio is calculated as:

    .. math::

        ER = \frac{
            \displaystyle \int_{lim3}^{lim4} p_2^2(t) \, dt
        }{
            \displaystyle \int_{lim1}^{lim2} p_1^2(t) \, dt
        }
    where :math:`[lim1, ..., lim4]` are the time limits and :math:`p(t)` is the
    pressure of a room impulse response. Here, the energy ratio is
    efficiently computed from the EDC :math:`e(t)` directly by:

    .. math::

        ER = \frac{
            \displaystyle e_2(lim3) - e_2(lim4)
        }{
            \displaystyle e_1(lim1) - e_1(lim2)
        }.

    By definition, the EDC represents the remaining energy up to
    :math:`\infty` and converges to zero, i.e., :math:`e(\infty)=0`.
    Thus, ``np.inf`` may be used as a limit to select the full
    remaining energy of an EDC.

    Parameters
    ----------
    limits : np.ndarray, list or tuple
        Four time limits (:math:`t_e`) in seconds, shape (4,)
        in ascending order. Limits must be either numerical or np.inf.
    energy_decay_curve1 : pyfar.TimeData
        Energy decay curve 1 (EDC1) of the room impulse response
        (time-domain signal). The EDC must start at time zero.
    energy_decay_curve2 : pyfar.TimeData
        Energy decay curve 2 (EDC2) of the room impulse response
        (time-domain signal). The EDC must start at time zero.

    Returns
    -------
    energy ratio : numpy.ndarray[float]
        energy-ratio index (early-to-late energy ratio),
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#iso2] ISO 3382, Acoustics — Measurement of the reverberation time of
        rooms with reference to other acoustical parameters.
    """

    # Check input type
    if not isinstance(energy_decay_curve1, pf.TimeData):
        raise TypeError(
            "energy_decay_curve1 must be a pyfar.TimeData or derived object.")
    if not isinstance(energy_decay_curve2, pf.TimeData):
        raise TypeError(
            "energy_decay_curve2 must be a pyfar.TimeData or derived object.")

    # Check that both EDCs have the same channel shape
    if energy_decay_curve1.cshape != energy_decay_curve2.cshape:
        raise ValueError(
            f"energy_decay_curve1 and energy_decay_curve2 must have the same "
            f"channel shape. Got cshape={energy_decay_curve1.cshape} and "
            f"cshape={energy_decay_curve2.cshape}.")

    if isinstance(limits, (list, tuple)):
        limits = np.asarray(limits)

    if not isinstance(limits, np.ndarray):
        raise TypeError("limits must be a numpy ndarray, list, or tuple.")

    # Check shape
    if limits.shape != (4,):
        raise ValueError(
            "limits must have shape (4,), " \
            "containing [lim1, lim2, lim3, lim4].",
            )

    # Check if limits are within valid time range
    if (
        np.any(limits[0:2] < 0) or
        np.any((limits[0:2] > energy_decay_curve1.signal_length) &
               (np.isfinite(limits[0:2])))
    ):
        raise ValueError(
            f"limits[0:2] must be between 0 and "
            f"{energy_decay_curve1.signal_length} seconds or np.inf.",
        )
    if (
        np.any(limits[2:4] < 0) or
        np.any((limits[2:4] > energy_decay_curve2.signal_length) &
               (np.isfinite(limits[2:4])))
    ):
        raise ValueError(
            f"limits[2:4] must be between 0 and "
            f"{energy_decay_curve2.signal_length} seconds or np.inf.",
        )

    limits_denominator = limits[0:2]
    limits_numerator = limits[2:4]

    # finite limits mask
    finite_limits_denominator = np.isfinite(limits_denominator)
    finite_limits_numerator = np.isfinite(limits_numerator)

    # Denominator values (EDC1, limits 0:2)
    values_shape1 = energy_decay_curve1.cshape + (2,)
    energy_decay_curve1_values = np.zeros(values_shape1)
    if np.any(finite_limits_denominator):
        limits_energy_decay_curve1_idx = np.atleast_1d(
            energy_decay_curve1.find_nearest_time(
                limits_denominator[finite_limits_denominator],
            ),
        )

        energy_decay_curve1_values[..., finite_limits_denominator] = \
            energy_decay_curve1.time[..., limits_energy_decay_curve1_idx]

    # Numerator values (EDC2, limits 2:4)
    values_shape2 = energy_decay_curve2.cshape + (2,)
    energy_decay_curve2_values = np.zeros(values_shape2)
    if np.any(finite_limits_numerator):
        limits_energy_decay_curve2_idx = np.atleast_1d(
            energy_decay_curve2.find_nearest_time(
                limits_numerator[finite_limits_numerator],
            ),
        )

        energy_decay_curve2_values[..., finite_limits_numerator] = \
            energy_decay_curve2.time[..., limits_energy_decay_curve2_idx]

    # using 'minus' because np.diff yields negative result
    numerator = -np.diff(energy_decay_curve2_values, axis=-1)[..., 0]
    denominator = -np.diff(energy_decay_curve1_values, axis=-1)[..., 0]

    energy_ratio = numerator / denominator

    return energy_ratio

def late_lateral_sound_level(energy_decay_curve_ref_10m,
                       energy_decay_curve_lateral):
    r"""
    Calculate the late lateral sound level.

    The late lateral sound level :math:`L_\mathrm{J}` is a measure of the
    strength of late-arriving lateral sound.
    According to [#isoLLJ]_, it is defined as the level ratio between the
    lateral sound energy captured with a figure of eight microphone
    arriving after 80 ms and the total sound energy of a reference
    impulse response measured with an omnidirectional microphone
    at a distance of 10 m in the free field.

    The parameter is defined as

    .. math::

        L_\mathrm{J} =
        10 \log_{10}
        \frac{
            \displaystyle \int_{0.08}^{\infty} p_\mathrm{L}^2(t)\,\mathrm{d}t
        }{
            \displaystyle \int_{0}^{\infty} p_{10}^2(t)\,\mathrm{d}t
        }

    where :math:`p_\mathrm{L}(t)` is the lateral sound pressure measured with a
    figure-eight microphone whose zero axis is oriented towards the source,
    and :math:`p_{10}(t)` is the instantaneous sound pressure of the
    impulse response measured with an omnidirectional microphone
    at 10 m distance in the free field.

    Using the energy decay curves of the reference response
    :math:`e_{10}(t)` and the lateral response :math:`e_\mathrm{L}(t)`,
    the parameter can be computed efficiently as

    .. math::

        L_\mathrm{J} =
        10 \log_{10}
        \frac{
            e_\mathrm{L}(0.08)
        }{
            e_{10}(0)
        }.

    Parameters
    ----------
    energy_decay_curve_ref_10m : pyfar.TimeData
        Energy decay curve of the reference impulse response measured
        with an omnidirectional microphone at 10 m distance in the
        free field. The EDC must start at time zero.

    energy_decay_curve_lateral : pyfar.TimeData
        Energy decay curve of the room impulse response measured with a
        figure-eight microphone oriented according to [#isoLLJ]_
        (zero axis pointing towards the source). The EDC must start at
        time zero.

        Both EDCs must have identical ``signal.cshape``.

    Returns
    -------
    Late Lateral Sound Level : numpy.ndarray
        Late lateral sound level (:math:`L_\mathrm{J}`) in decibels,
        shaped according to the channel shape of the input EDCs.

    References
    ----------
    .. [#isoLLJ] ISO 3382-1, ISO 3382, Acoustics — Measurement of the
        reverberation time of rooms with reference to other acoustical
        parameters.
    """

    limits = np.array([0.0, np.inf, 0.08, np.inf])

    return 10 * np.log10(_energy_ratio(limits,
                                       energy_decay_curve_ref_10m,
                                       energy_decay_curve_lateral))

def sound_strength(energy_decay_curve_room,
             energy_decay_curve_free_field):
    r"""
    Calculate the room-acoustic strength parameter (:math:`G`).

    The strength parameter (:math:`G`) is defined as the ratio between the
    total arriving sound energy and the total arriving sound energy
    of a reference free-field response measured at 10 m with the same
    source. It is a measure of the room-induced level amplification
    at the receiver position [#isoStrength]_.

    The parameter is defined as

    .. math::

        G =
        10 \log_{10}
        \frac{
            \displaystyle \int_{0}^{\infty} p^2(t)\,dt
        }{
            \displaystyle \int_{0}^{\infty} p_\mathrm{10}^2(t)\,dt
        }

    where :math:`p(t)` is the room sound pressure and
    :math:`p_\mathrm{10}(t)` is the reference free-field sound pressure
    at 10 m measured with the same loudspeaker.

    Using the energy decay curves of the room response
    :math:`e(t)` and the reference response
    :math:`e_\mathrm{10}(t)`, the
    parameter can be computed efficiently as

    .. math::

        G =
        10 \log_{10}
        \frac{
            e(0) - e(\infty)
        }{
            e_\mathrm{10}(0) - e_\mathrm{10}(\infty)
        }.

    Parameters
    ----------
    energy_decay_curve_room : pyfar.TimeData
        Energy decay curve of the room impulse response. The EDC must
        start at time zero.

    energy_decay_curve_free_field : pyfar.TimeData
        Energy decay curve of the reference free-field impulse response
        at 10 m. The EDC must start at time zero.
        Both EDCs must have identical ``signal.cshape``.

    Returns
    -------
    strength : numpy.ndarray
        Strength parameter (:math:`G`) in decibels,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#isoStrength] ISO 3382, Acoustics — Measurement of the reverberation
        time of rooms with reference to other acoustical parameters.
    """

    limits = np.array([0.0, np.inf, 0.0, np.inf])

    return 10*np.log10(_energy_ratio(limits,
                                     energy_decay_curve_free_field,
                                     energy_decay_curve_room))
