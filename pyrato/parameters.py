"""
Sub-module implementing calculations of room acoustic parameters from
simulated or experimental data.
"""
import re
import numpy as np
import pyfar as pf
from matplotlib import pyplot as plt
from math import erfc
import warnings
from scipy.integrate import trapezoid

def reverberation_time_linear_regression(
        energy_decay_curve, T='T20', return_intercept=False):
    """Estimate the reverberation time from a given energy decay curve.

    The linear regression is performed using least squares error minimization
    according to the ISO standard 3382 [#]_.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve.
    T : 'T15', 'T20', 'T30', 'T40', 'T50', 'T60', 'EDT', 'LDT'
        Decay interval to be used for the reverberation time extrapolation. EDT
        corresponds to the early decay time extrapolated from the interval
        ``[0, -10]`` dB, LDT corresponds to the late decay time extrapolated
        from the interval ``[-25, -35]`` dB.
    return_intercept : bool
        If `True`, the function returns the intercept of the linear regression,
        which corresponds to the amplitude of the energy decay curve on a
        linear scale. The default is False.

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
        Early lateral energy fraction (:math:`J_\mathrm{LF}`),
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

def late_lateral_sound_level(energy_decay_curve_free_field,
                       energy_decay_curve_room_lateral):
    r"""
    Calculate the late lateral sound level.

    The late lateral sound level :math:`L_\mathrm{J}` quantifies the strength
    of late-arriving lateral sound energy. According to ISO 3382-1 [#isoLLJ]_,
    it is defined as the level ratio between the late lateral sound energy
    captured with a figure-of-eight microphone and the total sound energy of a
    reference impulse response measured with an omnidirectional microphone at
    a distance of 10 m in the free field. It is a measure of listener
    envelopment.

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
    energy_decay_curve_free_field : pyfar.TimeData
        Energy decay curve of the reference free field impulse response
        measured vwith an omnidirectional microphone at 10 m distance in the
        free field. The EDC must start at time zero.

    energy_decay_curve_room_lateral : pyfar.TimeData
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
                                       energy_decay_curve_free_field,
                                       energy_decay_curve_room_lateral))

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


def mixing_time_abel(rir, window_length=1024, onset_threshold_db=-30,
                     peak_secure_margin=100, plot=False):
    r"""Estimate the mixing time from the echo density after Abel and Huang.

    The direct-sound onset is detected per channel as the first sample
    exceeding `onset_threshold_db` below the peak, and the impulse response
    is cut `peak_secure_margin` samples before the onset, or at its start if
    fewer samples precede the onset. Multichannel input is cut at the earliest
    onset across channels, which preserves inter-channel time differences: a
    channel is then measured from the first arrival across all channels, so
    its mixing time includes its propagation delay relative to the earliest
    channel. Pass channels individually to measure each from its own onset.

    The echo density counts the samples with :math:`|h| > \sigma` inside a
    rectangular window (growing at the start, of length `window_length` in
    the middle, shrinking at the end), divided by `window_length` in every
    case and by the Gaussian expectation
    :math:`\mathrm{erfc}(1/\sqrt{2}) \approx 0.317`. Here :math:`\sigma` is
    the sample standard deviation of the window. The mixing time is

    .. math::

        t_\mathrm{Abel} = (d - m) / f_s,

    where :math:`d` is the first sample at which the echo density exceeds
    unity, i.e., criterion I of Abel and Huang [#abel]_, and :math:`m` the
    number of samples kept before the onset.

    Parameters
    ----------
    rir : pyfar.Signal
        Room impulse response.
    window_length : int, optional
        Length of the sliding window in samples. Must be even. The default is
        ``1024``, which corresponds to about 21 ms at 48 kHz, the window
        length recommended by Abel and Huang [#abel]_. Scale it for other
        sampling rates to keep the window length in time.
    onset_threshold_db : float, optional
        Peak criterion for onset detection, in dB below the peak. The default
        is ``-30``.
    peak_secure_margin : int, optional
        Number of samples kept before the detected onset, so the echo
        density window covers the direct sound. The mixing time is measured
        from the onset regardless of the margin. The default is ``100``.
    plot : bool, optional
        If ``True``, plot the impulse response with the detected onset and
        the echo density with the mixing time. Only possible for single
        channel input. The default is ``False``.

    Returns
    -------
    mixing_time : numpy.ndarray
        Mixing time in seconds relative to the detected onset, of shape
        ``rir.cshape``. ``numpy.nan`` with a warning for channels whose echo
        density never exceeds unity.
    echo_density : pyfar.TimeData
        Normalized echo density of shape ``rir.cshape``, starting at the
        sample where the impulse response was cut. The times refer to the
        time axis of ``rir``.
    onset : float
        Detected onset in seconds on the time axis of ``rir``, the earliest
        across channels for multichannel input.

    References
    ----------
    .. [#abel] J. S. Abel and P. Huang, "A simple, robust measure of
        reverberation echo density," in Proc. of the 121st AES Convention,
        San Francisco, 2006.

    Examples
    --------
    .. plot::

        >>> import pyfar as pf
        >>> import pyrato as ra
        ...
        >>> rir = pf.signals.files.room_impulse_response(sampling_rate=48000)
        >>> t_abel, echo_density, onset = ra.parameters.mixing_time_abel(
        ...     rir, plot=True)
    """
    if not isinstance(rir, pf.Signal):
        raise TypeError("Input data must be a pyfar.Signal.")
    if (not isinstance(window_length, (int, np.integer))
            or window_length <= 0 or window_length % 2):
        raise ValueError(
            f"window_length is {window_length} but must be a positive even "
            "integer.")
    if onset_threshold_db >= 0:
        raise ValueError(
            f"onset_threshold_db is {onset_threshold_db} but must be "
            "negative.")
    if (not isinstance(peak_secure_margin, (int, np.integer))
            or peak_secure_margin < 0):
        raise ValueError(
            f"peak_secure_margin is {peak_secure_margin} but must be a "
            "non-negative integer.")
    if plot and np.prod(rir.cshape) != 1:
        raise ValueError("The plot can only be done for single channel input.")

    fs = rir.sampling_rate
    half = window_length // 2

    peaks = np.max(np.abs(rir.time), axis=-1)
    if np.any(peaks == 0):
        raise ValueError(
            "Onset detection failed. At least one channel contained only "
            "zeros.")
    threshold = 10 ** (onset_threshold_db / 20)
    # A negative threshold lets the peak sample clear the criterion, so
    # argmax always finds an onset.
    above = np.abs(rir.time) > peaks[..., np.newaxis] * threshold
    onset = int(np.argmax(above, axis=-1).min())
    start = max(0, onset - peak_secure_margin)
    margin = onset - start
    data = rir.time[..., start:]
    n_samples = data.shape[-1]
    if n_samples < window_length:
        raise ValueError(
            f"The impulse response is {n_samples} samples long after onset "
            "detection, which is shorter than the analysis window length of "
            f"{window_length} samples.")

    # Gaussian noise has a fraction erfc(1/sqrt(2)) of samples beyond sigma.
    norm = window_length * erfc(1 / np.sqrt(2))
    # NaN padding gives the growing and shrinking windows at both ends the
    # full length; nanstd and the comparison ignore it.
    nan_pad = np.full(half, np.nan)
    # windows are processed in blocks to bound the memory for long responses
    block_size = 4096

    mixing_time = np.empty(rir.cshape)
    echo_density = np.empty(data.shape)
    for ch in np.ndindex(rir.cshape):
        windows = np.lib.stride_tricks.sliding_window_view(
            np.concatenate((nan_pad, data[ch], nan_pad)), window_length)
        for i0 in range(0, n_samples, block_size):
            block = windows[i0:min(i0 + block_size, n_samples)]
            sigma = np.nanstd(block, axis=-1, ddof=1)
            echo_density[ch][i0:i0 + block.shape[0]] = np.sum(
                np.abs(block) > sigma[:, np.newaxis], axis=-1) / norm

        exceeds = echo_density[ch] > 1
        if np.any(exceeds):
            # 1-based index of the first window exceeding unity
            d = np.argmax(exceeds) + 1
            mixing_time[ch] = (d - margin) / fs
        else:
            mixing_time[ch] = np.nan

    if np.any(np.isnan(mixing_time)):
        warnings.warn(
            "The echo density never exceeds unity in "
            f"{np.count_nonzero(np.isnan(mixing_time))} channel(s). NaN is "
            "returned for these channels.", stacklevel=2)

    times = (start + np.arange(n_samples)) / fs
    echo_density = pf.TimeData(echo_density, times)
    onset = onset / fs

    if plot:
        _plot_mixing_time(
            rir, onset, echo_density, mixing_time, r'$t_\mathrm{Abel}$')

    return mixing_time, echo_density, onset


def perceptual_mixing_time_lindau(
        rir, percentile=50, window_length=1024, onset_threshold_db=-30,
        peak_secure_margin=100, plot=False):
    r"""Predict the perceptual mixing time after Lindau et al.

    The perceptual mixing time is predicted from the mixing time
    :math:`t_\mathrm{Abel}` of :py:func:`mixing_time_abel` by the
    regressions of Lindau et al. [#lindau]_

    .. math::

        t_\mathrm{mp50} = 0.8 \, t_\mathrm{Abel} - 8\,\mathrm{ms},
        \quad
        t_\mathrm{mp95} = 1.77 \, t_\mathrm{Abel} - 38\,\mathrm{ms},

    where :math:`t_\mathrm{mp50}` is the average perceptual mixing time and
    :math:`t_\mathrm{mp95}` the 95 % point of the distribution of perceptual
    mixing times across listeners, i.e., a transition into a static tail that
    is inaudible to at least half, respectively nearly all, of the expert
    listeners in the study. The slope of 1.77 is the unrounded fit; the
    paper prints 1.8.

    Parameters
    ----------
    rir : pyfar.Signal
        Room impulse response.
    percentile : {50, 95}, optional
        Percentile of the perceptual mixing time distribution predicted by
        the regression. The default is ``50``.
    window_length : int, optional
        See :py:func:`mixing_time_abel`.
    onset_threshold_db : float, optional
        See :py:func:`mixing_time_abel`.
    peak_secure_margin : int, optional
        See :py:func:`mixing_time_abel`.
    plot : bool, optional
        If ``True``, plot the impulse response with the detected onset and
        the echo density with the perceptual mixing time. Only possible for
        single channel input. The default is ``False``.

    Returns
    -------
    perceptual_mixing_time : numpy.ndarray
        Perceptual mixing time in seconds relative to the detected onset, of
        shape ``rir.cshape``. ``numpy.nan`` with a warning where the
        regression predicts a negative value or no mixing time is found.

    References
    ----------
    .. [#lindau] A. Lindau, L. Kosanke, and S. Weinzierl, "Perceptual
        evaluation of model- and signal-based predictors of the mixing time
        in binaural room impulse responses," J. Audio Eng. Soc., vol. 60,
        no. 11, pp. 887-898, 2012.

    Examples
    --------
    .. plot::

        >>> import pyfar as pf
        >>> import pyrato as ra
        ...
        >>> rir = pf.signals.files.room_impulse_response(sampling_rate=48000)
        >>> t_mp95 = ra.parameters.perceptual_mixing_time_lindau(
        ...     rir, percentile=95, plot=True)
    """
    if percentile == 50:
        slope, intercept = 0.8, -0.008
    elif percentile == 95:
        slope, intercept = 1.77, -0.038
    else:
        raise ValueError(
            f"percentile is {percentile} but must be 50 or 95.")
    if not isinstance(rir, pf.Signal):
        raise TypeError("Input data must be a pyfar.Signal.")
    if plot and np.prod(rir.cshape) != 1:
        raise ValueError("The plot can only be done for single channel input.")

    t_abel, echo_density, onset = mixing_time_abel(
        rir, window_length, onset_threshold_db, peak_secure_margin)

    perceptual_mixing_time = slope * t_abel + intercept
    negative = perceptual_mixing_time < 0
    if np.any(negative):
        warnings.warn(
            f"The predicted t_mp{percentile} is negative in "
            f"{np.count_nonzero(negative)} channel(s), because t_Abel is "
            "below the range of the regression. NaN is returned for these "
            "channels.", stacklevel=2)
        perceptual_mixing_time[negative] = np.nan

    if plot:
        _plot_mixing_time(
            rir, onset, echo_density, perceptual_mixing_time,
            rf'$t_\mathrm{{mp{percentile}}}$')

    return perceptual_mixing_time


def _plot_mixing_time(rir, onset, echo_density, mixing_time, label):
    """Plot the impulse response and the echo density in a 2 by 1 layout."""
    mixing_time = mixing_time.flat[0]

    with pf.plot.context():
        fig, ax = plt.subplots(2, 1, sharex=True)
        # unit is fixed because the markers below are placed in seconds
        pf.plot.time(rir, dB=True, unit='s', ax=ax[0], label='RIR')
        ax[0].axvline(onset, color='k', linestyle='--', label='Onset')
        pf.plot.time(echo_density, unit='s', ax=ax[1], label='Echo density')
        ax[1].axhline(1, color='k', linestyle=':')
        ax[1].set_ylabel('Normalized echo density')
        if not np.isnan(mixing_time):
            for axis in ax:
                axis.axvline(
                    onset + mixing_time, color=pf.plot.color('r'),
                    label=label)
            ax[1].set_xlim(
                max(rir.times[0], onset - 0.01),
                onset + max(3 * mixing_time, 0.05))
        ax[0].set_xlabel(None)
        ax[0].legend()
        ax[1].legend()
        fig.align_ylabels()


def speech_transmission_index_indirect(
        rir, rir_type="acoustical", level=None, snr=np.inf,
        ambient_noise_correction=True):
    """
    Computes the Speech Transmission Index (STI) according to
    IEC 60268-16:2020 using the indirect method.

    The STI is a scalar measure between 0 (bad) and 1 (excellent)
    describing speech intelligibility. It is computed from the
    :py:func:`~modulation_transfer_function`, optionally including auditory
    masking and ambient noise effects.

    STI considers 7 octave bands from 125 Hz to 8 kHz
    and 14 modulation frequencies between 0.63 Hz and
    12.5 Hz [#iec]_.

    Parameters
    ----------
    rir : pyfar.Signal
        Single or multi-channel room impulse response for which the
        STI is computed. The room impulse response must be at least
        1.6 seconds long. See [#iec]_, Section 6.2.
    rir_type : ``'electrical'``, ``'acoustical'``
        Determines whether input signals given by `rir` were obtained
        acoustically or electrically. Default is ``'acoustical'``.
        Auditory masking effects are only applied for acoustical
        signals [#iec]_, section A.3.1.
    level : numpy.ndarray or None, optional
        Test signal level without noise in dB SPL, given per octave band
        (125 Hz–8 kHz). Shape can be ``(7,)`` (7 octave bands: 125 Hz–8 kHz)
        to use the same values for all channels, or ``(rir.cshape, 7)``
        for channel-specific values.
        If ``None`` is provided, auditory and ambient noise corrections are
        omitted. See [#iec]_, section A.3.2.
    snr : numpy.ndarray or float, optional
        Signal-to-noise ratio in dB for each octave band (125 Hz–8 kHz).
        Shape can be ``(7,)`` (7 octave bands: 125 Hz–8 kHz)
        to use the same values for all channels, or ``(rir.cshape, 7)``
        for channel-specific values.
        Default is ``np.inf`` (no ambient noise). See [#iec]_, section 3.
    ambient_noise_correction: bool, optional
        Apply ambient noise correction according to [#iec]_,
        Annex A.2.3. Default is ``True``. Only applied when
        ``level is not None`` and ``ambient_noise_correction is True``.

    Returns
    -------
    sti : numpy.ndarray
        Channel-wise Speech Transmission Index with shape ``rir.cshape``.

    Notes
    -----
    pyfar uses octave-band filters of order 14 and the filter order
    influences the MTF. Higher filter orders produce steeper roll-off and
    a more ideal band separation, which affects the energy distribution
    within each octave band and thus the computed modulation depth.
    The influence on the broadband STI is negligible, since individual
    deviations in the MTF tend to average out in the weighted sum over
    octave bands and modulation frequencies.

    References
    ----------
    .. [#iec] IEC 60268-16:2020
     Sound system equipment - Part 16: Objective rating of speech
     intelligibility by speech transmission index.
    """
    # check if input data is a pyfar.Signal
    if not isinstance(rir, pf.Signal):
        raise TypeError("Input data must be a pyfar.Signal.")

    if np.ndim(snr) == 0:
        snr = np.full((7,), float(snr))
    snr = np.asarray(snr)
    # Check if SNR has valid shape: (7,) uses same values for all
    # channels, (cshape, 7) for channel-specific values
    if snr.shape != (7,) and snr.shape != rir.cshape + (7,):
        raise ValueError(
            f"Input 'snr' must have shape (7,) or "
            f"{rir.cshape + (7,)} (7 octave bands or matching "
            f"rir channels + 7 octave bands), but got {snr.shape}.",
        )
    if np.any(snr < 20):
        warnings.warn(
            "Input 'snr' should be at least 20 dB for every octave band.",
            stacklevel=2,
            )

    if level is not None:
        # check if input level is a numpy array
        if not isinstance(level, np.ndarray):
            raise TypeError("Input 'level' must be a numpy array.")
        level = np.asarray(level)
        # Check if level has valid shape: (7,) uses same values for
        # all channels, (cshape, 7) for channel-specific values
        if level.shape != (7,) and level.shape != rir.cshape + (7,):
            raise ValueError(
                f"Input 'level' must have shape (7,) or "
                f"{rir.cshape + (7,)} (7 octave bands or matching "
                f"rir channels + 7 octave bands), but got {level.shape}.",
            )

    # check rir_type
    if rir_type is None:
        rir_type = "acoustical"
    if rir_type not in ["electrical", "acoustical"]:
        raise ValueError(f"rir_type is '{rir_type}' but must be "
                         "'electrical' or 'acoustical'.")

    # Validate ambient_noise_correction parameter
    if not isinstance(ambient_noise_correction, bool):
        raise TypeError("ambient_noise_correction must be a boolean.")

    sti = np.zeros(rir.cshape)

    for ch in np.ndindex(rir.cshape):
        # Use snr/level directly if shape is (7,), otherwise index by channel
        current_snr = snr if snr.shape == (7,) else snr[ch]
        current_level = (
            None if level is None
            else (level if level.shape == (7,) else level[ch])
        )
        mtf = modulation_transfer_function(rir[ch],
                                           rir_type,
                                           current_level,
                                           current_snr,
                                           ambient_noise_correction)
        sti[ch] = _sti_calc(mtf)

    return sti


def modulation_transfer_function(
        rir, rir_type="acoustical", level=None, snr=np.inf,
        ambient_noise_correction=True):
    """
    Compute the modulation transfer function (MTF) of an impulse response
    according to IEC 60268-16:2020.

    The MTF describes the reduction of modulation depth caused by the
    transmission path. It is evaluated for 7 octave bands (125 Hz–8 kHz)
    and 14 modulation frequencies (0.63 Hz–12.5 Hz) and forms the basis
    of the Speech Transmission Index (STI).

    Parameters
    ----------
    rir : pyfar.Signal
        Single-channel room impulse response with ``rir.cshape = (1, )``.
        The room impulse response must be at least 1.6 seconds long.
    rir_type : {'electrical', 'acoustical'}, optional
        Determines whether input signals given by `rir` were obtained
        acoustically or electrically. Default is ``'acoustical'``.
        Auditory masking effects are only applied for acoustical
        signals [#iecMTF]_, section A.3.1.
    level : numpy.ndarray or None, optional
        Test signal level without noise in dB SPL, given per octave band
        (125 Hz–8 kHz). Shape must be ``(7,)`` (7 octave bands: 125 Hz–8 kHz).
        If ``None`` is provided, auditory and ambient noise corrections are
        omitted. Default is ``None``. See [#iecMTF]_, section A.3.2.
    snr : numpy.ndarray or float, optional
        Signal-to-noise ratio in dB for each octave band (125 Hz–8 kHz).
        Shape must be ``(7,)`` (7 octave bands: 125 Hz–8 kHz).
        Default is ``np.inf`` (no ambient noise). See [#iecMTF]_, section 3.
    ambient_noise_correction : bool, optional
        Apply ambient noise correction according to [#iecMTF]_,
        Annex A.2.3. Default is ``True``. Only applied when
        ``level is not None`` and ``ambient_noise_correction is True``.

    Returns
    -------
    mtf : numpy.ndarray
        Modulation transfer function with shape ``(7, 14)``.

    Notes
    -----
    pyfar uses octave-band filters of order 14 and the filter order
    influences the MTF. Higher filter orders produce steeper roll-off and
    a more ideal band separation, which affects the energy distribution
    within each octave band and thus the computed modulation depth.
    The influence on the broadband STI is negligible, since individual
    deviations in the MTF tend to average out in the weighted sum over
    octave bands and modulation frequencies.

    References
    ----------
    .. [#iecMTF] IEC 60268-16:2020
       Sound system equipment - Part 16: Objective rating of speech
       intelligibility by speech transmission index.
    """
    # Check if input data is a pyfar.Signal
    if not isinstance(rir, pf.Signal):
        raise TypeError("Input data must be a pyfar.Signal.")

    # Check if data is single-channel
    if rir.cshape != (1,):
        raise ValueError(
            "Input must be a single-channel impulse response, "
            f"but got shape {rir.cshape}.",
        )

    # Check if signal is at least 1.6 seconds long
    # (IEC 60268-16:2020, Section 6.2)
    if rir.n_samples / rir.sampling_rate < 1.6:
        raise ValueError(
            "Input signal must be at least 1.6 seconds long "
            "(see IEC 60268-16:2020, Section 6.2).",
        )

    if np.ndim(snr) == 0:
        snr = np.full((7,), float(snr))
    if not isinstance(snr, np.ndarray):
        raise TypeError("snr must be a numpy array.")
    if snr.shape != (7,):
        raise ValueError(
            f"snr must have shape (7,) for 7 octave bands "
            f"(125 Hz - 8 kHz), but got {snr.shape}.",
        )
    if np.any(snr < 20):
        warnings.warn(
            "Input 'snr' should be at least 20 dB for every "
            "octave band.",
            stacklevel=2,
        )

    if level is not None:
        # check if input level is a numpy array
        if not isinstance(level, np.ndarray):
            raise TypeError("level must be a numpy array or None.")
        if level.shape != (7,):
            raise ValueError(
                f"Level must have shape (7,) for 7 octave bands "
                f"(125 Hz - 8 kHz), but got {level.shape}.",
            )
        if np.any(level < 1):
            warnings.warn(
                "Input 'level' should be at least 1 dB for every "
                "octave band.",
                stacklevel=2,
            )

    # check rir_type
    if rir_type is None:
        rir_type = "acoustical"
    if rir_type not in ["electrical", "acoustical"]:
        raise ValueError(f"Data_type is '{rir_type}' but must be "
                         "'electrical' or 'acoustical'.")

    # Validate ambient_noise_correction parameter
    if not isinstance(ambient_noise_correction, bool):
        raise TypeError("ambient_noise_correction must be a boolean.")

    rir_oct = pf.dsp.filter.fractional_octave_bands(
        rir, num_fractions=1, frequency_range=(125, 8000))

    # Modulation frequencies (fm) in Hz for MTF calculation
    # Defined in IEC 60268-16:2020, Section A.1.4
    modulation_frequencies = np.array(
        [0.63, 0.80, 1.0, 1.25, 1.60, 2.0, 2.5,
         3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5],
    )

    modulation_frequencies = np.tile(
        modulation_frequencies, (rir_oct.cshape[0], 1))
    energy = rir_oct.time ** 2

    # MTF calculation (IEC 60268-16:2020, Section 6.1)
    term_exp = np.exp(
        -2j * np.pi * modulation_frequencies[:, :, None] * rir_oct.times)
    numerator = np.abs(np.sum(energy * term_exp, axis=-1))
    denominator = np.sum(energy, axis=-1)
    mtf = numerator / denominator

    # Ambient noise correction (IEC 60268-16:2020, Annex A.2.3)
    mtf *= 1 / (1 + 10 ** (-snr[:, None] / 10))

    if level is not None and ambient_noise_correction is True:
        mtf = _ambient_noise_correction(mtf, level, snr, rir_type)

    # Clip MTF to valid range [0, 1] (IEC 60268-16:2020, Section A.2.4)
    return np.clip(mtf, 0, 1)


def _ambient_noise_correction(mtf, level, snr, rir_type):
    r"""
    Weight the MTF by auditory masking, ambient noise, and hearing threshold.

    The basic MTF (after SNR correction) only accounts for reverberation and
    the signal-to-noise ratio. This function applies an additional perceptual
    weighting defined in [#iecANC]_, Annex A.2.4:

    .. math::

        m''(f_m, k) = m'(f_m, k) \cdot
        \frac{I_k}{I_k + I_{\mathrm{am},k} + I_{\mathrm{rt},k}}

    where :math:`f_m` is the modulation frequency, :math:`k` is the octave band
    index, :math:`m'(f_m, k)` is the MTF after the ambient noise correction
    (Annex A.2.3), and the three intensity terms are:

    - :math:`I_k = I_{z,k} + I_{n,k}` — total acoustic intensity level
      for octave band :math:`k`, i.e. test signal (:math:`I_{z,k}`) plus
      background noise (:math:`I_{n,k}`).
    - :math:`I_{\mathrm{am},k}` — masking intensity due to the
      level-dependent auditory masking effect acting on octave band :math:`k`
      from band :math:`k-1`, according to [#iecANC]_, Annex A.4.2
      (acoustical signals only).
    - :math:`I_{\mathrm{rt},k}` — acoustic intensity level of the
      reception threshold for octave band :math:`k`, according to
      [#iecANC]_, Annex A.4.3 (acoustical signals only).

    For electrical signals the auditory terms are zero and this correction
    has no effect.

    Parameters
    ----------
    mtf : numpy.ndarray
        Modulation transfer function after SNR correction, shape ``(7, 14)``.
    level : numpy.ndarray
        Test signal level without noise in dB SPL per octave band,
        shape ``(7,)``.
    snr : numpy.ndarray
        Signal-to-noise ratio in dB per octave band, shape ``(7,)``.
    rir_type : {'electrical', 'acoustical'}
        Auditory masking and hearing threshold corrections are only applied
        for ``'acoustical'`` signals.

    Returns
    -------
    mtf : numpy.ndarray
        Corrected modulation transfer function, shape ``(7, 14)``.

    References
    ----------
    .. [#iecANC] IEC 60268-16:2020
       Sound system equipment - Part 16: Objective rating of speech
       intelligibility by speech transmission index.
    """
    # Total intensity per octave band: signal + noise
    # (IEC 60268-16:2020, Annex A.2.4)
    Ik = 10 ** (level / 10) + 10 ** ((level - snr) / 10)

    # Auditory masking and reception threshold terms are zero for
    # electrical signals (IEC 60268-16:2020, Annex A.4.2, A.4.3)
    I_amk = np.zeros(7)
    I_rt = np.zeros((7, 1))

    if rir_type == "acoustical":
        # Level-dependent auditory masking factor
        # (IEC 60268-16:2020, Annex A.4.2)
        amdb = level.copy()
        amdb[amdb < 63] = 0.5*amdb[amdb < 63] - 65
        amdb[(63 <= amdb) & (amdb < 67)] = 1.8 * amdb[(63 <= amdb) &
                                                    (amdb < 67)] - 146.9
        amdb[(67 <= amdb) & (amdb < 100)] = 0.5 * amdb[(67 <= amdb) &
                                                    (amdb < 100)] - 59.8
        amdb[100 <= amdb] = amdb[100 <= amdb] - 10
        a = 10**(amdb/10)
        # Masking intensity (IEC 60268-16:2020, Annex A.4.2)
        I_k1 = np.roll(Ik, 1)
        I_amk = I_k1 * a
        I_amk[0] = 0  # No masking for lowest band (125 Hz)
        # Absolute speech reception threshold
        # (IEC 60268-16:2020, Annex A.4.3)
        A_k = np.array([[46, 27, 12, 6.5, 7.5, 8, 12]]).T
        I_rt = 10 ** (A_k / 10)

    # Auditory masking + threshold correction (IEC 60268-16:2020, Annex A.2.4)
    mtf = mtf * Ik[:, None] / (Ik[:, None] + I_amk[:, None] + I_rt)
    return mtf


def _sti_calc(mtf):
    """
    Computes the Speech Transmission Index (STI) from the MTF.

    Parameters
    ----------
    mtf : numpy.ndarray
        Modulation transfer function with shape ``(7, 14)`` for which
        the STI is computed.

    Returns
    -------
    sti : float
        Speech Transmission Index.
    """
    # Check if mtf is a numpy array
    if not isinstance(mtf, np.ndarray):
        raise TypeError("mtf must be a numpy array.")

    # Check if mtf has the correct shape
    if mtf.shape != (7, 14):
        raise ValueError(
            f"mtf must have shape (7, 14) for 7 octave bands and "
            f"14 modulation frequencies, but got {mtf.shape}.",
        )

    # Effective SNR from MTF (IEC 60268-16:2020, Annex A.2.1)
    with np.errstate(divide='ignore'):
        snr_eff = 10 * np.log10(mtf / (1 - mtf))

    # Clip SNR to [-15, 15] dB range (IEC 60268-16:2020, Annex A.2.1)
    snr_eff = np.clip(snr_eff, -15, 15)

    # Transmission index (TI) for each octave band and modulation
    # frequency (IEC 60268-16:2020, Annex A.2.1)
    ti = (snr_eff + 15) / 30

    # Modulation transmission index per octave band: average over
    # modulation frequencies (IEC 60268-16:2020, Annex A.2.1)
    mti = np.mean(ti, axis=-1)

    # STI Octave evaluation factors (IEC 60268-16:2020, Table A.1)
    alpha = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])

    # STI (IEC 60268-16:2020, Annex A.2.1)
    sti = np.sum(alpha * mti) - np.sum(beta * np.sqrt(mti[:-1] * mti[1:]))

    # limit STI to 1 (IEC 60268-16:2020, Section A.2.1)
    return min(sti, 1.0)


def center_time(energy_decay_curve):
    r"""
    Calculate the center time (:math:`T_s`) from the energy decay curve.

    The center time :math:`T_s` is the time of the centroid of the squared
    impulse response. It quantifies the balance between early and late
    sound energy [#isoTs]_.

    The parameter is defined as

    .. math::

        T_s =
        \frac{
            \displaystyle \int_{0}^{\infty} t \cdot p^2(t)\,\mathrm{d}t
        }{
            \displaystyle \int_{0}^{\infty} p^2(t)\,\mathrm{d}t
        }

    where :math:`p(t)` is the room impulse response sound pressure.

    Using the energy decay curve :math:`e(t)`, the parameter can be
    computed efficiently via the Schroeder backward integral identity
    :math:`e(t) = \int_t^\infty p^2(\tau)\,\mathrm{d}\tau` [#schroeder]_
    and Fubini's theorem as

    .. math::

        T_s =
        \frac{
            \displaystyle \int_{0}^{\infty} e(t)\,\mathrm{d}t
        }{
            e(0)
        }.

    Parameters
    ----------
    energy_decay_curve : pyfar.TimeData
        Energy decay curve of the room impulse response.

    Returns
    -------
    center_time : numpy.ndarray
        Center time (:math:`T_s`) in seconds,
        shaped according to the channel shape of the input EDC.

    References
    ----------
    .. [#isoTs] ISO 3382, Acoustics — Measurement of the reverberation
        time of rooms with reference to other acoustical parameters.
    .. [#schroeder] M. R. Schroeder, "New Method of Measuring Reverberation
        Time," J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, 1965.
        doi:10.1121/1.1909343
    """

    if not isinstance(energy_decay_curve, pf.TimeData):
        raise TypeError(
            "energy_decay_curve must be a pyfar.TimeData or derived object.")

    if not np.isclose(energy_decay_curve.times[0], 0.0):
        raise ValueError("energy_decay_curve must start at time zero.")

    if np.any(energy_decay_curve.time[..., 0] == 0):
        raise ValueError(
            "Initial energy of energy_decay_curve must not be zero.")

    initial_energy = energy_decay_curve.time[..., 0]
    # NaN values in the tail (e.g. from Lundeby/Chu truncation) are treated as
    # zero, since the EDC is assumed to converge to zero.
    edc = np.nan_to_num(energy_decay_curve.time, nan=0.0)
    center_time = (
        trapezoid(edc, energy_decay_curve.times, axis=-1)
        / initial_energy
    )

    return center_time
