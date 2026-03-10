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


def speech_transmission_index_indirect(
        rir, rir_type="acoustical", level=None, snr=None, ambient_noise=True):
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
    snr : numpy.ndarray or None, optional
        Signal-to-noise ratio in dB per octave band (125 Hz–8 kHz).
        Shape can be ``(7,)`` (7 octave bands: 125 Hz–8 kHz)
        to use the same values for all channels, or ``(rir.cshape, 7)``
        for channel-specific values.
        If ``None`` is provided, infinite SNR is assumed.
        See [#iec]_, section 3.
    ambient_noise: bool, optional
        Apply ambient noise correction according to [#iec]_,
        Annex A.2.3. Default is ``True``.

    Returns
    -------
    sti : numpy.ndarray
        Channel-wise Speech Transmission Index with shape ``rir.cshape``.

    Notes
    -----
    pyfar uses octave-band filters of order 14 and the filter order
    influences the MTF. Higher filter orders produce steeper roll-off and
    a more ideal band separation, which affects the energy distribution
    within each octave band and thus the computed modulation depth. However,
    the resulting STI is not meaningfully affected, since individual
    deviations in the MTF tend to cancel out in the weighted sum over
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

    if snr is not None:
        # check if input snr is a numpy array
        if not isinstance(snr, np.ndarray):
            raise TypeError("Input 'snr' must be a numpy array.")
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
    else:
        # default to infinite SNR for all octave bands
        snr = np.full((7,), np.inf)

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

    # Validate ambient_noise parameter
    if not isinstance(ambient_noise, bool):
        raise TypeError("ambient_noise must be a boolean.")

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
                                           ambient_noise)
        sti[ch] = _sti_calc(mtf)

    return sti


def modulation_transfer_function(
        rir, rir_type="acoustical", level=None, snr=None,
        ambient_noise=True):
    """
    Compute the modulation transfer function (MTF) of an impulse response
    according to IEC 60268-16:2020.

    The MTF describes the reduction of modulation depth caused by the
    transmission path. It is evaluated for 7 octave bands (125 Hz–8 kHz)
    and 14 modulation frequencies (0.63 Hz–12.5 Hz) and forms the basis
    of the Speech Transmission Index (STI).

    The calculation includes:

    - Energy-based MTF estimation from octave-band impulse responses
    - Limitation due to signal-to-noise ratio
    - Optional ambient noise correction (Annex A.2.3)
    - Optional auditory masking and absolute threshold effects for
      acoustical signals only (Annex A.2.4)

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
    snr : numpy.ndarray or None, optional
        Signal-to-noise ratio when the test source is turned off, in
        dB per octave band (125 Hz–8 kHz).
        Shape must be ``(7,)`` (7 octave bands: 125 Hz–8 kHz).
        If ``None`` is provided, an infinite SNR is assumed.
        Default is ``None``. See [#iecMTF]_, section 3.
    ambient_noise : bool, optional
        Apply ambient noise correction according to [#iecMTF]_,
        Annex A.2.3. Default is ``True``.

    Returns
    -------
    mtf : numpy.ndarray
        Modulation transfer function with shape ``(7, 14)``.

    Notes
    -----
    pyfar uses octave-band filters of order 14 and the filter order
    influences the MTF. Higher filter orders produce steeper roll-off and
    a more ideal band separation, which affects the energy distribution
    within each octave band and thus the computed modulation depth. However,
    the resulting STI is not meaningfully affected, since individual
    deviations in the MTF tend to cancel out in the weighted sum over
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

    if snr is not None:
        # check if input snr is a numpy array
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
    # set snr to infinity if not given
    else:
        snr = np.full((7,), np.inf)

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

    # Validate ambient_noise parameter
    if not isinstance(ambient_noise, bool):
        raise TypeError("ambient_noise must be a boolean.")

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

    # SNR correction (IEC 60268-16:2020, Eq. A.9)
    mtf *= 1 / (1 + 10 ** (-snr[:, None] / 10))

    if level is not None and ambient_noise:
        # Total intensity per octave band: signal + noise
        # (IEC 60268-16:2020, Annex A.2.4)
        Ik = 10 ** (level / 10) + 10 ** ((level - snr) / 10)
        if rir_type == "acoustical":
            # Compute level-dependent auditory masking factor I_amk
            # (IEC 60268-16:2020, Annex A.4.2)
            amdb = level.copy()
            amdb[amdb < 63] = 0.5*amdb[amdb < 63] - 65
            amdb[(63 <= amdb) & (amdb < 67)] = 1.8 * amdb[(63 <= amdb) &
                                                        (amdb < 67)] - 146.9
            amdb[(67 <= amdb) & (amdb < 100)] = 0.5 * amdb[(67 <= amdb) &
                                                        (amdb < 100)] - 59.8
            amdb[100 <= amdb] = amdb[100 <= amdb] - 10
            a = 10**(amdb/10)
            # Masking intensity (IEC 60268-16:2020, Eq. A.12)
            I_k1 = np.roll(Ik, 1)
            I_amk = I_k1 * a
            I_amk[0] = 0  # No masking for lowest band (125 Hz)
            # Absolute speech reception threshold
            # (IEC 60268-16:2020, Annex A.4.3)
            A_k = np.array([[46, 27, 12, 6.5, 7.5, 8, 12]]).T
            I_rt = 10 ** (A_k / 10)
            # Auditory masking + threshold correction
            # (IEC 60268-16:2020, Eq. A.11)
            mtf = (mtf * Ik[:, None]
                   / (Ik[:, None] + I_amk[:, None] + I_rt))
    # Clip MTF to valid range [0, 1] (IEC 60268-16:2020, Section A.2.4)
    return np.clip(mtf, 0, 1)


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
    TI = (snr_eff + 15) / 30

    # Modulation transmission index per octave band: average over
    # modulation frequencies (IEC 60268-16:2020, Annex A.2.1)
    mti = np.mean(TI, axis=-1)

    # STI Octave evaluation factors (IEC 60268-16:2020, Table A.1)
    alpha = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])

    # STI (IEC 60268-16:2020, Annex A.2.1)
    sti = np.sum(alpha * mti) - np.sum(beta * np.sqrt(mti[:-1] * mti[1:]))

    # limit STI to 1 (IEC 60268-16:2020, Section A.2.1)
    return min(sti, 1.0)


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
