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


def speech_transmission_index_indirect(
        rir, rir_type="acoustical", level=None, snr=None, ambient_noise=True):
    """
    Computes the Speech Transmission Index (STI) according to IEC 60268-16:2020 
    using the indirect method.

    The STI is a scalar measure between 0 (bad) and 1 (excellent) describing
    speech intelligibility. It is computed from the py:func:`~modulation_transfer_function`, 
    including auditory masking and ambient noise effects.

    STI considers 7 octave bands from 125 Hz to 8 kHz
    and 14 modulation frequencies between 0.63 Hz and
    12.5 Hz [#iec]_.

    Parameters
    ----------
    rir : pyfar.Signal
        Single or multi-channel room impulse response for which the STI is computed.
        The room impulse response must be at least 1.6 seconds long.
        See [#iec]_, Section 6.2.

    rir_type : ``'electrical'``, ``'acoustical'``
        Determines whether input signals given by `rir` were obtained acoustically or electrically. 
        Default is ``'acoustical'``.
        Auditory masking effects are only applied for acoustical
        signals [#iec]_, section A.3.1.

    level : numpy.ndarray or None, optional
        Test signal level without noise in dB SPL, given per octave band
        (125 Hz–8 kHz). Shape can be ``(7,)`` to use the same values for all
        channels, or ``(rir.cshape, 7)`` for channel-specific values.
        If ``None`` is provided, auditory and ambient noise corrections are
        omitted. See [#iec]_, section A.3.2.

    snr : numpy.ndarray or None, optional
        Signal-to-noise ratio in dB per octave band (125 Hz–8 kHz).
        Shape can be ``(7,)`` to use the same values for all channels,
        or ``(rir.cshape, 7)`` for channel-specific values.
        If ``None`` is provided, infinite snr is assumed.
        See [#iec]_, section 3.

    ambient_noise: bool, optional
        Apply ambient noise correction according to [#iec]_,
        Annex A.2.3. Default is ``True``.

    Returns
    -------
    sti : np.ndarray
        Channel-wise Speech Transmission Index with shape ``rir.cshape``.

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
            raise ValueError("Input 'snr' must be a numpy array.")
        snr = np.asarray(snr)
        # Check if SNR has valid shape: (7,) uses same values for all channels, 
        # (cshape, 7) for channel-specific values
        if snr.shape != (7,) and snr.shape != rir.cshape + (7,):
            raise ValueError(
                f"Input 'snr' must have shape (7,) or {rir.cshape + (7,)} "
                f"(7 octave bands or matching rir channels + 7 octave bands), but got {snr.shape}."
            )
        if np.any(snr < 20):
            warnings.warn(
                "Input 'snr' should be at least 20 dB for every octave band.",
                stacklevel=1,
                )
    else:
        # default to infinite SNR for all octave bands
        snr = np.full((7,), np.inf)
    

    if level is not None:
        # check if input level is a numpy array
        if not isinstance(level, np.ndarray):
            raise ValueError("Input 'level' must be a numpy array.")
        level = np.asarray(level)
        # Check if level has valid shape: (7,) uses same values for all channels, 
        # (cshape, 7) for channel-specific values
        if level.shape != (7,) and level.shape != rir.cshape + (7,):
            raise ValueError(
                f"Input 'level' must have shape (7,) or {rir.cshape + (7,)} "
                f"(7 octave bands or matching rir channels + 7 octave bands), but got {level.shape}."
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

    sti = np.zeros(rir.cshape)

    for ch in np.ndindex(rir.cshape):
        # Use snr/level directly if shape is (7,), otherwise index by channel
        current_snr = snr if snr.shape == (7,) else snr[ch]
        current_level = None if level is None else (level if level.shape == (7,) else level[ch])
        mtf = modulation_transfer_function(rir[ch],
                                           rir_type,
                                           current_level,
                                           current_snr,
                                           ambient_noise)
        sti[ch] = _sti_calc(mtf)

    return sti


def modulation_transfer_function(rir, rir_type, level, snr, ambient_noise):
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

    rir_type : {'electrical', 'acoustical'}
        Determines whether input signals given by `rir` were obtained acoustically or electrically. 
        Default is ``'acoustical'``.
        Auditory masking effects are only applied for acoustical
        signals [#iec]_, section A.3.1.

    level : numpy.ndarray or None
        Test signal level without noise in dB SPL, given per octave band
        (125 Hz–8 kHz). Shape must be ``(7,)``.
        If ``None`` is provided, auditory and ambient noise corrections are
        omitted. See [#iec]_, section A.3.2.

    snr : numpy.ndarray
        Signal-to-noise ratio in when the test source is turned off in
        dB per octave band (125 Hz–8 kHz).
        Shape must be ``(7,)``. If ``None`` is provided, infinite
        snr assumed. See [#iec]_, section 3.

    ambient_noise : bool
        Apply ambient noise correction according to [#iec]_,
        Annex A.2.3. Default is ``True``.

    Returns
    -------
    mtf : numpy.ndarray
        Modulation transfer function with shape ``(7, 14)``.
        
    References
    ----------
    .. [#] IEC 60268-16:2020
       Sound system equipment - Part 16: Objective rating of speech
       intelligibility by speech transmission index.
    """
    # Check if input data is a pyfar.Signal
    if not isinstance(rir, pf.Signal):
        raise TypeError("Input data must be a pyfar.Signal.")
    
    # Check if data is single-channel
    if rir.cshape != (1,):
        raise ValueError(
            f"Input must be a single-channel impulse response, but got shape {rir.cshape}."
        )
    
    # Check if the signal is at least 1.6 seconds long (IEC 60268-16:2020, Section 6.2)
    if not rir.n_samples / rir.sampling_rate >= 1.6:
        raise ValueError(
            "Input signal must be at least 1.6 seconds long (see IEC 60268-16:2020, Section 6.2)."
        )
        
    if snr is not None: 
        # check if input snr is a numpy array
        if not isinstance(snr, np.ndarray):
            raise TypeError("snr must be a numpy array.")
        if snr.shape != (7,):
            raise ValueError(f"snr must have shape (7,) for 7 octave bands (125 Hz - 8 kHz), but got {snr.shape}.")
        if np.any(snr < 20):
                warnings.warn(
                    "Input 'snr' should be at least 20 dB for every octave band.",
                    stacklevel=1,
                    )
    # set snr to infinity if not given
    else:
        snr = np.full((7,), np.inf)
    
    if level is not None:
        # check if input level is a numpy array
        if not isinstance(level, np.ndarray):
            raise TypeError("level must be a numpy array or None.")
        if level.shape != (7,):
            raise ValueError(f"Level must have shape (7,) for 7 octave bands (125 Hz - 8 kHz), but got {level.shape}.")
        if np.any(level < 1):
                warnings.warn(
                    "Input 'level' should be at least 1 dB for every octave band.",
                    stacklevel=1,
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

    modulation_frequencies = np.tile(modulation_frequencies, (rir_oct.cshape[0], 1))
    energy = rir_oct.time ** 2

    # Modulation transfer function calculation according to IEC 60268-16:2020, section 6.1
    term_exp = np.exp(-2j * np.pi * modulation_frequencies[:, :, None] * rir_oct.times)
    numerator = np.abs(np.sum(energy * term_exp, axis=-1))
    denominator = np.sum(energy, axis=-1)
    mtf = numerator / denominator
    mtf *= 1 / (1 + 10 ** (-snr[:, None] / 10))

    if level is not None:
        # Total intensity (signal + noise) according to IEC 60268-16:2020, Annex A.2.4
        Ik = 10 * np.log10(10 ** (level / 10) + 10 ** ((level - snr) / 10))
        if ambient_noise:
            # Ambient noise correction according to IEC 60268-16:2020, Annex A.2.3
            mtf *= level[:, None] / Ik[:, None]
        if rir_type == "acoustical":
            # Compute level-dependent auditory masking factor `I_amk` 
            # according to IEC 60268-16:2020, Annex A.4.2
            amdb = level.copy()
            amdb[amdb < 63] = 0.5*amdb[amdb < 63] - 65
            amdb[(63 <= amdb) & (amdb < 67)] = 1.8*amdb[(63 <= amdb) &
                                                        (amdb < 67)]-146.9
            amdb[(67 <= amdb) & (amdb < 100)] = 0.5*amdb[(67 <= amdb) &
                                                        (amdb < 100)]-59.8
            amdb[100 <= amdb] = amdb[100 <= amdb]-10
            a = 10**(amdb/10)
            # Masking intensity according to IEC 60268-16:2020, Annex A.4.2
            L_k1 = np.roll(Ik,1)
            I_k1 = 10**(L_k1/10)
            # Upward spread of masking: lower frequency bands mask higher bands
            I_amk = 10*np.log10(I_k1*a)
            I_amk[0] = 0  # No masking for lowest band (125 Hz)
            # Absolute speech reception threshold according to IEC 60268-16:2020, Annex A.4.3
            A_k = np.array([[46, 27, 12, 6.5, 7.5, 8, 12]]).T
            I_rt = 10**(A_k/10)
            # Apply auditory and masking effects according to IEC 60268-16:2020, Annex A.2.4
            mtf = mtf* ( Ik[:,None] / (
                10*np.log10(10**(Ik[:,None]/10)+10**(I_amk[:,None] /10) + I_rt)
                ))

    # Clip MTF values to valid range [0, 1] according to IEC 60268-16:2020, Section A.2.4 
    return np.clip(mtf, 0, 1)


def _sti_calc(mtf):
    """
    Computes the Speech Transmission Index (STI) from the MTF.

    Parameters
    ----------
    mtf : numpy.ndarray
        Modulation transfer function with shape ``(7, 14)`` for which the STI is computed.

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
            f"mtf must have shape (7, 14) for 7 octave bands and 14 modulation frequencies, but got {mtf.shape}."
        )
    
    # Effective SNR from MTF according to IEC 60268-16:2020, Annex A.2.1
    snr_eff = 10 * np.log10(mtf / (1 - mtf))
  
    # Clip SNR to [-15, 15] dB range according to IEC 60268-16:2020, Annex A.2.1
    snr_eff = np.clip(snr_eff, -15, 15)
    
    # Transmission index (TI) for each octave band and modulation frequency
    # according to IEC 60268-16:2020, Annex A.2.1
    TI = (snr_eff + 15) / 30
    
    # Modulation transmission index per octave band: average over modulation frequencies
    # according to IEC 60268-16:2020, Annex A.2.1
    mti = np.mean(TI, axis=-1)

    # STI Octave evaluation factors according to IEC 60268-16:2020, Table A.1
    alpha = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
    
    # Speech Transmission Index (STI) according to IEC 60268-16:2020, Annex A.2.1
    sti = np.sum(alpha * mti) - np.sum(beta * np.sqrt(mti[:-1] * mti[1:]))
    
    # limit STI to 1 according to IEC 60268-16:2020, section A.2.1
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
    impulse-response measurements [#iso]_.

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

    Parameters
    ----------
    limits : np.ndarray, list or tuple
        Four time limits (:math:`t_e`) in seconds, shape (4,)
        in ascending order.
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
    .. [#iso] ISO 3382, Acoustics — Measurement of the reverberation time of
        rooms with reference to other acoustical parameters.
    """

    # Check input type
    if not isinstance(energy_decay_curve1, pf.TimeData):
        raise TypeError(
            "energy_decay_curve1 must be a pyfar.TimeData or derived object.")
    if not isinstance(energy_decay_curve2, pf.TimeData):
        raise TypeError(
            "energy_decay_curve2 must be a pyfar.TimeData or derived object.")

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
        np.any(limits[0:2] < 0)
        or np.any(limits[0:2] > energy_decay_curve1.signal_length)
    ):
        raise ValueError(
            f"limits[0:2] must be between 0 and "
            f"{energy_decay_curve1.signal_length} seconds.",
        )
    if (
        np.any(limits[2:4] < 0)
        or np.any(limits[2:4] > energy_decay_curve2.signal_length)
    ):
        raise ValueError(
            f"limits[2:4] must be between 0 and "
            f"{energy_decay_curve2.signal_length} seconds.",
        )

    limits_energy_decay_curve1_idx = energy_decay_curve1.find_nearest_time(
        limits[0:2])
    limits_energy_decay_curve2_idx = energy_decay_curve2.find_nearest_time(
        limits[2:4])

    numerator = np.diff(
        energy_decay_curve2.time[..., limits_energy_decay_curve2_idx],
        axis=-1)[..., 0]
    denominator = np.diff(
        energy_decay_curve1.time[..., limits_energy_decay_curve1_idx],
        axis=-1)[..., 0]

    energy_ratio = numerator / denominator

    return energy_ratio
