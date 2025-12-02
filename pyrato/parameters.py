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

def speech_transmission_index(
    data,
    data_type=None,
    level=None,
    snr=None,
    amb=True):
    """
    This function calculates the speech transmission index (STI)
    according to [#iec]_ using the indirect method.

    Returns a numpy array with the STI, a single number value
    on a metric scale between 0 (bad) and 1 (excellent) for quality assessment
    of speech transmission, in shape of input channels.

    The indices are based on the modulation transfer function (MTF) that
    determines affections of the intensity envelope throughout the
    transmission. The MTF values are assessed from the IR and are further
    modified based on auditory, ambient noise and masking aspects.

    STI considers 7 octaves between 125 Hz and 8 kHz
    and 14 modulation frequencies between 0.63 Hz and
    12 Hz [#iec]_.

    Parameters
    ----------
    data : pyfar.Signal
        The room impulse response with dimension [channel, n_samples].

    data_type : 'electrical', 'acoustical'
        Determines weather input signals are obtained acoustically or
        electrically. Auditory effects can only be considered when "acoustical"
        [#iec]_, section A.3.1. Default is 'acoustical'.

    level: np.ndarray, None
        Level of the test signal without any present noise sources.
        Given in 7 octave bands 125 Hz - 8000 Hz in dB_SPL. Np array with
        7 elements per row and rows for all given IR.
        See [#iec]_, section A.3.2.

    snr: np.ndarray, None
        Ratio between test signal level (see above) and noise level when
        the test source is turned of. Given in 7 octave bands 125 Hz - 8000 Hz
        in dB_SPL. Np array with 7 elements per row and rows for all given IR.
        See [1], section 3

    amb: bool, True
        Consideration of ambient noise effects as proposed in [2],
        section A.2.3. Default is True.

    References
    ----------
    .. [#iec] IEC 60268-16: 2021-10
     Sound system equipment - Part 16: Objective rating of speech
     intelligibility by speech transmission index.
    """

    # check if input data a pyfar.Signal
    if not isinstance(data, pf.Signal):
        raise TypeError("Input data must be a pyfar.Signal.")

    # Check if the signal is at least 1.6 seconds long ([1], sectionn 6.2)
    if not data.n_samples / data.sampling_rate >= 1.6:
        raise ValueError("Input signal must be at least 1.6 seconds long.")

    # flatten for easy loop
    cshape = data.cshape
    # data = data.flatten()

    if snr is not None:
        snr = np.asarray(snr)
        # Check if SNR has the correct number of components and SNR-threshold
        if snr.shape != data.cshape:
            raise ValueError("SNR consists of wrong number of components.")
        if np.any(snr < 20):
            warnings.warn(
                "SNR should be at least 20 dB for every octave band.",
                stacklevel=2)
    else:
        snr = np.ones((data.cshape[0],7))*np.inf

    if level is not None:
        level = np.asarray(level)
        # Check if level has the correct number of components
        if level.shape != data.cshape:
            raise ValueError("Level consists of wrong number of components.")
        if np.any(level < 1):
            warnings.warn(
                "Level should be at least 1 dB for every octave band.",
                stacklevel=2)
    else:
        level = np.full((data.cshape[0]), None)

     # check data_type
    if data_type is None:
        warnings.warn("Data type is considered as acoustical. Consideration "
                      "of masking effects not valid for electrically obtained "
                      "signals.", stacklevel=2)
        data_type = "acoustical"
    if data_type not in ["electrical", "acoustical"]:
        raise ValueError(f"Data_type is '{data_type}' but must be "
                         "'electrical' or 'acoustical'.")

    sti_ = np.zeros(data.cshape)

    # Loop through each channel
    for cc in range(data.cshape[0]):

        # calculate mtf for 14 modulation frequencies in 7 octave bands
        mtf = modulation_transfer_function(data[cc], data_type, level[cc], snr[cc], amb)

        # calculate sti from mtf
        sti_[cc] = sti_calc(mtf, data[cc])

    sti_ = np.reshape(sti_, cshape)
    return sti_


def modulation_transfer_function(data, data_type, level, snr, amb):
    """
    Calculate the modulation transfer function (MTF) for given
    impulse response.

    Parameters
    ----------
    data : pyfar.Signal
        The room impulse response with dimension [n_samples].

    data_type : str
        Type of input signals, either 'electrical' or 'acoustical'.

    level : np.array
        Level of the test signal without any present noise sources.
        Given in 7 octave bands 125 Hz - 8000 Hz in dB_SPL. Np array with
        7 elements per row.

    snr : np.array
        Ratio between test signal level and noise level when the test source
        is turned off. Given in 7 octave bands 125 Hz - 8000 Hz in dB_SPL.
        Np array with 7 elements per row.

    amb : bool
        Consideration of ambient noise effects. Default is True.

    Returns
    -------
    mtf : np.array
        Modulation transfer function
    """

    # fractional octave band filtering
    data_oct = pf.dsp.filter.fractional_octave_bands(data, num_fractions=1,
                                            freq_range=(125, 8e3))

    # modulation frequencies for each octave band([1], section 6.1)
    f_m = np.array([[0.63, 0.80, 1, 1.25, 1.60, 2, 2.5, 3.15, 4, 5, 6.3, 8,
                     10, 12.5],]*data_oct.cshape[0])
    #f_m = np.tile(f_m[:,:,None], data.times.shape) 

    #data_oct_en = np.sum(data_oct.time, axis=-1)
    #data_oct_energy = data_oct.time[:,:,np.newaxis]**2

    # energy
    data_oct_energy = data_oct.time**2
    #data_oct_energy = np.transpose(data_oct_energy,(0,2,1))
    #term_exp = np.exp(-2j * np.pi * f_m  * np.transpose(data_oct.times[:,None,None],(1,2,0)))

     # modulation transfer function (MTF) ([1], section A.2.2)
    term_exp = np.exp(-2j * np.pi * f_m[:,:,None]  * data_oct.times)
    term_a = np.abs(np.sum(data_oct_energy * term_exp,axis=-1))
    term_b  = np.sum(data_oct_energy, axis=-1) 
    mtf =   (term_a / term_b) * (1 / (1 + 10 ** (-snr[:,None]/10))) 

    # Adjustment of mtf for ambient noise, auditory masking and threshold
    # effects ([1], A.2.3, A.2.4) mtf =   (term_a / term_b[:,None]) * (1 / (1 + 10 ** (-snr/10)))
    if level is not None:
        # overall intensity level
        Ik = 10 * np.log10(10**(level/10) + 10**((level-snr)/10))
        # apply ambient noise effects ([1], A.2.3)
        if amb is True:
            mtf = mtf*(level[:,None] / Ik[:,None])
        # consideration of auditory effects, only for acoustical signals
        # ([1], section A.2.4)
        if data_type == "electrical":
            pass
        else:
            # level-dependent auditory masking ([1], section A.4.2)
            amdb = level.copy()
            amdb[amdb < 63] = 0.5*amdb[amdb < 63] - 65
            amdb[(63 <= amdb) & (amdb < 67)] = 1.8*amdb[(63 <= amdb) &
                                                        (amdb < 67)]-146.9
            amdb[(67 <= amdb) & (amdb < 100)] = 0.5*amdb[(67 <= amdb) &
                                                         (amdb < 100)]-59.8
            amdb[100 <= amdb] = amdb[100 <= amdb]-10
            a = 10**(amdb/10) 

            # masking intensity
            L_k1 = np.roll(Ik,1)
            I_k1 = 10**(L_k1/10)

            I_amk = 10*np.log10(I_k1*a)
            I_amk[0] = 0
            # absolute speech reception threshold ([1], section A.4.3)
            A_k = np.array([[46, 27, 12, 6.5, 7.5, 8, 12]]).T
            I_rt = 10**(A_k/10)
            # apply auditory and masking effects ([1], section A.2.4)
            mtf = mtf* ( Ik[:,None] /( 10*np.log10(10**(Ik[:,None]/10) 
                                                   +10**(I_amk[:,None] /10) + I_rt))) 
    # limit mtf to 1
    mtf[mtf > 1] = 1

    return mtf


def sti_calc(mtf, data):
    # effective SNR per octave and modulation frequency ([1], section A.2.1)
    with np.errstate(divide='ignore'):
        snr_eff = 10*np.log10(mtf / (1-mtf))
    # min value: -15 dB, max. value +15 dB
    snr_eff[snr_eff < -15] = -15
    snr_eff[snr_eff > 15] = 15

    # transmission index TI_k,fm per octave and modulation frequency ([1],
    # section A.2.1)
    TI = ((snr_eff + 15) / 30)

    # modulation transmission indices (MTI) per octave
    mti = 1/14*np.sum(TI, axis=-1)

    # STI Octave evaluation factors according tabelle A.1
    alpha = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
    # speech transmission index (STI)
    sti = np.sum(alpha * mti) - np.sum(beta * np.sqrt(mti[:6] * mti[1:]))

    # reshape output to initial signal shape
    sti = sti.reshape(data.cshape)

    # limit STI to 1 ([1], section A.5.6)
    sti[sti > 1] = 1
    return sti

