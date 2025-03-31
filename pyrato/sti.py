import warnings
import pyfar.dsp.filter as filt
import numpy as np


def sti(signal, data_type=None, gender='male', level=None, snr=None, amb=True):

    """
    Calculation of the speech transmission index (STI).

    Returns a np array with the female or male STI , a single number value
    on a metric scale between 0 (bad) and 1 (excellent) for quality assessment
    of speech transmission channels.

    The indices are based on the modulation transfer function (MTF) that
    determines affections of the intensity envelope throughout the
    transmission. The MTF values are assessed from the IR and are further
    modified based on auditory, ambient noise and masking aspects.

    STI considers 7 octaves between 125 Hz and 8 kHz (125 Hz is not considered
    for the female STI) and 14 modulation frequencies between 0.63 Hz and
    12 Hz.

    References
    ----------
    ..  [1] IEC 60268-16:2011
    Sound system equipment - Part 16: Objective rating of speech
    intelligibility by speech transmission index

    ..  [2] IEC 60268-16/Ed.5: 2019-08 (DRAFT)
    Sound system equipment - Part 16: Objective rating of speech
    intelligibility by speech transmission index
    ============================
    Parameters
    ---------
    signal : Signal
        The impulse responses (IR) to be analyzed. Length must be at least
        1.6 s and not shorter than 1/2 RT60. [1], section 6.2

    data_type : 'electrical', 'acoustical'
        Determines weather input signals are obtained acoustically or
        electrically. Auditory effects can only be considered when "acoustical"
        [1], section A.3.1. Default is 'None'.

    gender: 'female', 'male'
        Defines the applied weighting factors. Default is 'male' because the
        STI is more critical in this case due to the expanded low frequency
        range of the male voice.

    level: np.array, None
        Level of the test signal without any present noise sources.
        Given in 7 octave bands 125 Hz - 8000 Hz in dB_SPL. Np array with
        7 elements per row and rows for all given IR. See [1], section A.3.2

    snr: np.array, None
        Ratio between test signal level (see above) and noise level when
        the test source is turned of. Given in 7 octave bands 125 Hz - 8000 Hz
        in dB_SPL. Np array with 7 elements per row and rows for all given IR.
        See [1], section 3

    amb: bool, True
        Consideration of ambient noise effects as proposed in [2],
        section A.2.3. Default is True.

    """
    # preprocess and verify input data
    sig, inp_sig_oct, inp_da_ty, inp_gen, inp_lvl, inp_snr, inp_amb = \
        preprocess(signal, data_type, gender, level, snr, amb)
    # calculate IR for 14 modulation frequencies in 7 octave bands
    mtf_data = mtf(inp_sig_oct, inp_da_ty, inp_lvl, inp_snr, inp_amb)
    # calculate sti from MTF
    sti_data = sti_calc(mtf_data, signal, inp_gen)
    # return result
    return sti_data


def preprocess(signal, data_type=None, gender='male', level=None, snr=None,
               amb=True):

    # get flattened signal copy
    sig = signal.copy().flatten()

    # check / flatten snr
    if snr is not None:
        snr = np.asarray(snr).flatten()
        if np.squeeze(snr.flatten().shape)/7 != (np.squeeze(sig.cshape)):
            raise ValueError("SNR consists of wrong number of components.")
        if np.any(snr < 20):
            warnings.warn("SNR should be at least 20 dB for every octave "
                          "band.")
        snr = np.reshape(snr, (-1, 7)).T
    # set snr to infinity if not given
    else:
        snr = np.ones([7, np.squeeze(sig.cshape)])*np.inf

    # check / flatten level
    if level is not None:
        level = np.asarray(level).flatten()
        if np.squeeze(level.flatten().shape)/7 != (np.squeeze(sig.cshape)):
            raise ValueError("Level consists of wrong number of components.")
        level = np.reshape(level, (-1, 7)).T

    # check for sufficient signal length ([1], section 6.2)
    if signal.n_samples/sig.sampling_rate < 1.6:
        warnings.warn("Signal length below 1.6 seconds.")

    # check data_type
    if data_type is None:
        warnings.warn("Data type is considered as acoustical. Consideration "
                      "of masking effects not valid for electrically obtained "
                      "signals.")
        data_type = "acoustical"
    if data_type not in ["electrical", "acoustical"]:
        raise ValueError(f"Data_type is '{data_type}' but must be "
                         "'electrical' or 'acoustical'.")

    # check gender
    if gender not in ["male", "female"]:
        raise ValueError(f"Gender is '{gender}' but must be 'male' "
                         "or 'female'.")

    # apply octave band filters (preliminary with crossover; later: perf.
    # reconstructing oct. filter)
    sig_oct = (filt.fractional_octave_bands(sig, num_fractions=1,
                                            freq_range=(125, 8e3)))

    return sig, sig_oct, data_type, gender, level, snr, amb


def mtf(sig_oct, data_type, level, snr, amb):
    # MTF per octave and modulation frequency ([1], section 6.1)
    mf = [0.63, 0.80, 1, 1.25, 1.60, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12]
    mtf = np.zeros((len(mf),)+sig_oct.cshape)
    sig_en = np.sum(sig_oct.time**2, axis=-1)
    t = np.arange(sig_oct.n_samples)
    with np.errstate(divide='ignore'):  # return nan for empty IR
        for i, f in enumerate(mf):
            mtf[i] = np.abs(np.sum(sig_oct.time**2*np.exp(-2*1j*np.pi*mf[i] *
                                                          t/44100),
                                   axis=-1))/sig_en * np.squeeze(1/(1+10 **
                                                                    (-snr/10)))

    # Adjustment of mtf for ambient noise, auditory masking and threshold
    # effects ([1], sections A.3, A.5.3)
    if level is not None:
        # overall intensity ([1], section A.3.2)
        i_k = 10**(level/10)+10**((level-snr)/10)
        # apply ambient noise effects (proposed in [2], section A.2.3)
        if amb is True:
            mtf = mtf*(10**(np.squeeze(level)/10)/np.squeeze(i_k))
        # consideration of auditory effects only for acoustical signals
        # ([1], section A.3.1)
        if data_type == "electrical":
            pass
        else:
            # level-dependent auditory masking ([1], section A.3.2)
            amdb = level.copy()
            amdb[amdb < 63] = 0.5*amdb[amdb < 63]-65
            amdb[(63 <= amdb) & (amdb < 67)] = 1.8*amdb[(63 <= amdb) &
                                                        (amdb < 67)]-146.9
            amdb[(67 <= amdb) & (amdb < 100)] = 0.5*amdb[(67 <= amdb) &
                                                         (amdb < 100)]-59.8
            amdb[100 <= amdb] = amdb[100 <= amdb]-10
            amf = 10**(amdb/10)

            # masking intensity
            i_am = np.zeros(i_k.shape)
            i_am = i_k*amf

            # absolute speech reception threshold ([1], section A.3.3)
            artdb = np.array([[46, 27, 12, 6.5, 7.5, 8, 12]]).T
            i_rt = 10**(artdb/10)
            # apply auditory and masking effects ([1], section A.5.3)
            i_T = i_k/(i_k+i_am+i_rt)
            i_T = np.squeeze(i_T)
            mtf = mtf*i_T

        # limit mtf to 1 ([1], section A.5.3)
    mtf[mtf > 1] = 1

    return mtf


def sti_calc(mtf, signal, gender):
    # effective SNR per octave and modulation frequency ([1], section A.5.4)
    with np.errstate(divide='ignore'):
        snr_eff = 10*np.log10(mtf/(1-mtf))
    # min value: -15 dB, max. value +15 dB
    snr_eff[snr_eff < -15] = -15
    snr_eff[snr_eff > 15] = 15

    # transmission index (TI) per octave and modulation frequency ([1],
    # section A.5.5)
    ti = ((snr_eff+15)/30)

    # modulation transmission indices (MTI) per octave ([1], section A.5.6)
    mti = (np.array(1/14*np.sum(ti, axis=0))).reshape(7, signal.flatten()
                                                      .cshape[-1])

    # speech transmission index (STI) ([1], section A.5.6)
    if gender == "female":
        alpha = np.array([[0], [0.117], [0.223], [0.216], [0.328], [0.250],
                         [0.194]])
        beta = np.array([[0], [0.099], [0.066], [0.062], [0.025], [0.076]])
        sti = np.sum(alpha * mti, axis=0) - np.sum(beta *
                                                   np.sqrt(mti[:6, ...] *
                                                           mti[1:, ...]),
                                                   axis=0)

    elif gender == "male":
        alpha = np.array([[0.085], [0.127], [0.230], [0.233], [0.309], [0.224],
                         [0.173]])
        beta = np.array([[0.085], [0.078], [0.065], [0.011], [0.047], [0.095]])
        sti = np.sum(alpha * mti, axis=0) - np.sum(beta *
                                                   np.sqrt(mti[:6, ...] *
                                                           mti[1:, ...]),
                                                   axis=0)

    # reshape output to initial signal shape
    sti = sti.reshape(signal.cshape)

    # limit STI to 1 ([1], section A.5.6)
    sti[sti > 1] = 1
    return sti
