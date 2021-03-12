"""
Returns a np array with the female or male STI (speech transmission index), a 
single number value on a metric scale between 0 (bad) and 1 (excellent) for 
quality assessment of speech transmission channels.

The indices are based on the modulation transfer function (MTF) that determinines
affections of the intensity envelope throughout the transmission. The MTF values
are assessed from the IR and are further modified based on auditory, ambient 
noise and masking aspects.

STI considers 7 octaves between 125 Hz and 8 kHz (125 Hz is not considered for
the female STI) and 14 modulation frequencies between 0.63 Hz and 12 Hz. 


============================

Reference
---------
..  [1] IEC 60268-16:2011
    Sound system equipment - Part 16: Objective rating of speech intelligibility 
    by speech transmission index

"""

import warnings
from pyfar import Signal  
import pyfar.dsp.filter as filt
import numpy as np

def sti(signal, data_type=None, gender='male', level=np.array([np.nan]), 
            snr=np.array([np.nan])):
    """
    Parameters
    ---------
    signal : Signal
        The IR to be analyzed. Length must be at least 1.6 s and not shorter than
        1/2 RT60. [1],  p. ??

    data_type : 'electrical', 'acoustical'
        Determines weather input signals are obtained acoustically or electrically.
        Auditory effects can only be considered when "acoustical" [1], p. ??. 
        Default is 'none'.

    gender: 'female', 'male'
        Defines the applied weighting factors. Default is 'male' as the STI 
        algorithm is initially based on the male speech spectrum.

    noise: np.array, 0
        Level of the test signal in octave for bands 125 Hz - 8000 Hz in dB_SPL.

    snr: np.array, 0
        SNR in octave for bands 125 Hz - 8000 Hz in dB.

    """
    

    # process signal, check input
    sig = signal.copy()
    #2D Signal
    sig = sig.flatten()

#check for sufficient signal length (IEC 60268-16 p. ## and p. ##) 
    if signal.n_samples/sig.sampling_rate < 1.6:
        warnings.warn("Signal length below 1.6 seconds.")

# apply octave band filters (preliminary with crossover; later: perf. 
# reconstructing oct. filter)
    sig_oct = (filt.crossover(sig, 4, [90, 180, 355, 710, 1400, 2800, 5600,
    11200])[1:-1,:])
# MTF per octave and modulation frequency
    mf = [0.63, 0.80, 1, 1.25, 1.60, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12]
    mtf = np.zeros((14,)+sig_oct.cshape)
    sig_en = np.sum(np.square(sig_oct.time),axis=-1)
    sig_oct_newlength = sig_oct.copy()

    #calculate mtf
    with np.errstate(divide='ignore'):  #return nan for empty IR
        for i in range(len(mf)):
            fbin = int(np.ceil(mf[i]*sig_oct.n_samples/sig_oct.sampling_rate))
            newlength = int(sig_oct.sampling_rate*fbin/mf[i])
            sig_oct_newlength.time = np.concatenate((sig_oct.time, 
                np.zeros((sig_oct.cshape + (newlength-sig_oct.n_samples,)))),
                axis=-1)
            sig_oct_newlength_sq = Signal(np.square(sig_oct_newlength.time),
                sig_oct.sampling_rate)
            mtf[i]=abs(sig_oct_newlength_sq.freq[...,fbin])/sig_en


# Adjustment of mtf for ambient noise, auditory masking and threshold effects

    # calculation only if levels are given
    if not np.isnan(level).all():
        # warn if SNR is below 20 dB in any band
        level = level
        snr = snr
        if np.any(snr<20):
            warnings.warn("SNR should be at least 20 dB for every octave band.")
        # set snr to infinity if not given
        if np.isnan(snr).all():
            warnings.warn("SNR is not or incompletely given, calculation with \
            infinite SNR.")
            snr = np.ones(level.shape)*np.inf
        # overall intensity
        i_k = 10**(level/10)+10**((level-snr)/10)
        # Apply ambient noise effects
        mtf =  mtf*(10**(level.T/10)/i_k.T)

        # consideration of auditory effects only for acoustical signals
        if data_type == "electrical":
            pass
        else: 
            if data_type != "acoustical":
                warnings.warn("Data type is considered as acoustical. \
                Consideration of masking effects not valid for electrically \
                obtained signals.") 
            
            # level-dependent auditory masking
            amdb = level.copy()
            amdb[amdb<63]=0.5*amdb[amdb<63]-65
            amdb[(63<=amdb) & (amdb<67)]=1.8*amdb[(63<=amdb) & (amdb<67)]-146.9
            amdb[(67<=amdb) & (amdb<100)]=0.5*amdb[(67<=amdb) & (amdb<100)]-59.8
            amdb[100<=amdb]=amdb[100<=amdb]-10
            amf = 10**(amdb/10)

            # masking intensity
            i_am = np.zeros(i_k.shape)
            i_am = i_k*amf

            # absolute speech reception threshold
            artdb = np.array([[46,27,12,6.5,7.5,8,12]])
            i_rt = 10**(artdb/10)
            # apply auditory and masking effects
            i_T = i_k/(i_k+i_am+i_rt)
            i_T = np.squeeze(i_T)
            mtf = mtf*i_T.T

    else:
       warnings.warn("Signal level not or incompletely given. Auditory and \
           masking effects not considered.")

# limit mtf to 1
    mtf[mtf>1] = 1

# effective SNR per octave and modulation frequency
    with np.errstate(divide='ignore'):
        snr_eff = 10*np.log10(mtf/(1-mtf))
    # min value: -15 dB, max. value +15 dB
    snr_eff[snr_eff<-15] = -15
    snr_eff[snr_eff>15] = 15

# transmission index (TI) per octave and modulation frequency
    ti = ((snr_eff+15)/30)

# modulation transmission indices (MTI) per octave
    mti = (np.array(1/14*np.sum(ti, axis = 0))).reshape(7,sig.cshape[-1])

# speech transmission index (STI)
    if gender == "female":
        alpha = np.array([[0], [0.117], [0.223], [0.216], [0.328], [0.250], [0.194]])
        beta = np.array([[0], [0.099], [0.066], [0.062], [0.025], [0.076]])
        sti = np.sum(alpha * mti, axis = 0) - np.sum(beta * np.sqrt(mti[:6,...]
            * mti[1:,...]),axis = 0)
        sti = sti.reshape(signal.cshape)
        return sti
    elif gender == "male":
        alpha = np.array([[0.085], [0.127], [0.230], [0.233], [0.309], [0.224], [0.173]])
        beta = np.array([[0.085], [0.078], [0.065], [0.011], [0.047], [0.095]])
        sti = np.sum(alpha * mti, axis = 0) - np.sum(beta * np.sqrt(mti[:6,...]
            * mti[1:,...]),axis = 0)
        sti = sti.reshape(signal.cshape)
        return sti
    else:
        warnings.warn("Gender must be 'male' or 'female'")
        return


