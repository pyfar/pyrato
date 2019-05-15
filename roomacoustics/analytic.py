# -*- coding: utf-8 -*-

import numpy as np

def rectangular_room_rigid_walls(dimensions,
                                 source,
                                 receiver,
                                 reverberation_time,
                                 max_freq,
                                 samplingrate=44100,
                                 speed_of_sound=343.6,
                                 n_samples=2**18):
    """Calculate the transfer function of a rectangular room based on the
    analytic model.

    Parameters
    ----------
    dimensions : double, ndarray
        The dimensions of the room in the form [L_x, L_y, L_z]
    source : double, array
        The source position in Cartesian coordinates [x, y, z]
    receiver : double, ndarray
        The receiver position in Cartesian coordinates [x, y, z]
    max_freq : double
        The maximum frequency to consider for the calculation of the
        eigenfrequencies of the room
    samplingrate : int
        The sampling rate
    speed_of_sound : double, optional (343.6)
        THe speed of sound
    n_samples : int
        number of samples for the calculation

    Returns
    -------
    TODO

    """
    delta_n_raw = 3*np.log(10)/reverberation_time

    c = speed_of_sound
    L = np.asarray(dimensions)
    L_x = dimensions[0]
    L_y = dimensions[1]
    L_z = dimensions[2]
    source = np.asarray(source)
    receiver = np.asarray(receiver)
    f_max = max_freq

    idx = 0
    n_x_max = int(np.floor(2*f_max/c * L_x))
    for n_x in range(0, n_x_max):
        n_y_max = int(np.floor(np.real(np.sqrt(2*f_max/c)**2 - (n_x/L_x)**2) * L_y))
        for n_y in range(0, n_y_max):
            idx += int(np.floor(np.real(np.sqrt((2*f_max/c)**2 - (n_x/L_x)**2 - (n_y/L_y)**2) * L_z)) + 1)

    n_modes = idx
    print("Found {} eigenfrequencies.".format(n_modes))

    n = np.zeros((3, n_modes))

    idx = 0
    n_x_max = int(np.floor(2*f_max/c*L_x))
    for n_x in range(0, n_x_max):
        n_y_max = int(np.floor(np.real(np.sqrt(2*f_max/c)**2 - (n_x/L_x)**2) * L_y))
        for n_y in range(0, n_y_max):
            n_z_max = int(np.floor(np.real(np.sqrt((2*f_max/c)**2 - (n_x/L_x)**2 - (n_y/L_y)**2) * L_z)))

            idx_end = idx + n_z_max
            n[0, idx:idx_end] = n_x
            n[1, idx:idx_end] = n_y
            n[2, idx:idx_end] = np.arange(0, n_z_max)

            idx += n_z_max + 1

    print("Calculated {} eigenfrequencies.".format(idx - n_z_max))

    f_n = c/2*np.sqrt(np.sum((n/L[np.newaxis].T)**2, axis=0))

    coeff_receiver = np.cos(np.pi*n[0]*receiver[0]/L_x) \
                    *np.cos(np.pi*n[1]*receiver[1]/L_y) \
                    *np.cos(np.pi*n[2]*receiver[2]/L_z)
    coeff_source  =  np.cos(np.pi*n[0]*source[0]/L_x) \
                    *np.cos(np.pi*n[1]*source[1]/L_y) \
                    *np.cos(np.pi*n[2]*source[2]/L_z)

    K_n = np.prod(L) * 0.5**(np.sum(n > 0, axis=0))
    factor = c**2 / K_n
    coeff = coeff_source * coeff_receiver * factor

    # import ipdb; ipdb.set_trace()

    coeff[0] = 0.

    # delta = np.ones(n_modes) * delta_n_raw

    freqs = np.fft.rfftfreq(n_samples, d=1 / samplingrate)
    n_bins = freqs.size
    omega = 2*np.pi*freqs
    omega_n = 2*np.pi*f_n
    omega_squared = omega**2

    transfer_function = np.zeros(n_bins, np.complex)
    for om_n, coeff_n in zip(omega_n, coeff):
        den = (omega_squared - delta_n_raw**2 - om_n**2 - 2*1j*delta_n_raw*omega)
        transfer_function += (coeff_n/den)

    rir = np.fft.irfft(transfer_function, n=n_samples)
    return rir, transfer_function, freqs

