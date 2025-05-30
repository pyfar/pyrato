"""Analytic functions for room acoustics."""
# -*- coding: utf-8 -*-
import pyfar as pf
import numpy as np


def eigenfrequencies_rectangular_room_rigid(
        dimensions, max_freq, speed_of_sound=343.9, sort=True):
    """Calculate the eigenfrequencies of a rectangular room with rigid
    walls [1]_.

    Parameters
    ----------
    dimensions : float, numpy.ndarray
        The dimensions of the room in the form [L_x, L_y, L_z]
    max_freq : float
        The maximum frequency to consider for the calculation of the
        eigenfrequencies.
    speed_of_sound : double, optional
        The speed of sound in meters per second. The default is 343.9
    sort : bool, optional
        If ``True``, the return values will be sorted with ascending
        frequencies. By default this is ``True``.

    Returns
    -------
    f_n : double, ndarray
        The eigenfrequencies of the room
    n : int, ndarray
        The modal index

    References
    ----------
    .. [1] H. Kuttruff, Room acoustics, pp. 64-66, 4th Ed. Taylor & Francis,
           2009.

    Examples
    --------
    Calculate the eigenfrequencies under 75 Hz of a small room and plot.

    .. plot::

        >>> import numpy as np
        >>> import pyrato as ra
        >>> import matplotlib.pyplot as plt
        >>> from pyrato.analytic import eigenfrequencies_rectangular_room_rigid
        ...
        >>> L = [4, 5, 2.6]
        >>> f_n, n = eigenfrequencies_rectangular_room_rigid(
        ...     L, max_freq=75, speed_of_sound=343.6, sort=True)
        ...
        >>> ax = plt.axes()
        >>> ax.semilogx(f_n, np.arange(f_n.size), linestyle='', marker='o')
        >>> labels = [str(nn) for nn in n.T]
        >>> ax.set_yticks(np.arange(f_n.size))
        >>> ax.set_yticklabels(labels)
        >>> ax.set_xticks([30,  40, 50, 60, 70, 80])
        >>> ax.set_xticklabels(['30', '40', '50', '60', '70', '80'])
        >>> ax.set_xlabel('Frequency (Hz)')
        >>> ax.set_ylabel('Eigenfrequency index [$n_x, n_y, n_z$]')
        >>> plt.tight_layout()

    """
    c = speed_of_sound
    L = np.asarray(dimensions)
    L_x = dimensions[0]
    L_y = dimensions[1]
    L_z = dimensions[2]
    f_max = max_freq

    n_modes = 0
    n_x_max = int(np.floor(2*f_max/c * L_x)) + 1
    for n_x in range(0, n_x_max):
        n_y_max = int(np.floor(np.real(
            np.sqrt((2*f_max/c)**2 - (n_x/L_x)**2) * L_y))) + 1
        for n_y in range(0, n_y_max):
            n_modes += int(np.floor(np.real(
                np.sqrt(
                    (2*f_max/c)**2 - (n_x/L_x)**2 - (n_y/L_y)**2,
                ) * L_z))) + 1

    n = np.zeros((3, n_modes), dtype=int)

    idx = 0
    n_x_max = int(np.floor(2*f_max/c * L_x)) + 1
    for n_x in range(0, n_x_max):
        n_y_max = int(np.floor(np.real(
            np.sqrt((2*f_max/c)**2 - (n_x/L_x)**2) * L_y))) + 1
        for n_y in range(0, n_y_max):
            n_z_max = int(np.floor(np.real(
                np.sqrt(
                    (2*f_max/c)**2 - (n_x/L_x)**2 - (n_y/L_y)**2,
                ) * L_z))) + 1

            idx_end = idx + n_z_max
            n[0, idx:idx_end] = n_x
            n[1, idx:idx_end] = n_y
            n[2, idx:idx_end] = np.arange(0, n_z_max)

            idx += n_z_max

    f_n = c/2*np.sqrt(np.sum((n/L[np.newaxis].T)**2, axis=0))

    if sort is True:
        sort_idx = np.argsort(f_n)
        f_n = f_n[sort_idx]
        n = n[:, sort_idx]

    return f_n, n


def rectangular_room_rigid_walls(
        dimensions,
        source,
        receiver,
        reverberation_time,
        max_freq,
        samplingrate=44100,
        speed_of_sound=343.9,
        n_samples=2**18):
    r"""Calculate the transfer function of a rectangular room based on the
    analytic model.

    Implementation as given in [#]_ . The model is based on the solution
    for a room with rigid walls. The damping of the modes is included as
    a damping in the medium, not as a damping caused by the boundary.
    Consequently, all modes share the same damping factor calculated from
    the reverberation time as :math:`\delta = \frac{3\log(10)}{T_{60}}`.

    Parameters
    ----------
    dimensions : double, ndarray
        The dimensions of the room in the form [L_x, L_y, L_z]
    source : double, array
        The source position in Cartesian coordinates [x, y, z]
    receiver : double, ndarray
        The receiver position in Cartesian coordinates [x, y, z]
    reverberation_time : double
        The reverberation time of the room in seconds.
    max_freq : double
        The maximum frequency to consider for the calculation of the
        eigenfrequencies of the room
    samplingrate : int
        The sampling rate
    speed_of_sound : double, optional (343.9)
        The speed of sound
    n_samples : int
        number of samples for the calculation

    Returns
    -------
    rir : pyfar.Signal
        The room impulse response
    eigenfrequencies: ndarray, double
        The eigenfrequencies for which the room impulse response was
        calculated

    References
    ----------
    .. [#] H. Kuttruff, Room acoustics, pp. 64-66, 4th Ed. Taylor & Francis,
           2009.

    Example
    -------
    Calculate the sound field in a rectangular room with 1 s reverberation
    time for a given source and receiver combination.

    .. plot::

        >>> import numpy as np
        >>> import pyfar as pf
        >>> from pyrato.analytic import rectangular_room_rigid_walls
        ...
        >>> L = np.array([8, 5, 3])/10
        >>> source_pos = np.array([5, 3, 1.2])/10
        >>> receiver_pos = np.array([1, 1, 1.2])/10
        >>> rir, _ = rectangular_room_rigid_walls(
        >>>     L, source_pos, receiver_pos,
        >>>     reverberation_time=1, max_freq=1e3, n_samples=2**16,
        >>>     speed_of_sound=343.9)
        >>> pf.plot.time_freq(rir)

    """
    delta_n_raw = 3*np.log(10)/reverberation_time

    c = speed_of_sound
    L = np.asarray(dimensions)
    L_x = dimensions[0]
    L_y = dimensions[1]
    L_z = dimensions[2]
    source = np.asarray(source)
    receiver = np.asarray(receiver)

    f_n, n = eigenfrequencies_rectangular_room_rigid(
        dimensions, max_freq, speed_of_sound)

    coeff_receiver = \
        np.cos(np.pi*n[0]*receiver[0]/L_x) * \
        np.cos(np.pi*n[1]*receiver[1]/L_y) * \
        np.cos(np.pi*n[2]*receiver[2]/L_z)
    coeff_source = \
        np.cos(np.pi*n[0]*source[0]/L_x) * \
        np.cos(np.pi*n[1]*source[1]/L_y) * \
        np.cos(np.pi*n[2]*source[2]/L_z)

    K_n = np.prod(L) * 0.5**(np.sum(n > 0, axis=0))
    factor = c**2 / K_n
    coeff = coeff_source * coeff_receiver * factor

    coeff[0] = 0.

    freqs = np.fft.rfftfreq(n_samples, d=1 / samplingrate)
    n_bins = freqs.size
    omega = 2*np.pi*freqs
    omega_n = 2*np.pi*f_n
    omega_squared = omega**2

    transfer_function = np.zeros(n_bins, complex)
    for om_n, coeff_n in zip(omega_n, coeff):
        den = omega_squared - delta_n_raw**2 - om_n**2 - 2*1j*delta_n_raw*omega
        transfer_function += (coeff_n/den)

    rir = np.fft.irfft(transfer_function, n=n_samples)

    rir = pf.Signal(rir, samplingrate)
    return rir, f_n
