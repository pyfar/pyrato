# -*- coding: utf-8 -*-
from itertools import count

import numpy as np
from scipy import optimize


def eigenfrequencies_rectangular_room_rigid(
        dimensions, max_freq, speed_of_sound):
    """Calculate the eigenfrequencies of a rectangular room with rigid walls.

    Parameters
    ----------
    dimensions : double, ndarray
        The dimensions of the room in the form [L_x, L_y, L_z]
    max_freq : double
        The maximum frequency to consider for the calculation of the
        eigenfrequencies of the room
    speed_of_sound : double, optional (343.9)
        The speed of sound

    Returns
    -------
    f_n : double, ndarray
        The eigenfrequencies of the room
    n : int, ndarray
        The modal index

    References
    ----------
    .. [1]  H. Kuttruff, Room acoustics, pp. 64-66, 4th Ed. Taylor & Francis,
            2009.
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
                    (2*f_max/c)**2 - (n_x/L_x)**2 - (n_y/L_y)**2
                ) * L_z))) + 1

    n = np.zeros((3, n_modes))

    idx = 0
    n_x_max = int(np.floor(2*f_max/c * L_x)) + 1
    for n_x in range(0, n_x_max):
        n_y_max = int(np.floor(np.real(
            np.sqrt((2*f_max/c)**2 - (n_x/L_x)**2) * L_y))) + 1
        for n_y in range(0, n_y_max):
            n_z_max = int(np.floor(np.real(
                np.sqrt(
                    (2*f_max/c)**2 - (n_x/L_x)**2 - (n_y/L_y)**2
                ) * L_z))) + 1

            idx_end = idx + n_z_max
            n[0, idx:idx_end] = n_x
            n[1, idx:idx_end] = n_y
            n[2, idx:idx_end] = np.arange(0, n_z_max)

            idx += n_z_max

    f_n = c/2*np.sqrt(np.sum((n/L[np.newaxis].T)**2, axis=0))

    return f_n, n


def rectangular_room_rigid_walls(dimensions,
                                 source,
                                 receiver,
                                 reverberation_time,
                                 max_freq,
                                 samplingrate=44100,
                                 speed_of_sound=343.9,
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
    speed_of_sound : double, optional (343.9)
        The speed of sound
    n_samples : int
        number of samples for the calculation

    Returns
    -------
    rir : ndarray, double
        The room impulse response
    eigenfrequencies: ndarray, double
        The eigenfrequencies for which the room impulse response was
        calculated

    References
    ----------
    .. [1]  H. Kuttruff, Room acoustics, pp. 64-66, 4th Ed. Taylor & Francis,
            2009.


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

    transfer_function = np.zeros(n_bins, np.complex)
    for om_n, coeff_n in zip(omega_n, coeff):
        den = omega_squared - delta_n_raw**2 - om_n**2 - 2*1j*delta_n_raw*omega
        transfer_function += (coeff_n/den)

    rir = np.fft.irfft(transfer_function, n=n_samples)
    return rir, f_n


def transcendental_equation_eigenfrequencies_impedance(k_n, k, L, zeta):
    """The transcendental equation to be solved for the estimation of the
    complex eigenfrequencies of the rectangular room with uniform impedances.
    This function is intended as the cost function for solving for the roots.

    Parameters
    ----------
    k_n : array, double
        The real and imaginary part of the complex eigenfrequency
    k : double
        The real valued wave number
    L : double
        The room dimension
    zeta : array, double
        The normalized specific impedance

    Returns
    -------
    func : array, double
        The real and imaginary part of the transcendental equation
    """
    k_n_real = k_n[0]
    k_n_imag = k_n[1]

    k_n_complex = k_n_real + 1j*k_n_imag

    left = np.tan(k_n_complex*L)
    right = \
        (1j*k*L*(zeta[0] + zeta[1])) / (k_n_complex*L * (zeta[0]*zeta[1] +
                                        (k*L)**2/(k_n_complex*L)**2))
    func = left - right

    return [func.real, func.imag]


def initial_solution_transcendental_equation(k, L, zeta):
    """ Initial solution to the transcendental equation for the complex
    eigenfrequencies of the rectangular room with uniform impedance at
    the boundaries. This will approximate the zeroth order mode.

    Parameters
    ----------
    k : array, double
        Wave number array

    Returns
    -------
    k_0 : array, complex
        The complex zero order eigenfrequency
    """
    zeta_0 = zeta[0]
    zeta_L = zeta[1]
    k_0 = 1/L*np.sqrt(-(k*L)**2/zeta_0/zeta_L + 1j*k*L*(1/zeta_0+1/zeta_L))

    return k_0


def eigenfrequencies_rectangular_room_1d(
        L_l, ks, k_max, zeta):
    """Estimates the complex eigenvalues in the wavenumber domain for one
    dimension by numerically solving for the roots of the transcendental
    equation. A initial approximation to the zeroth order mode is applied to
    improve the conditioning of the problem.

    Parameters
    ----------
    L_l : double
        The dimension in m
    ks : array, double
        The wave numbers for which the eigenvalues are to be solved.
    k_max : double
        The real part of the largest eigenvalue. This solves as a stopping
        criterion independent from the real wave number k.
    zeta : array, double
        The normalized specific impedance on the boundaries.

    Returns
    -------
    k_ns : array, complex
        The complex eigenvalues for each wavenumber

    Note
    ----
    This function assumes that the real part of the largest eigenvalue may be
    calculated using the approximation for rigid walls.

    """
    ks = np.atleast_1d(ks)
    n_l_max = int(np.ceil(k_max/np.pi*L_l))

    k_ns_l = np.zeros((n_l_max, len(ks)), dtype=np.complex64)
    k_n_init = initial_solution_transcendental_equation(ks[0], L_l, zeta)
    for idx_k, k in enumerate(ks):
        idx_n = 0
        while k_n_init.real < k_max:
            args_costfun = (k, L_l, zeta)
            kk_n = optimize.fsolve(
                transcendental_equation_eigenfrequencies_impedance,
                (k_n_init.real, k_n_init.imag),
                args=args_costfun)
            if kk_n[0] > k_max:
                break
            else:
                kk_n_cplx = kk_n[0] + 1j*kk_n[1]
                k_ns_l[idx_n, idx_k] = kk_n_cplx
                k_n_init = (kk_n_cplx*L_l + np.pi) / L_l
                idx_n += 1

        k_n_init = k_ns_l[0, idx_k]

    return k_ns_l


def normal_eigenfrequencies_rectangular_room_impedance(
        L, ks, k_max, zeta):

    k_ns = []
    for dim, L_l, zeta_l in zip(count(), L, zeta):
        k_ns_l = eigenfrequencies_rectangular_room_1d(
            L_l, ks, k_max, zeta_l)
        k_ns.append(k_ns_l)
    return k_ns


def eigenfrequencies_rectangular_room_impedance(L, ks, k_max, zeta):
    """Estimates the complex eigenvalues in the wavenumber domain for a
    rectangular room with arbitrary uniform impedances on the boundary by
    numerically solving for the roots of the transcendental equation.
    A initial approximation to the zeroth order mode is applied to
    improve the conditioning of the problem. The eigenvalues corresponding to
    tangential and oblique modes are calculated from the eigenvalues of the
    respective axial modes. Resulting eigenvalues with a real part larger than
    k_max will be discarded.

    Parameters
    ----------
    L : array, double
        The dimensions in m
    ks : array, double
        The wave numbers for which the eigenvalues are to be solved.
    k_max : double
        The real part of the largest eigenvalue. This solves as a stopping
        criterion independent from the real wave number k.
    zeta : array, double
        The normalized specific impedance on the boundaries.

    Returns
    -------
    k_ns : array, complex
        The complex eigenvalues for each wavenumber
    mode_indices : array, integer
        The wave number indices of respective eigenvalues.

    Note
    ----
    Eigenvalues smaller for a wave number $k < 0.02$ will be replaced by the
    value for the closest larger wave number to ensure finding the root.

    """
    ks = np.atleast_1d(ks)
    mask = ks >= 0.02
    ks_search = ks[mask]
    k_ns = normal_eigenfrequencies_rectangular_room_impedance(
        L, ks_search, k_max, zeta
    )
    for idx in range(0, len(L)):
        k_ns[idx] = np.vstack(
            (np.tile(k_ns[idx][0], (np.sum(~mask), 1)), k_ns[idx])
            )

    n_z = np.arange(0, k_ns[2].shape[0])
    n_y = np.arange(0, k_ns[1].shape[0])
    n_x = np.arange(0, k_ns[0].shape[0])

    combs = np.meshgrid(n_x, n_y, n_z)
    perms = np.array(combs).T.reshape(-1, 3)

    kk_ns = np.sqrt(
        k_ns[0][perms[:, 0]]**2 +
        k_ns[1][perms[:, 1]]**2 +
        k_ns[2][perms[:, 2]]**2)

    mask_perms = (kk_ns[:, -1].real < k_max)

    mask_bc = np.broadcast_to(
        np.atleast_2d(mask_perms).T,
        (len(mask_perms), len(ks)))

    kk_ns = kk_ns[mask_bc].reshape(-1, len(ks))

    mode_indices = perms[mask_bc[:, 0]]

    return kk_ns, mode_indices


def pressure_mode_solution(position, eigenvals, phase_shift):
    return np.cosh(1j*eigenvals * position + phase_shift)


def pressure_modal_superposition(
        ks, omegas, k_ns, kk_ns, mode_indices, r_R, r_S, L, zeta_0):

    phi = []
    p_ns_r = []
    p_ns_s = []
    K_n = []

    for idx in range(0, 3):
        phi.append(np.arctanh(
            np.tile(ks, ((k_ns[idx].shape[-1]), 1)).T / (k_ns[idx]*zeta_0[idx])
        ))
        p_ns_r.append(pressure_mode_solution(k_ns[idx], r_R[idx], phi[idx]))
        p_ns_s.append(pressure_mode_solution(k_ns[idx], r_S[idx], phi[idx]))
        K_n.append(
            L[idx]/2*(
                1+1/(1j*k_ns[idx]*L[idx]) *
                np.sinh(1j*k_ns[idx]*L[idx]) *
                np.cosh(1j*k_ns[idx]*L[idx] + 2*phi[idx])
            )
        )

    p_x_n = np.zeros(len(ks), dtype=np.complex)

    for lin_idx, mode_idx in zip(count(), mode_indices):
        pp_n_r = \
            p_ns_r[0][:, mode_idx[0]] * \
            p_ns_r[1][:, mode_idx[1]] * \
            p_ns_r[2][:, mode_idx[2]]
        pp_n_s = \
            p_ns_s[0][:, mode_idx[0]] * \
            p_ns_s[1][:, mode_idx[1]] * \
            p_ns_s[2][:, mode_idx[2]]
        KK_n = \
            K_n[0][:, mode_idx[0]] * \
            K_n[1][:, mode_idx[1]] * \
            K_n[2][:, mode_idx[2]]
        nom = 1j*omegas*1.2*pp_n_r*pp_n_s
        denom = (KK_n*(kk_ns[:, lin_idx]**2 - ks**2))
        p_x_n += nom / denom

    return p_x_n
