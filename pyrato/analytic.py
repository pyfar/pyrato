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
    ..  [2] H. Kuttruff, Room acoustics, pp. 64-66, 4th Ed. Taylor & Francis,
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
    r"""Calculate the transfer function of a rectangular room based on the
    analytic model as given in [2]_ . The model is based on the solution
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
    ..  [2] H. Kuttruff, Room acoustics, pp. 64-66, 4th Ed. Taylor & Francis,
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

    transfer_function = np.zeros(n_bins, complex)
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
        (1j*k*L*np.sum(zeta)) / (k_n_complex*L * (np.prod(zeta) +
                                 (k*L)**2/(k_n_complex*L)**2))
    func = left - right

    return [func.real, func.imag]


def transcendental_equation_eigenfrequencies_impedance_newton(k_n, k, L, zeta):
    r"""The transcendental equation to be solved for the estimation of the
    complex eigenfrequencies of the rectangular room with uniform impedances.
    This function is intended as the cost function for solving for the roots.

    .. math::

        \frac{ i k \left(\zeta_{0} + \zeta_{L}\right)}{k_{n}
        \left(\frac{k^{2}}{k_{n}^{2}} + \zeta_{0} \zeta_{L}\right)} +
        \tan{\left(L k_{n} \right)} = 0

    Parameters
    ----------
    k_n : complex
        The complex eigenvalue
    k : double
        The real valued wave number
    L : double
        The room dimension
    zeta : array, double
        The normalized specific impedance

    Returns
    -------
    func : array, double
        The complex transcendental equation
    """
    left = np.tan(k_n*L)
    zeta_prod = zeta[0]*zeta[1]
    zeta_sum = zeta[0]+zeta[1]

    right = \
        (1j*k*zeta_sum) / (k_n * (zeta_prod + (k/k_n)**2))
    func = left - right

    return func


def gradient_trancendental_equation_eigenfrequencies_impedance(
        k_n, k, L, zeta):
    r"""The gradient of the  transcendental equation for the estimation of the
    complex eigenfrequencies of the rectangular room with uniform impedances.
    This function is intended as the analytic jacobian function for solving
    for the roots of the transcendental equation.

    .. math::

        L \left(\tan^{2}{\left(L k_{n} \right)} + 1\right) -
        \frac{2 i k^{3} \left(\zeta_{0} + \zeta_{L}\right)}{k_{n}^{4}
        \left(\frac{k^{2}}{k_{n}^{2}} + \zeta_{0} \zeta_{L}\right)^{2}} +
        \frac{ i k \left(\zeta_{0} + \zeta_{L}\right)}{k_{n}^{2}
        \left(\frac{k^{2}}{k_{n}^{2}} + \zeta_{0} \zeta_{L}\right)} = 0

    Parameters
    ----------
    k_n : complex
        The complex eigenvalue
    k : double
        The real valued wave number
    L : double
        The room dimension
    zeta : array, double
        The normalized specific impedance

    Returns
    -------
    func : array, double
        The complex transcendental equation
    """
    zeta_prod = zeta[0]*zeta[1]
    zeta_sum = zeta[0]+zeta[1]

    tan = L * (np.tan(L * k_n)**2 + 1)
    denom = (k_n**2 * ((k/k_n)**2 + zeta_prod))
    left = k**3 / denom**2
    right = k / denom

    d_k_n = tan + (-2*left + right)*zeta_sum*1j
    return d_k_n


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
        L_l, ks, k_max, zeta, gradient=True):
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
    gradient : boolean, optional (True)
        Use the analytic gradient of the transcendental equation instead
        of an numerical approximation in the solver

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

    if gradient:
        fprime = gradient_trancendental_equation_eigenfrequencies_impedance
    else:
        fprime = False

    k_ns_l = np.zeros((n_l_max, len(ks)), dtype=complex)
    k_n_init = initial_solution_transcendental_equation(ks[0], L_l, zeta)
    for idx_k, k in enumerate(ks):
        idx_n = 0
        while k_n_init.real < k_max:
            args_costfun = (k, L_l, zeta)
            kk_n = optimize.newton(
                transcendental_equation_eigenfrequencies_impedance_newton,
                k_n_init,
                fprime=fprime,
                args=args_costfun)
            if kk_n.real > k_max:
                break
            else:
                k_ns_l[idx_n, idx_k] = kk_n
                k_n_init = (kk_n*L_l + np.pi) / L_l
                idx_n += 1

        k_n_init = k_ns_l[0, idx_k]

    return k_ns_l


def normal_eigenfrequencies_rectangular_room_impedance(
        L, ks, k_max, zeta):
    r"""Caller function for the eigenvalue estimation of all room dimensions.
    See the function `eigenfrequencies_rectangular_room_impedance` or
    `eigenfrequencies_rectangular_room_1d` for more information.

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
    k_ns : list, complex
        List of arrays with the complex eigenvalues for each wavenumber and
        each room dimension.
    """
    k_ns = []
    for dim, L_l, zeta_l in zip(count(), L, zeta):
        k_ns_l = eigenfrequencies_rectangular_room_1d(
            L_l, ks, k_max, zeta_l)
        k_ns.append(k_ns_l)
    return k_ns


def eigenfrequencies_rectangular_room_impedance(
        L, ks, k_max, zeta, only_normal=False):
    r"""Estimates the complex eigenvalues in the wavenumber domain for a
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
    only_normal : boolean, optional (False)
        Only return the eigenvalues corresponding to the axial modes.
        The mode indices will still contain the indices for all modes in
        the defined frequency range. The complete set of eigenvalues
        can be calculated as
        :math:`k_n = \sqrt{ k_{n,x}^2+k_{n,y}^2+k_{n,z}^2 }`.

    Returns
    -------
    k_ns : array, complex
        The complex eigenvalues for each wavenumber
    mode_indices : array, integer
        The wave number indices of respective eigenvalues.

    Note
    ----
    Eigenvalues smaller for a wave number :math:`k < 0.02` will be replaced by
    the value for the closest larger wave number to ensure finding the root.

    """
    ks = np.atleast_1d(ks)
    mask = ks >= 0.02
    ks_search = ks[mask]
    k_ns = normal_eigenfrequencies_rectangular_room_impedance(
        L, ks_search, k_max, zeta
    )
    for idx in range(0, len(L)):
        k_ns[idx] = np.hstack((
            np.tile(k_ns[idx][:, 0], (np.sum(~mask), 1)).T,
            k_ns[idx]))

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

    if only_normal:
        kk_ns = k_ns
    else:
        kk_ns = kk_ns[mask_bc].reshape(-1, len(ks))

    mode_indices = perms[mask_bc[:, 0]]

    return kk_ns, mode_indices


def mode_function_impedance(position, eigenvalue, phase):
    r""" The modal function for a room with boundary impedances [4]_ .

    .. math::

        p_{n,i}(x_i) = \cosh(x_i k_{n,i} + \phi_{n,i})

    Parameters
    ----------
    position : ndarray, double, (3,)
        The position in Cartesian coordinates
    eigenvalue : ndarray, complex, (3, N, n_bins)
        The N complex eigenvalues in x,y,z coordinates for each
        frequency bin (wavenumber)
    phase : ndarray, complex, (3, N, n_bins)
        The phase shift introduced by the boundary impedance

    Returns
    -------
    p_n : ndarray, complex, (3, N, n_bins)
        The modal function

    References
    ----------
    ..  [4] M. Nolan and J. L. Davy, “Two definitions of the inner product of
        modes and their use in calculating non-diffuse reverberant sound
        fields,” The Journal of the Acoustical Society of America, vol.
        145, no. 6, pp. 3330–3340, Jun. 2019.
    """
    return np.cosh(1j*eigenvalue * position + phase)


def pressure_modal_superposition(
        ks, omegas, k_ns, mode_indices, r_R, r_S, L, zeta):
    r""" Calculate modal composition for a rectangular room with arbitrary
    boundary impedances.

    Parameters
    ----------
    ks : ndarray, double
        The wave number array
    omegas : ndarray, double
        The angular frequency array :math:`omega`
    k_ns : list, complex
        List containing the complex eigenvalues for each dimension and
        wavenumber
    r_R : ndarray, double, (3, n_receivers)
        The receiver positions in Cartesian coordinates
    r_S : ndarray, double, (3)
        The source position in Cartesian coordinates
    L : ndarray, double, (3,)
        The room dimensions in meters
    zeta : ndarray, double, (3, 2)
        The normalized impedance :math:`\zeta_i = \frac{Z_i}{\rho_o c}`
        for each wall.

    Returns
    -------
    spec : ndarray, complex
        The complex sum of all mode functions corresponding to the eigenvalues
        `k_ns`

    """

    zeta_0 = zeta[:, 0]
    r_R = np.atleast_2d(r_R)

    kk_ns = np.sqrt(
        k_ns[0][mode_indices[:, 0]]**2 +
        k_ns[1][mode_indices[:, 1]]**2 +
        k_ns[2][mode_indices[:, 2]]**2)

    k_ns_xyz = np.array([
        k_ns[0][mode_indices[:, 0]],
        k_ns[1][mode_indices[:, 1]],
        k_ns[2][mode_indices[:, 2]]
        ])

    phi = np.arctanh(ks/(zeta_0 * k_ns_xyz.T).T)
    K_n_sc = \
        np.sinh(1j * k_ns_xyz.T * L) * \
        np.cosh(1j * k_ns_xyz.T * L + 2*phi.T)

    K_n = np.prod((L/2 * (1 + 1/(1j*k_ns_xyz.T * L) * K_n_sc)).T, axis=0)
    denom = K_n * (kk_ns**2 - ks**2)

    p_ns_s = np.prod(mode_function_impedance(r_S, k_ns_xyz.T, phi.T).T, axis=0)

    spec = np.zeros((r_R.shape[0], ks.size), dtype=complex)
    for idx_R in range(r_R.shape[0]):
        p_ns_r = np.prod(
            mode_function_impedance(
                r_R[idx_R, :], k_ns_xyz.T, phi.T).T,
            axis=0)

        nom = 1j*omegas*1.2*p_ns_r*p_ns_s

        spec[idx_R, :] = np.sum(nom / denom, axis=0)

    spec = np.squeeze(spec)

    return spec


def rectangular_room_impedance(
        L,
        r_S,
        r_R,
        normalized_impedance,
        max_freq,
        samplingrate=44100,
        c=343.9,
        n_samples=2**12,
        remove_cavity_mode=False):
    r""" Calculate the room impulse response and room transfer function for a
    rectangular room with arbitrary boundary impedances.

    Parameters
    ----------
    L : ndarray, double, (3,)
        The room dimensions in meters
    r_S : ndarray, double, (3)
        The source position in Cartesian coordinates
    r_R : ndarray, double, (3, n_receivers)
        The receiver positions in Cartesian coordinates
    normalized_impedance : ndarray, double, (3, 2)
        The normalized impedance :math:`\zeta_i = \frac{Z_i}{\rho_o c}`
        for each wall.
    max_freq : double
        The highest frequency to be considered for the estimation of the
        eigenfrequencies.
    samplingrate : int, 44100
        The samplingrate
    c : float, 343.9
        The speed of sound in m/s
    n_samples : int, 2**12
        The number of samples for which the RIR is calculated
    remove_cavity_mode : boolean, False
        When true, the cavity mode (0, 0, 0) will be removed before summation
        of all modes

    Returns
    -------
    rir : ndarray, double, (n_receivers, n_samples)
        The room impulse response
    rtf : ndarray, double, (n_receivers, n_bins)
        The room transfer function in the frequency domain
    eigenvalues : ndarray, complex
        The complex eigenvalues in the form
        :math:`k_n = \omega_n / c + i \delta_n`
    """

    zeta = normalized_impedance
    freqs = np.fft.rfftfreq(n_samples, 1/samplingrate)
    ks = 2*np.pi*freqs/c

    k_max = max_freq*2*np.pi/c
    k_ns, mode_indices = eigenfrequencies_rectangular_room_impedance(
        L, ks, k_max, zeta, only_normal=True)

    if remove_cavity_mode:
        mask = np.prod(np.array([0, 0, 0]) == mode_indices, axis=-1) == 1
        mode_indices = mode_indices[~mask]

    spectrum = pressure_modal_superposition(
        ks, freqs*2*np.pi, k_ns, mode_indices, r_R, r_S, L, zeta)

    rir = np.fft.irfft(spectrum)

    k_ns_xyz = np.array([
        k_ns[0][mode_indices[:, 0]],
        k_ns[1][mode_indices[:, 1]],
        k_ns[2][mode_indices[:, 2]]
        ])

    return rir, spectrum, k_ns_xyz
