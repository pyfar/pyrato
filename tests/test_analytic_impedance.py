import numpy as np
import numpy.testing as npt
import pyrato.analytic.impedance as analytic


def test_eigenfreq_impedance_1d_real():
    L = 8/10
    zeta = np.ones(2) * 1e10
    c = 343.9

    k = [0.1]
    k_max = 1e3*2*np.pi/c

    k_ns = analytic.eigenfrequencies_rectangular_room_1d(
        L, k, k_max, zeta
    )

    truth = np.array([
        1e-6,
        2.149375000000000,
        4.298750000000000,
        6.448125000000000,
        8.597500000000000]) * 1e2

    f_n = np.squeeze(c*k_ns.real/2/np.pi)
    npt.assert_allclose(f_n, truth, rtol=1e-3, atol=1e-3)


def test_eigenfreq_impedance_1d_real_jac():
    L = 8/10
    zeta = np.ones(2) * 1e10
    c = 343.9

    k = [0.1]
    k_max = 1e3*2*np.pi/c

    k_ns = analytic.eigenfrequencies_rectangular_room_1d(
        L, k, k_max, zeta, gradient=True
    )

    truth = np.array([
        1e-6,
        2.149375000000000,
        4.298750000000000,
        6.448125000000000,
        8.597500000000000]) * 1e2

    f_n = np.squeeze(c*k_ns.real/2/np.pi)
    npt.assert_allclose(f_n, truth, rtol=1e-3, atol=1e-3)


def test_analytic_shoebox_eigenfreqs_impedance_multi_k():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 1e10

    c = 343.9

    k_max = 1e3*2*np.pi/c
    k = np.linspace(0, k_max*1.1, 2**10)

    k_ns, _ = analytic.eigenfrequencies_rectangular_room_impedance(
            L, k, k_max, zetas
        )

    truth = np.array([
        4.196434e-05,
        5.731666666666666,
        3.439000000000000,
        6.684214522124330,
        6.877999999999999,
        8.953149545147662,
        2.149375000000000,
        6.121422683363956,
        4.055432639142833,
        7.021291666666667,
        7.206018102296510,
        9.207534939841542,
        4.298750000000000,
        7.164583333333334,
        5.505086063132890,
        7.947199213576930,
        8.110865278285665,
        9.931673491425187,
        6.448125000000000,
        8.627300782597231,
        7.307874999999999,
        9.287466812506130,
        9.427894781743429,
        8.597500000000000,
        9.259790885867778]) * 1e2

    f_n = np.mean(np.squeeze(c*k_ns.real/2/np.pi), axis=-1)
    npt.assert_allclose(np.sort(f_n), np.sort(truth), atol=1e-3, rtol=1e-3)


def test_analytic_shoebox_eigenfreqs_impedance():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 1e10

    c = 343.9

    k = [0.1]
    k_max = 1e3*2*np.pi/c

    k_ns, _ = analytic.eigenfrequencies_rectangular_room_impedance(
            L, k, k_max, zetas
        )

    truth = np.array([
        4.440943e-06,
        5.731666666666666,
        3.439000000000000,
        6.684214522124330,
        6.877999999999999,
        8.953149545147662,
        2.149375000000000,
        6.121422683363956,
        4.055432639142833,
        7.021291666666667,
        7.206018102296510,
        9.207534939841542,
        4.298750000000000,
        7.164583333333334,
        5.505086063132890,
        7.947199213576930,
        8.110865278285665,
        9.931673491425187,
        6.448125000000000,
        8.627300782597231,
        7.307874999999999,
        9.287466812506130,
        9.427894781743429,
        8.597500000000000,
        9.259790885867778]) * 1e2

    f_n = np.squeeze(c*k_ns.real/2/np.pi)
    npt.assert_allclose(np.sort(f_n), np.sort(truth), atol=1e-3, rtol=1e-3)


def test_analytic_eigenfrequencies_impedance_cplx():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 1e10

    c = 343.9

    k_max = 1e3*2*np.pi/c
    k_min = 150*2*np.pi/c
    k = np.linspace(k_min, k_max*1.1, 2**10)

    k_ns, _ = analytic.eigenfrequencies_rectangular_room_impedance(
            L, k, k_max, zetas, only_normal=True
        )

    k_ns_x = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_x.csv',
        delimiter=',',
        dtype=complex)
    k_ns_y = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_y.csv',
        delimiter=',',
        dtype=complex)
    k_ns_z = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_z.csv',
        delimiter=',',
        dtype=complex)

    npt.assert_allclose(k_ns[0], k_ns_x, rtol=1e-6)
    npt.assert_allclose(k_ns[1], k_ns_y, rtol=1e-6)
    npt.assert_allclose(k_ns[2], k_ns_z, rtol=1e-6)


def test_analytic_eigenfrequencies_impedance_zeta15():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 15

    c = 343.9

    k_max = 1e3*2*np.pi/c
    k_min = 150*2*np.pi/c
    k = np.linspace(k_min, k_max*1.1, 2**10)

    k_ns, _ = analytic.eigenfrequencies_rectangular_room_impedance(
            L, k, k_max, zetas, only_normal=True
        )

    k_ns_x = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_x_zeta15.csv',
        delimiter=',',
        dtype=complex)
    k_ns_y = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_y_zeta15.csv',
        delimiter=',',
        dtype=complex)
    k_ns_z = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_z_zeta15.csv',
        delimiter=',',
        dtype=complex)

    npt.assert_allclose(k_ns[0], k_ns_x, rtol=1e-6)
    npt.assert_allclose(k_ns[1], k_ns_y, rtol=1e-6)
    npt.assert_allclose(k_ns[2], k_ns_z, rtol=1e-6)


def test_analytic_pressure_shoebox_impedance():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 1e10

    c = 343.9

    k_max = 1e3*2*np.pi/c
    k_min = 150*2*np.pi/c
    k = np.linspace(k_min, k_max*1.1, 2**10)

    k_ns_x = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_x.csv',
        delimiter=',',
        dtype=complex)
    k_ns_y = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_y.csv',
        delimiter=',',
        dtype=complex)
    k_ns_z = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_z.csv',
        delimiter=',',
        dtype=complex)

    k_ns = list((k_ns_x, k_ns_y, k_ns_z))

    mode_indices = np.loadtxt(
        'tests/data/analytic_impedance/mode_indices.csv',
        delimiter=',',
        dtype=int)

    r_R = np.array([3.3, 1.6, 1.8])/10
    r_S = np.array([5.3, 3.6, 1.2])/10

    p_x = analytic.pressure_modal_superposition(
        k, k*c, k_ns, mode_indices, r_R, r_S, L, zetas)

    truth = np.loadtxt(
        'tests/data/analytic_impedance/p_x.csv',
        delimiter=',',
        dtype=complex)

    npt.assert_allclose(p_x, truth, atol=1e-6, rtol=1e-6)


def test_analytic_pressure_shoebox_impedance_multi_R():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 1e10

    c = 343.9

    k_max = 1e3*2*np.pi/c
    k_min = 150*2*np.pi/c
    k = np.linspace(k_min, k_max*1.1, 2**10)

    k_ns_x = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_x.csv',
        delimiter=',',
        dtype=complex)
    k_ns_y = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_y.csv',
        delimiter=',',
        dtype=complex)
    k_ns_z = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_z.csv',
        delimiter=',',
        dtype=complex)

    k_ns = list((k_ns_x, k_ns_y, k_ns_z))

    mode_indices = np.loadtxt(
        'tests/data/analytic_impedance/mode_indices.csv',
        delimiter=',',
        dtype=int)

    r_R = np.array([
        [3.3, 1.6, 1.8],
        [3.3, 1.6, 1.8]])/10
    r_S = np.array([5.3, 3.6, 1.2])/10

    p_x = analytic.pressure_modal_superposition(
        k, k*c, k_ns, mode_indices, r_R, r_S, L, zetas)

    truth = np.loadtxt(
        'tests/data/analytic_impedance/p_x.csv',
        delimiter=',',
        dtype=complex)

    truth = np.vstack((truth, truth))

    npt.assert_allclose(p_x, truth, atol=1e-6, rtol=1e-6)


def test_analytic_pressure_shoebox_impedance_zeta15():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 15

    c = 343.9

    k_max = 1e3*2*np.pi/c
    k_min = 150*2*np.pi/c
    k = np.linspace(k_min, k_max*1.1, 2**10)

    k_ns_x = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_x_zeta15.csv',
        delimiter=',',
        dtype=complex)
    k_ns_y = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_y_zeta15.csv',
        delimiter=',',
        dtype=complex)
    k_ns_z = np.loadtxt(
        'tests/data/analytic_impedance/k_ns_z_zeta15.csv',
        delimiter=',',
        dtype=complex)

    k_ns = list((k_ns_x, k_ns_y, k_ns_z))

    mode_indices = np.loadtxt(
        'tests/data/analytic_impedance/mode_indices_zeta15.csv',
        delimiter=',',
        dtype=int)

    r_R = np.array([3.3, 1.6, 1.8])/10
    r_S = np.array([5.3, 3.6, 1.2])/10

    p_x = analytic.pressure_modal_superposition(
        k, k*c, k_ns, mode_indices, r_R, r_S, L, zetas)

    truth = np.loadtxt(
        'tests/data/analytic_impedance/p_x_zeta15.csv',
        delimiter=',',
        dtype=complex)

    npt.assert_allclose(p_x, truth, atol=1e-4, rtol=1e-4)


def test_analytic_shoebox_spec_impedance():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 100

    c = 343.9

    r_R = np.array([3.3, 1.6, 1.8])/10
    r_S = np.array([5.3, 3.6, 1.2])/10

    f_max = 1e3

    samplingrate = 2200
    n_samples = 2**10

    rir, spec, k_ns = analytic.rectangular_room_impedance(
        L,
        r_S,
        r_R,
        zetas,
        f_max,
        samplingrate,
        c,
        n_samples,
        remove_cavity_mode=False)

    truth_rtf = np.load('tests/data/analytic_rtf_impedance.npy')
    truth_rir = np.load('tests/data/analytic_rir_impedance.npy')

    npt.assert_allclose(spec, truth_rtf, atol=1e-2, rtol=1e-2)
    npt.assert_allclose(rir, truth_rir, atol=1e-2, rtol=1e-2)


def test_analytic_shoebox_spec_impedance_no_cavity_mode():
    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 100

    c = 343.9

    r_R = np.array([3.3, 1.6, 1.8])/10
    r_S = np.array([5.3, 3.6, 1.2])/10

    f_max = 1e3

    samplingrate = 2200
    n_samples = 2**10

    rir, spec, k_ns = analytic.rectangular_room_impedance(
        L,
        r_S,
        r_R,
        zetas,
        f_max,
        samplingrate,
        c,
        n_samples,
        remove_cavity_mode=True)

    truth_rtf = np.load('tests/data/analytic_rtf_impedance_no_cav.npy')
    truth_rir = np.load('tests/data/analytic_rir_impedance_no_cav.npy')

    npt.assert_allclose(spec, truth_rtf, atol=1e-2, rtol=1e-2)
    npt.assert_allclose(rir, truth_rir, atol=1e-2, rtol=1e-2)
