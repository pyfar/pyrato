import numpy as np
import numpy.testing as npt
import roomacoustics.analytic as analytic


def test_analytic_shoebox_eigenfreqs():
    L = np.array([8, 5, 3])/10

    eigenfreqs, _ = \
        analytic.eigenfrequencies_rectangular_room_rigid(
            L,
            max_freq=1e3,
            speed_of_sound=343.9)

    f_n = np.array([0,
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

    npt.assert_allclose(eigenfreqs, f_n)


def test_analytic_shoebox_rir():
    L = np.array([8, 5, 3])/10
    source_pos = np.array([5, 3, 1.2])/10
    receiver_pos = [1, 0, 0]

    rir, _ = analytic.rectangular_room_rigid_walls(
        L,
        source_pos,
        receiver_pos,
        1,
        max_freq=1e3,
        n_samples=2**16,
        speed_of_sound=343.9)

    ref = np.loadtxt(
        './tests/data/analytic_rir_rigid_walls.csv',
        delimiter=',',
        skiprows=1)

    npt.assert_allclose(rir, ref)


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



def test_analytic_pressure_shoebox_impedance():

    # import matplotlib.pyplot as plt

    L = np.array([8, 5, 3])/10
    zetas = np.ones((3, 2)) * 1e10

    c = 343.9

    k_max = 1e3*2*np.pi/c
    k_min = 150*2*np.pi/c
    k = np.linspace(k_min, k_max*1.1, 2**10)

    k_ns, mode_indices = analytic.eigenfrequencies_rectangular_room_impedance(
            L, k, k_max, zetas, only_normal=True
        )

    r_R = np.array([3.3, 1.6, 1.8])/10
    r_S = np.array([5.3, 3.6, 1.2])/10

    p_x = analytic.pressure_modal_superposition(
        k, k*c, k_ns, mode_indices, r_R, r_S, L, zetas)

    kk_ns = np.sqrt(
        k_ns[0][mode_indices[:, 0]]**2 +
        k_ns[1][mode_indices[:, 1]]**2 +
        k_ns[2][mode_indices[:, 2]]**2)

    f_n = np.mean(np.squeeze(c*kk_ns.real/2/np.pi), axis=-1)
    print(f_n)

    truth = 0
    npt.assert_allclose(p_x, truth, atol=1e-3, rtol=1e-3)
