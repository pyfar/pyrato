import numpy as np
import numpy.testing as npt
import pyrato.analytic as analytic
import pyfar as pf


def test_analytic_shoebox_eigenfreqs():
    L = np.array([8, 5, 3])/10

    eigenfreqs, _ = \
        analytic.eigenfrequencies_rectangular_room_rigid(
            L,
            max_freq=1e3,
            speed_of_sound=343.9,
            sort=False)

    f_n = np.array([
        0,
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

    ref = pf.Signal(ref, 44100)

    npt.assert_allclose(rir.time, ref.time)
