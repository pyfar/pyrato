import numpy as np
import matplotlib.pyplot as plt

import roomacoustics.analytic

L = np.array([8, 5, 3])/10
source_pos = np.array([5, 3, 1.2])/10
receiver_pos = [0, 0, 0]

rir, res, freqs = roomacoustics.analytic.rectangular_room_rigid_walls(
        L,
        source_pos,
        receiver_pos,
        1,
        max_freq=10000,
        n_samples=2**18)
