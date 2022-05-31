#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The generate_test_data module provides the functionallity to generate
    adequate test data for automatic testing.
"""

__author__ = "Johannes Imort"
__version__ = "0.1.0"
__maintainer__ = "Johannes Imort"
__email__ = "johannes.imort@rwth-aachen.de"
__status__ = "Development"

# %%
import numpy as np
import pyrato
from pyrato import analytic


sampling_rate = 3000
room_dimensions = [1, 1, 1]
src_pos = [0.2, 0.3, 0.4]
rec_pos = [0.5, 0.6, 0.7]
t_60 = 1
max_freq = 1000
n_samples = 2**13

rir_1 = analytic.rectangular_room_rigid_walls(
    dimensions=room_dimensions, source=src_pos, receiver=rec_pos, reverberation_time=t_60, max_freq=max_freq,
    samplingrate=sampling_rate, speed_of_sound=343.9, n_samples=n_samples)[0]

rir_2 = analytic.rectangular_room_rigid_walls(
    dimensions=room_dimensions, source=src_pos, receiver=rec_pos, reverberation_time=t_60*2, max_freq=max_freq,
    samplingrate=sampling_rate, speed_of_sound=343.9, n_samples=n_samples*2)[0]

rir_array = np.zeros(([2, rir_2.size]))

psnr = 50

rir_1 /= np.amax(np.abs(rir_1))
rir_2 /= np.amax(np.abs(rir_2))

rir_array[0,:rir_1.shape[0]] = rir_1
rir_array[1,:] = rir_2

# fix the seed
np.random.seed(1)
noise = np.random.normal(0, 1, rir_2.size)

rir_array += 10**(-(psnr-10) / 20) * noise
# %%

noise_energy_1D = pyrato.estimate_noise_energy(
    rir_array[0], interval=[0.9, 1.0], is_energy=False)
noise_energy_2D = pyrato.estimate_noise_energy(
    rir_array, interval=[0.9, 1.0], is_energy=False)

preprocessing_1D = pyrato.edc.preprocess_rir(
    rir_array[0], is_energy=False, time_shift=False, channel_independent=False)
preprocessing_2D = pyrato.edc.preprocess_rir(
    rir_array, is_energy=False, time_shift=False, channel_independent=False)

preprocessing_time_shift_1D = pyrato.edc.preprocess_rir(
    rir_array[0], is_energy=False, time_shift=True, channel_independent=False)
preprocessing_time_shift_2D = pyrato.edc.preprocess_rir(
    rir_array, is_energy=False, time_shift=True, channel_independent=False)

preprocessing_time_shift_channel_independent_1D = pyrato.edc.preprocess_rir(
    rir_array[0], is_energy=False, time_shift=True, channel_independent=True)
preprocessing_time_shift_channel_independent_2D = pyrato.edc.preprocess_rir(
    rir_array, is_energy=False, time_shift=True, channel_independent=True)

smoothed_rir_1D = pyrato.edc.smooth_rir(rir_array[0], sampling_rate, smooth_block_length=0.075)
smoothed_rir_2D = pyrato.edc.smooth_rir(rir_array, sampling_rate, smooth_block_length=0.075)

substracted_1D = pyrato.edc.subtract_noise_from_squared_rir(rir_array[0]**2)
substracted_2D = pyrato.edc.subtract_noise_from_squared_rir(rir_array**2)

edc_truncation_1D = pyrato.energy_decay_curve_truncation(
    rir_array[0], sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True)
edc_truncation_2D = pyrato.energy_decay_curve_truncation(
    rir_array, sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True)

edc_lundeby_1D = pyrato.energy_decay_curve_lundeby(
    rir_array[0], sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True, plot=False)
edc_lundeby_2D = pyrato.energy_decay_curve_lundeby(
    rir_array, sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True, plot=False)

edc_lundeby_chu_1D = pyrato.energy_decay_curve_chu_lundeby(
    rir_array[0], sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True, plot=False)
edc_lundeby_chu_2D = pyrato.energy_decay_curve_chu_lundeby(
    rir_array, sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True, plot=False)

edc_chu_1D = pyrato.energy_decay_curve_chu(
    rir_array[0], sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True, plot=False)
edc_chu_2D = pyrato.energy_decay_curve_chu(
    rir_array, sampling_rate, freq='broadband', is_energy=False, time_shift=True,
    channel_independent=False, normalize=True, plot=False)

intersection_time_1D = pyrato.intersection_time_lundeby(
    rir_array[0], sampling_rate, freq='broadband', is_energy=False, time_shift=False,
    channel_independent=False, plot=False)
intersection_time_2D = pyrato.intersection_time_lundeby(
    rir_array, sampling_rate, freq='broadband', is_energy=False, time_shift=False,
    channel_independent=False, plot=False)

# noise_energy_from_edc_1D = pyrato.edc.estimate_noise_energy_from_edc(
#     edc_lundeby_chu_1D, intersection_time_1D[0], sampling_rate)
# noise_energy_from_edc_2D = pyrato.edc.estimate_noise_energy_from_edc(
#     edc_lundeby_chu_2D, intersection_time_2D[0], sampling_rate)

# %%


np.savetxt("analytic_rir_psnr50_1D.csv", rir_array[0], delimiter=",")
np.savetxt("analytic_rir_psnr50_2D.csv", rir_array, delimiter=",")

np.savetxt("noise_energy_1D.csv", noise_energy_1D, delimiter=",")
np.savetxt("noise_energy_2D.csv", noise_energy_2D, delimiter=",")

np.savetxt("preprocessing_1D.csv", preprocessing_1D[0], delimiter=",")
np.savetxt("preprocessing_2D.csv", preprocessing_2D[0], delimiter=",")

np.savetxt("preprocessing_time_shift_1D.csv", preprocessing_time_shift_1D[0], delimiter=",")
np.savetxt("preprocessing_time_shift_2D.csv", preprocessing_time_shift_2D[0], delimiter=",")

np.savetxt("preprocessing_time_shift_channel_independent_1D.csv", preprocessing_time_shift_channel_independent_1D[0], delimiter=",")
np.savetxt("preprocessing_time_shift_channel_independent_2D.csv", preprocessing_time_shift_channel_independent_2D[0], delimiter=",")

np.savetxt("smoothed_rir_1D.csv", smoothed_rir_1D[0], delimiter=",")
np.savetxt("smoothed_rir_2D.csv", smoothed_rir_2D[0], delimiter=",")

np.savetxt("substracted_1D.csv", substracted_1D, delimiter=",")
np.savetxt("substracted_2D.csv", substracted_2D, delimiter=",")

np.savetxt("edc_truncation_1D.csv", edc_truncation_1D, delimiter=",")
np.savetxt("edc_truncation_2D.csv", edc_truncation_2D, delimiter=",")

np.savetxt("edc_lundeby_1D.csv", edc_lundeby_1D, delimiter=",")
np.savetxt("edc_lundeby_2D.csv", edc_lundeby_2D, delimiter=",")

np.savetxt("edc_lundeby_chu_1D.csv", edc_lundeby_chu_1D, delimiter=",")
np.savetxt("edc_lundeby_chu_2D.csv", edc_lundeby_chu_2D, delimiter=",")

np.savetxt("edc_chu_1D.csv", edc_chu_1D, delimiter=",")
np.savetxt("edc_chu_2D.csv", edc_chu_2D, delimiter=",")

np.savetxt("intersection_time_1D.csv", intersection_time_1D, delimiter=",")
np.savetxt("intersection_time_2D.csv", intersection_time_2D, delimiter=",")

#     np.savetxt("noise_energy_from_edc_1D.csv", noise_energy_from_edc_1D, delimiter=",")
#     np.savetxt("noise_energy_from_edc_2D.csv", noise_energy_from_edc_2D, delimiter=",")
