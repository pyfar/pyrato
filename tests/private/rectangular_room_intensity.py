# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import roomacoustics.analytic as analytic

import matplotlib.pyplot as plt


# %%
L = np.array([8, 5, 3])/10
zetas = np.ones((3, 2)) * 100

c = 343.9

r_R = np.array([3.3, 1.6, 1.8])/10
r_S = np.array([5.3, 3.6, 1.2])/10

f_max = 1e3

samplingrate = 2200
n_samples = 2**10
freqs = np.fft.rfftfreq(n_samples, 1/samplingrate)
ks = 2*np.pi*freqs/c


# %%
k_ns, mode_idx = analytic.eigenfrequencies_rectangular_room_impedance(
    L, ks, f_max*2*np.pi/c, zetas, only_normal=True)


# %%
mask = np.prod(np.array([0,0,0]) == mode_idx, axis=-1) == 1


# %%
rir, spec, k_ns = analytic.rectangular_room_impedance(
    L, r_S, r_R, zetas, f_max, samplingrate, c, n_samples, remove_cavity_mode=False)


# %%
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(freqs, 20*np.log10(np.abs(spec)))
ax = plt.gca()
ax.set_ylabel("Magnitude [dB]")
ax.set_xlabel("Frequency [Hz]")
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(freqs, (np.angle(spec)))
plt.grid(True)
ax = plt.gca()
ax.set_ylabel("Phase [rad]")
ax.set_xlabel("Frequency [Hz]");


# %%
times = np.arange(0, n_samples)/samplingrate

plt.figure(figsize=(12, 4))
plt.plot(times, 20*np.log10(np.abs(rir)))
ax = plt.gca()
ax.set_ylabel("Magnitude [dB]")
ax.set_xlabel("Time [s]")
plt.grid(True)


# %%
np.save('../data/analytic_rtf_impedance.npy', spec)
np.save('../data/analytic_rir_impedance.npy', rir)


# %%
rir_rem, spec_rem, k_ns_rem = analytic.rectangular_room_impedance(
    L, r_S, r_R, zetas, f_max, samplingrate, c, n_samples, remove_cavity_mode=True)


# %%
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(freqs, 20*np.log10(np.abs(spec)))
plt.plot(freqs, 20*np.log10(np.abs(spec_rem)))
ax = plt.gca()
ax.set_ylabel("Magnitude [dB]")
ax.set_xlabel("Frequency [Hz]")
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(freqs, (np.angle(spec)))
plt.plot(freqs, (np.angle(spec_rem)))
plt.grid(True)
ax = plt.gca()
ax.set_ylabel("Phase [rad]")
ax.set_xlabel("Frequency [Hz]");


# %%
times = np.arange(0, n_samples)/samplingrate

plt.figure(figsize=(12, 4))
plt.plot(times, 20*np.log10(np.abs(rir)))
plt.plot(times, 20*np.log10(np.abs(rir_rem)))
ax = plt.gca()
ax.set_ylabel("Magnitude [dB]")
ax.set_xlabel("Time [s]")
plt.grid(True)


# %%
np.save('../data/analytic_rtf_impedance_no_cav.npy', spec_rem)
np.save('../data/analytic_rir_impedance_no_cav.npy', rir_rem)


