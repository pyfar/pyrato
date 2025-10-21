import numpy as np
import pytest
import pyfar as pf
from typing import Optional


@pytest.fixture
def make_edc_from_energy():
    """
    Fixture providing a factory for creating Energy Decay Curves (EDCs)
    as `pyfar.TimeData` objects.

    The factory supports either:
    - providing a custom energy array, or
    - generating an exponential EDC from a specified reverberation time (RT60).
    - producing a baseline EDC with minimal energy
    when neither input is supplied.

    Parameters
    ----------
    energy : array_like, optional
        Custom energy decay curve. If provided, this overrides
        all other inputs.
    rt : float, optional
        Reverberation time (RT60) in seconds. Generates an exponential energy
        decay curve E(t) = exp(-13.8155 * t / RT60), corresponding to -60 dB.
    sampling_rate : float, optional
        Sampling rate in Hz. Defaults to 1000.
    total_samples : int, optional
        Number of samples in the generated EDC. Defaults to 1000.
    normalize : bool, optional
        Normalize energy curve to its maximum value. Defaults to True.
    dynamic_range : float, optional
        Dynamic range limit in decibels below the peak (default: 65 dB).

    Returns
    -------
    callable
        Factory function returning a `pyfar.TimeData` object.
    """

    def _factory(
        *,
        energy: Optional[np.ndarray] = None,
        rt: Optional[float] = None,
        sampling_rate: float = 1000.0,
        total_samples: int = 1000,
        normalize: bool = True,
        dynamic_range: float = 65.0,
    ) -> pf.TimeData:
        """Construct a pyfar.TimeData object representing an EDC."""
        if energy is not None and not isinstance(energy, (np.ndarray,
                                                          list,
                                                          tuple)):
            raise TypeError("energy must be array-like or None")
        if rt is not None and (not np.isscalar(rt) or rt <= 0):
            raise ValueError("rt must be a positive scalar")

        # Lower energy threshold based on dynamic range
        min_energy = 10 ** (-dynamic_range / 10)

        if energy is not None:
            energy = np.asarray(energy, dtype=float)
        elif rt is not None:
            times = np.arange(total_samples) / sampling_rate
            # Exponential energy decay: -60 dB after RT60 seconds
            energy = np.exp(-13.8155 * times / rt)
        else:
            energy = np.zeros(total_samples)

        # Optional normalization
        if normalize and np.max(energy) > 0:
            energy = energy / np.max(energy)

        energy[energy < min_energy] = min_energy

        times = np.arange(energy.shape[-1]) / sampling_rate
        return pf.TimeData(energy, times)

    return _factory
