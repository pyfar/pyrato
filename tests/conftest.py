"""
Tests for roomacoustic parameters and related functions.
"""
import numpy as np
import pytest
import pyfar as pf

@pytest.fixture
def make_edc_from_energy():
    """
    Pytest fixture that provides a factory function to create a
    normalized Energy Decay Curve (EDC) as a `pyfar.TimeData` object.

    If no arguments are passed, a default simple linear-decay EDC is used.
    The factory can also generate predefined cases ("geometric", "real")
    or arbitrary energy arrays for flexible test setups.

    Examples
    --------
    >>> edc = make_edc_from_energy()  # uses default simple linear decay
    >>> edc = make_edc_from_energy(case="geometric")
    >>> edc = make_edc_from_energy(energy=my_curve, sampling_rate=48000)
    """
    default_energy = np.linspace(1, 0, 1000)
    default_samplerate = 1000

    def _factory(*, energy=None, case=None, sampling_rate=None):
        # Priority: explicit energy > case > default simple
        if energy is not None:
            energy = np.asarray(energy, dtype=float)
        elif case == "geometric":
            decay_factor = 0.9
            energy = decay_factor ** (2 * np.arange(200))
        elif case == "real":
            energy = np.array([
                1.00000000e+00, 8.39817186e-01, 7.05292906e-01, 5.92317103e-01,
                4.97438083e-01, 4.17757051e-01, 3.50839551e-01, 2.94641084e-01,
                2.47444646e-01, 2.07808266e-01, 1.74520953e-01, 1.46565696e-01,
                1.23088390e-01, 1.03371746e-01, 8.68133684e-02, 7.29073588e-02,
                6.12288529e-02, 5.14210429e-02, 4.31842755e-02, 3.62668968e-02,
                3.04575632e-02, 2.55787850e-02, 2.14815032e-02, 1.80405356e-02,
                1.51507518e-02, 1.27238618e-02, 1.06857178e-02, 8.97404943e-03,
                7.53656094e-03, 6.32933340e-03, 5.31548296e-03, 4.46403394e-03,
                3.74897242e-03, 3.14845147e-03, 2.64412365e-03, 2.22058049e-03,
                1.86488165e-03, 1.56615966e-03, 1.31528780e-03, 1.10460130e-03,
                9.27663155e-04, 7.79067460e-04, 6.54274242e-04, 5.49470753e-04,
                4.61454981e-04, 3.87537824e-04, 3.25460924e-04, 2.73327678e-04,
                2.29545281e-04, 1.92776072e-04,
            ])
        else:  # Default case: simple linear decay
            energy = default_energy

        sampling_rate = sampling_rate or default_samplerate

        # Normalize (avoid division by zero)
        if np.max(energy) != 0:
            energy = energy / np.max(energy)

        times = np.arange(energy.shape[-1]) / sampling_rate
        return pf.TimeData(energy, times)

    return _factory
