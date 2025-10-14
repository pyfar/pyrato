import numpy as np
import pytest
import pyfar as pf
from typing import Optional



@pytest.fixture
def make_edc_from_energy():
    """
    Fixture that provides a factory to create normalized
    Energy Decay Curves (EDCs) as `pyfar.TimeData` objects.

    The factory supports multiple predefined analytical cases and
    arbitrary user-defined input. It is designed for flexible testing
    of room acoustic metrics.

    Parameters
    ----------
    energy : array_like, optional
        Custom energy decay curve (any shape). If provided, this overrides
        all other case options.
    case : {"simple", "geometric", "eyring analytical", "sabine analytical"}
    , optional
        Selects a predefined energy decay profile:
        - "simple": Linear decay from 1 → 0 (default)
        - "geometric": Exponential geometric decay with configurable
          ``decay_factor`` and a fixed length of 200 samples
        - "eyring analytical": Calculated Eyring EDC from test_edc.py
        - "sabine analytical": Calculated Sabine EDC from test_edc.py
    sampling_rate : float, optional
        Sampling rate in Hz (default: 1000).
    decay_factor : float, optional
        Exponential decay constant for the "geometric" case (default: 0.9).
    total_samples : int, optional
        Total number of samples for "geometric" case impulse response
        (default: 200).
    normalize : bool, optional
        Whether to normalize the energy curve to its overall maximum
        (default: True).

    Returns
    -------
    callable
        A factory function returning a `pyfar.TimeData` object
        representing the normalized EDC.
    """
    default_energy = np.linspace(1, 0, 1000)
    default_samplerate = 1000


    def _factory(
        *,
        energy: Optional[np.ndarray] = None,
        case: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        decay_factor: Optional[float] = None,
        total_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> pf.TimeData:
        """Factory to construct a pyfar.TimeData object representing an EDC."""

        if energy is not None:
            energy = np.asarray(energy, dtype=float)

        elif case == "geometric":
            decay_factor = 0.9 if decay_factor is None else decay_factor
            total_samples = 200 if total_samples is None else total_samples
            energy = decay_factor ** (2 * np.arange(total_samples))

        elif case == "eyring analytical":
            energy = np.array([
                1.00000000e00, 8.39817186e-01, 7.05292906e-01, 5.92317103e-01,
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

        elif case == "sabine analytical":
            energy = np.array([
                1.00000000e00, 8.57869258e-01, 7.35939663e-01, 6.31340013e-01,
                5.41607188e-01, 4.64628156e-01, 3.98590212e-01, 3.41938289e-01,
                2.93338346e-01, 2.51645949e-01, 2.15879324e-01, 1.85196235e-01,
                1.58874157e-01, 1.36293255e-01, 1.16921793e-01, 1.00303612e-01,
                8.60473853e-02, 7.38174065e-02, 6.33256837e-02, 5.43251573e-02,
                4.66038824e-02, 3.99800380e-02, 3.42976455e-02, 2.94228957e-02,
                2.52409977e-02, 2.16534759e-02, 1.85758513e-02, 1.59356518e-02,
                1.36707058e-02, 1.17276782e-02, 1.00608146e-02, 8.63086356e-03,
                7.40415251e-03, 6.35179482e-03, 5.44900951e-03, 4.67453774e-03,
                4.01014222e-03, 3.44017773e-03, 2.95122272e-03, 2.53176324e-03,
                2.17192185e-03, 1.86322499e-03, 1.59840344e-03, 1.37122117e-03,
                1.17632849e-03, 1.00913605e-03, 8.65706791e-04, 7.42663242e-04,
                6.37107964e-04, 5.46555336e-04,
            ])

        elif case is None or case == "simple":
            energy = default_energy
        else:
            raise ValueError(
                f"Unknown EDC case '{case}'. "
                "Valid options: 'simple', 'geometric', "
                "'eyring analytical', 'sabine analytical'.",
            )

        # Normalize
        if normalize:
            max_val = np.max(energy)
            if max_val > 0:
                energy = energy / max_val

        # Build times
        sampling_rate = sampling_rate or default_samplerate
        n_samples = energy.shape[-1]
        times = np.arange(n_samples) / sampling_rate

        return pf.TimeData(energy, times)

    return _factory
