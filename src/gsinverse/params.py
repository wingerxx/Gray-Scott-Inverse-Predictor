"""Extract (feed, kill) parameter pairs from the source HDF5 dataset.

Only the ``feed``/``kill`` attributes are used -- the stored images are
ignored entirely. We additionally sanity-check that ``du``/``dv`` in the
source records match the fixed constants used throughout this project.
"""

import math
from typing import List, Tuple

import h5py

EXPECTED_DU = 0.16
EXPECTED_DV = 0.08
_TOL = 1e-6


def extract_fk_pairs(hdf5_path: str, check_diffusion: bool = True) -> List[Tuple[float, float]]:
    """Extract ``(f, k)`` pairs from every record in the HDF5 dataset.

    Args:
        hdf5_path: Path to the HDF5 file (e.g. ``grayscott_64x64_1-1000.hdf5``).
        check_diffusion: If ``True``, assert that each record's ``du``/``dv``
            attributes are close to the project's fixed constants
            (``du=0.16``, ``dv=0.08``).

    Returns:
        A list of ``(f, k)`` tuples, one per record, in the order the
        records appear in the file.

    Raises:
        AssertionError: If ``check_diffusion`` is ``True`` and a record's
            ``du``/``dv`` deviates from the expected constants.
    """
    pairs: List[Tuple[float, float]] = []

    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            meta = f[key]["meta"].attrs
            feed = float(meta["feed"])
            kill = float(meta["kill"])

            if check_diffusion:
                du = float(meta["du"])
                dv = float(meta["dv"])
                assert math.isclose(du, EXPECTED_DU, abs_tol=_TOL), (
                    f"Record {key}: du={du} does not match expected {EXPECTED_DU}"
                )
                assert math.isclose(dv, EXPECTED_DV, abs_tol=_TOL), (
                    f"Record {key}: dv={dv} does not match expected {EXPECTED_DV}"
                )

            pairs.append((feed, kill))

    return pairs
