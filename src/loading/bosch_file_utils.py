"""Lightweight Bosch file enumeration (no heavy I/O dependencies).

This module contains only the file-enumeration / metadata-parsing logic for the
Bosch CNC dataset, kept separate from :mod:`src.loading.bosch_cnc_machining`
(which imports ``h5py`` and ``git``) so that downstream scripts that only need
the per-recording metadata (machine / operation / label / year) — notably the
group/time-aware split evaluation — can import it without pulling in those
heavy, loader-specific dependencies.

See :func:`enumerate_files` for details.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

__all__ = ["enumerate_files"]


def enumerate_files(dir_path: Path) -> pd.DataFrame:
    """Enumerate all Bosch ``.h5`` recordings and parse their metadata.

    The Bosch recordings are nested as
    ``<dir_path>/<machine>/<operation>/<label>/<filename>.h5`` where the
    filename itself encodes the acquisition timeframe, e.g.
    ``M01_Aug_2019_OP00_000.h5`` -> machine ``M01``, month ``Aug``, year
    ``2019``, operation ``OP00``, recording index ``000``.

    Parameters
    ----------
    dir_path : pathlib.Path
        Root directory of the raw Bosch dataset (the ``data`` folder that
        contains the ``M01``/``M02``/``M03`` subfolders).

    Returns
    -------
    pandas.DataFrame
        One row per recording with columns
        ``["machine", "operation", "label", "filename", "path", "year"]``,
        in the same order used by the Bosch loader so that the rows stay
        aligned with previously extracted feature CSVs.

    Notes
    -----
    The path is parsed *relative to* ``dir_path`` (``Path.relative_to``) so the
    enumeration is independent of how deep the dataset lives on disk. The
    leading ``[2:]`` row drop reproduces the original loader behaviour: the
    already-extracted feature matrices (``results/02_*``) were built from this
    exact row order (1702 files -> 1700 kept rows), so the drop is preserved on
    purpose to keep the parsed metadata aligned with those features. ``year`` is
    parsed from the filename token (``filename.split("_")[2]``, e.g.
    ``M01_Aug_2019_OP00_000.h5`` -> 2019) and added as a new column for the
    group/time-aware split analysis.
    """
    file_list = [
        Path(root) / file
        for root, _, files in os.walk(dir_path)
        for file in files
        if file.endswith(".h5")
    ]

    rows = []
    for p in file_list:
        # Relative parts = (machine, operation, label, filename), depth-independent.
        rel_parts = list(p.relative_to(dir_path).parts)
        rows.append(rel_parts + [str(p).replace("\\", "/")])

    file_df = pd.DataFrame(
        rows[2:],
        columns=["machine", "operation", "label", "filename", "path"],
    )
    # Parse the acquisition year from the filename, e.g. "M01_Aug_2019_OP00_000.h5" -> 2019.
    file_df["year"] = file_df["filename"].str.split("_").str[2].astype(int)
    return file_df
