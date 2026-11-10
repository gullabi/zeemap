"""Shared data types between data.py, forward.py and likelihood.py (docs/DEV_INTERFACES.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A regular wavelength grid: start + step * k, k = 0 .. npix - 1, in A."""

    start: float
    step: float
    npix: int

    @property
    def wavelengths(self) -> np.ndarray:
        return self.start + self.step * np.arange(self.npix)

    @property
    def end(self) -> float:
        return self.start + self.step * (self.npix - 1)


@dataclass
class Spectrum:
    """One phase and one Stokes quantity (I or V/I) that is switched on, restricted to its fitted pixels."""

    phase_index: int
    phase: float
    kind: str  # "I" or "VI"
    pixels: np.ndarray  # indices into the grid, increasing
    data: np.ndarray
    sigma: np.ndarray
    weight: np.ndarray
    window: np.ndarray  # window index per pixel
    i_offset: bool = True

    @property
    def label(self) -> str:
        return f"{self.phase:.3f}_{self.kind}"


@dataclass
class DataSet:
    grid: Grid
    phases: np.ndarray  # (P,)
    spectra: list[Spectrum]
    I: np.ndarray  # (P, M), NaN where missing
    VI: np.ndarray  # (P, M)
    sig_I: np.ndarray  # (P, M)
    sig_VI: np.ndarray  # (P, M)
    notes: dict = field(default_factory=dict)  # dataset.json, copied into run.json

    @property
    def n_phases(self) -> int:
        return len(self.phases)
