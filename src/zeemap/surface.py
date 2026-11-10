"""Surfaces of patches on the unit sphere, the observer direction and limb darkening.

SPEC.md SURF-1 (Fibonacci sphere), FWD-2 (limb-darkening laws). The star's frame has z along the rotation axis.

- `fibonacci(n)`: the standard Fibonacci lattice, z_k = 1 - (2k + 1)/n, azimuth 2 pi k / golden ratio, k = 0 ... n-1;
  every patch has area 4 pi / n.
- `observer(u, phase)`: n = (sin i cos 2 pi phase, -sin i sin 2 pi phase, -cos i), u = cos i, sin i = sqrt(1 - u^2).
- `visible_weights`: mu = n . centre; weight = area mu LD(mu) for mu > 0, else 0 (SPEC.md FWD-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0
LD_MODES = (0, 1, 3)


@dataclass
class Surface:
    """Patches on the unit sphere."""

    scheme: str  # "fibonacci"
    centres: np.ndarray  # (N, 3) unit vectors
    areas: np.ndarray  # (N,)

    @property
    def n(self) -> int:
        return int(self.centres.shape[0])


def fibonacci(n: int) -> Surface:
    """Fibonacci sphere of n equal-area patches (SURF-1)."""
    n = int(n)
    if n < 1:
        raise ValueError(f"fibonacci: the number of patches must be at least 1, got {n}")
    k = np.arange(n, dtype=float)
    z = 1.0 - (2.0 * k + 1.0) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = 2.0 * np.pi * k / GOLDEN_RATIO
    centres = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return Surface("fibonacci", centres, np.full(n, 4.0 * np.pi / n))


def make_surface(cfg_model_surface: dict) -> Surface:
    """Surface from the configuration table {scheme = "fibonacci", n}."""
    scheme = cfg_model_surface.get("scheme")
    if scheme == "fibonacci":
        return fibonacci(cfg_model_surface["n"])
    raise ValueError(f"surface scheme {scheme!r} is not known (use 'fibonacci')")


def observers(u: float, phases) -> np.ndarray:
    """(P, 3) observer directions in the star's frame, one per phase; u = cos i."""
    u = float(u)
    if not -1.0 <= u <= 1.0:
        raise ValueError(f"observer: u = cos i must lie in [-1, 1], got {u}")
    si = np.sqrt(max(0.0, 1.0 - u * u))
    a = 2.0 * np.pi * np.atleast_1d(np.asarray(phases, dtype=float))
    return np.stack([si * np.cos(a), -si * np.sin(a), np.full(a.shape, -u)], axis=-1)


def observer(u: float, phase: float) -> np.ndarray:
    """(3,) observer direction n = (sin i cos 2 pi phase, -sin i sin 2 pi phase, -cos i), u = cos i."""
    return observers(u, [phase])[0]


def limb_darkening(mu, mode: int) -> np.ndarray:
    """Limb-darkening laws by mode: 0: 1; 1: 0.4 + 0.6 mu; 3: 0.7 + 0.3 mu (FWD-2)."""
    mu = np.asarray(mu, dtype=float)
    if mode == 0:
        return np.ones_like(mu)
    if mode == 1:
        return 0.4 + 0.6 * mu
    if mode == 3:
        return 0.7 + 0.3 * mu
    raise ValueError(f"limb-darkening mode {mode!r} is not supported (use 0, 1 or 3)")


def visible_weights(surface: Surface, n, ld_mode: int) -> tuple[np.ndarray, np.ndarray]:
    """mu = n . centre and weight = area mu LD(mu) for mu > 0, else 0. n (3,) gives (N,); n (P, 3) gives (P, N)."""
    mu = np.asarray(n, dtype=float) @ surface.centres.T
    vis = mu > 0.0
    weight = np.where(vis, surface.areas * mu * limb_darkening(np.where(vis, mu, 0.0), ld_mode), 0.0)
    return mu, weight
