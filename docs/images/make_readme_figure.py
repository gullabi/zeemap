"""Draw docs/images/quickstart_fit.png, the README figure, from the run directory of the README's quick start.

Usage: python docs/images/make_readme_figure.py out/first docs/images/quickstart_fit.png   (needs matplotlib)
Left: the toy star's Stokes I and V/I with the best mode's model (spectra/mode1_ph0.000.dat). Right: the field
strength over the visible disc for a centred dipole with the fitted polar field and angle to the line of sight
(modes.json), B = (Bp / 2) |3 (m . r) r - m| at the surface, with the line where the line-of-sight component changes sign.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

run, png = Path(sys.argv[1]), Path(sys.argv[2])
cols = np.genfromtxt(run / "spectra" / "mode1_ph0.000.dat", names=True)
ok = np.isfinite(cols["model_I"])
lam = cols["wavelength"][ok]
mode = json.loads((run / "modes.json").read_text())["modes"][0]["derived"]
bp, beta = mode["dipole_MG"], np.radians(mode["beta_deg"])

fig = plt.figure(figsize=(10, 4.2), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1])
for row, (key, label) in enumerate([("I", "Stokes I (scaled)"), ("VI", "V / I")]):
    ax = fig.add_subplot(gs[row, 0])
    ax.plot(lam, cols[f"data_{key}"][ok], color="0.55", lw=0.8, label="toy observation (S/N 40)")
    ax.plot(lam, cols[f"model_{key}"][ok], color="C3", lw=1.3, label="best zeemap model")
    ax.set_ylabel(label)
    if row == 0:
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("wavelength (Å)")

ax = fig.add_subplot(gs[:, 1])
x, y = np.meshgrid(np.linspace(-1, 1, 401), np.linspace(-1, 1, 401))
z = np.sqrt(np.clip(1 - x**2 - y**2, 0, None))  # z towards the observer
m = np.array([np.sin(beta), 0.0, np.cos(beta)])
mr = m[0] * x + m[2] * z
bvec = [3 * mr * x - m[0], 3 * mr * y, 3 * mr * z - m[2]]
b = 0.5 * bp * np.sqrt(sum(c**2 for c in bvec))
blos = 0.5 * bp * bvec[2]
disc = x**2 + y**2 <= 1
im = ax.imshow(np.where(disc, b, np.nan), origin="lower", extent=(-1, 1, -1, 1), cmap="magma", vmin=bp / 2, vmax=bp)
ax.contour(x, y, np.where(disc, blos, np.nan), levels=[0.0], colors="white", linestyles="--", linewidths=1)
ax.set_axis_off()
ax.set_title(f"fitted field: {bp:.1f} MG dipole,\naxis {np.degrees(beta):.0f}° from the line of sight", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.8, label="|B| on the visible disc (MG)")
fig.savefig(png, dpi=110)
