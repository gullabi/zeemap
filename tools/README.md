# tools/

| file | what |
|---|---|
| `check.sh` | the check runner: `tools/check.sh <tier> [V<k>...]` (AGENTS.md) |
| `scan.py`, `scan_allow.txt`, `forbidden_identifiers.txt` | repository hygiene scan (V8) |
| `zeeman_library.py` | synthetic toy library and test observation of a magnetic DA white dwarf (below) |
| `measure_lm_shortfall.py` | measures how far the Levenberg-Marquardt stop rules leave a polish above its minimum (decision #37) |

## zeeman_library.py

A toy, not a model atmosphere: Balmer lines (H-alpha to H-epsilon) as Voigt profiles on a Planck continuum, split by
the linear Zeeman effect with a quadratic blueshift estimate, Stokes I and V from weak-field angular weights. Written
from textbook formulas (module docstring); nothing comes from other codes. Use it for the quick start
and tests when no real library is at hand. Write outputs outside the repository (library and observation files must
not be committed).

```
.venv/bin/python tools/zeeman_library.py <outdir>            # library: 5 Teff x 40 B x 9 psi, 52 MB, ~10 s
.venv/bin/python tools/zeeman_library.py <outdir> --fast     # reduced grid for tests (3 Teff, B 1..36 step 5)
.venv/bin/python tools/zeeman_library.py inject <obsdir>     # ph00*.dat + dataset.json + truth.json: Teff 13000 K,
                                                             # dipole Bp 12 MG at beta 40 deg, S/N 40, ~10 s
```
`inject` options: `--teff --bp --beta --snr --sigma-vi --seed --npoints`. Checks: `tests/test_v2_zeeman.py` (V2e).
