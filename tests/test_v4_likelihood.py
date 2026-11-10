"""V4: error handling, likelihood items: V4c (a weight of 0 that reaches the full likelihood is refused; other
weights act in every kind since decision #27, DATA-4: V3g) and
V4d (invalid models get the defined worst value 1e30, never NaN, and are counted; LIKE-7, LIKE-8)."""

from __future__ import annotations

import numpy as np
import pytest

from zeemap.likelihood import WORST, Objective
from test_v1_likelihood import cfg_of, make_dataset, perturbed_models

pytestmark = pytest.mark.tier1


def test_v4c_zero_weight_refused_by_the_full_likelihood():
    import dataclasses

    rng = np.random.default_rng(40)
    ds, *_ = make_dataset(rng, weights=True)
    Objective(ds, cfg_of())  # positive weights: accepted (decision #27)
    sp = ds.spectra[0]
    w = sp.weight.copy()
    w[3] = 0.0
    ds0 = dataclasses.replace(ds, spectra=[dataclasses.replace(sp, weight=w)] + list(ds.spectra[1:]))
    with pytest.raises(ValueError, match="a weight of 0 reached the full likelihood"):
        Objective(ds0, cfg_of())
    Objective(ds0, cfg_of(kind="chi2"))  # the plain chi2 takes it (the pixel counts 0)


def test_v4c_unknown_objective_key_and_kind():
    rng = np.random.default_rng(41)
    ds, *_ = make_dataset(rng)
    with pytest.raises(ValueError, match="objective.bogus"):
        Objective(ds, cfg_of(bogus=1))
    with pytest.raises(ValueError, match="objective.kind"):
        Objective(ds, cfg_of(kind="q"))


def test_v4c_unmatched_kernel_label_refused():
    """An [[objective.kernel]] label that matches no spectrum of the data set raises ConfigError
    naming the label (it was ignored, and the spectrum silently used the initial kernel); matching labels pass."""
    from zeemap.config import ConfigError

    rng = np.random.default_rng(43)
    ds, *_ = make_dataset(rng)
    k = {"rho_short": 4.0, "a": 0.5, "rho_long": 80.0, "j": 0.05}
    Objective(ds, cfg_of(kernel_fit=False, kernel=[{"spectrum": sp.label, **k} for sp in ds.spectra]))
    for kind in ("full", "chi2"):
        with pytest.raises(ConfigError, match=r"objective\.kernel.*'9\.999_I'"):
            Objective(ds, cfg_of(kind=kind, kernel_fit=False, kernel=[{"spectrum": "9.999_I", **k}]))


@pytest.mark.parametrize("cfg", [cfg_of(), cfg_of(kind="chi2")])
def test_v4d_invalid_models_get_worst_value_and_are_counted(cfg):
    rng = np.random.default_rng(42)
    ds, I_true, VI_true = make_dataset(rng)
    I, VI = perturbed_models(rng, I_true, VI_true, 6)
    obj = Objective(ds, cfg)
    clean = obj.neg_log_l(I, VI, np.zeros(6, bool))
    assert obj.n_invalid == 0 and obj.n_evals == 6
    sp_i = ds.spectra[0]
    sp_v = ds.spectra[1]
    I[2, sp_i.phase_index, sp_i.pixels[5]] = np.nan
    VI[3, sp_v.phase_index, sp_v.pixels[0]] = np.inf
    I[4] = 0.0  # a model whose continuum design is singular: non-finite result in "full", plain chi2 in "chi2"
    flag = np.zeros(6, bool)
    flag[1] = True
    got = obj.neg_log_l(I, VI, flag)
    assert not np.any(np.isnan(got))
    assert got[1] == WORST and got[2] == WORST and got[3] == WORST
    np.testing.assert_allclose(got[[0, 5]], clean[[0, 5]], rtol=1e-12)
    n_bad_expected = 3 + (1 if cfg["objective"].get("kind", "full") == "full" else 0)
    if n_bad_expected == 4:
        assert got[4] == WORST
    else:
        assert np.isfinite(got[4]) and got[4] < WORST
    assert obj.n_invalid == n_bad_expected
    assert obj.n_evals == 12
    # NaN outside the fitted pixels does not matter
    I2 = I_true[None].copy()
    VI2 = VI_true[None].copy()
    I2[0, 0, 0] = np.nan
    assert obj.neg_log_l(I2, VI2)[0] < WORST
