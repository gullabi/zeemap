"""V5c: the MCMC's batching (DESIGN.md section 4, step 2; decision #25). On the toy star with 4 ensembles of
16 walkers: at most 10 % of the sampler's calls carry fewer proposals than a quarter of all walkers (16; the snooker
move's sub-step is a quarter of an ensemble), against 26 % without the lock-step batching (measured: 223 of 865 calls at 1-10
rows without, 34 of 734 with it). That the chains equal those of separate emcee samplers is test_v14a_lockstep_...
"""

from __future__ import annotations

import numpy as np
import pytest

from zeemap import search as S
from zeemap import uncertainty as U
from test_v14_mcmc import StarToy, _star_cfg

pytestmark = pytest.mark.tier1


def test_v5c_sampler_calls_stay_full(monkeypatch):
    p = StarToy()
    cfg = _star_cfg(20000)
    cfg["uncertainty"]["ensembles"] = 4
    sizes = []
    orig = U._LogProb.__call__

    def call(self, Z):
        sizes.append(np.atleast_2d(Z).shape[0])
        return orig(self, Z)

    monkeypatch.setattr(U._LogProb, "__call__", call)
    r = U.run_mcmc(p, S.Mode(p.value.copy(), 0.0, None, 1), cfg, seed=3, handback=False)
    s = np.array(sizes)
    quarter = 4 * r.nwalkers // 4
    assert np.mean(s < quarter) <= 0.10, (np.mean(s < quarter), np.unique(s, return_counts=True))
    assert np.median(s) >= 1.5 * quarter  # measured 28 of 64
