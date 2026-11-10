"""Run-time convergence diagnostics of the MCMC (VERIFICATION.md V14a-e; SPEC.md UNC-2).

Every function works on plain numpy arrays, so the rules can be tested on synthetic chains with known answers
(`tests/test_v14_mcmc.py`). Shapes: an ensemble's chain is (steps, walkers); a run is (ensembles, steps, walkers).

Definitions and sources:
- Integrated autocorrelation time (V14b): the normalised autocorrelation function of each walker (FFT), averaged over
  the walkers of an ensemble, summed as tau = 1 + 2 sum_{t=1}^{M} rho(t) with Sokal's automatic window: the smallest M
  with M >= c tau(M), c = 5 (Sokal 1997, "Monte Carlo methods in statistical mechanics"; Goodman & Weare 2010; the
  walker-averaged estimator recommended by the emcee documentation, Foreman-Mackey et al. 2013). Reimplemented here;
  emcee's own `integrated_time` is not used because it raises on short chains.
- Rank normalisation, split R-hat, folded R-hat, bulk ESS and tail ESS (V14c, V14d): Vehtari, Gelman, Simpson,
  Carpenter & Buerkner 2021, Bayesian Analysis 16, 667, "Rank-normalization, folding, and localization: an improved
  R-hat for assessing convergence of MCMC" (sections 3 and 4): ranks r of the pooled draws -> z = Phi^-1((r - 3/8) /
  (S + 1/4)); R-hat = max(split R-hat of z, split R-hat of the rank-normalised |theta - median|); bulk ESS = ESS of the
  rank-normalised split chains; tail ESS = min(ESS of I(theta <= q05), ESS of I(theta <= q95)) on split chains. The
  multi-chain ESS uses Geyer's initial monotone sequence on the combined autocorrelation
  rho_t = 1 - (W - mean_m acov_m(t)) / var+ (Stan reference manual, "Effective sample size"; Geyer 1992), with the
  antithetic cap tau >= 1 / log10(M N).
- Drift and stuck walkers (V14e): DESIGN.md section 4 and VERIFICATION.md V14e, see `drift` and `stuck_walkers`.

Per quantity (UNC-2, decision #17): besides the mode-level verdict (`V14Report.passed`, every rule as before),
`check_v14` gives each quantity its own verdict in `V14Report.quantity_failed` {name: [failed rules]}: V14b on the
quantity's own integrated autocorrelation time (the longest over the ensembles, `tau_quantity`; length >= tau_factor x
tau and a change of less than 5 % over the last quarter), V14c and V14d on the quantity, and the -ln L rules V14e
(drift, stuck walkers), which belong to the whole run and fail every quantity. When the mode-level verdict fails,
uncertainty.py reports an interval for each quantity whose list is empty (a converged mode reports every quantity).

Declared nuisance (decision #18): `nuisance` names sampling coordinates and quantities that are sampled but not
required to converge (the orientation of a single-phase fit). They are left out of the burn-in, of the longest
autocorrelation time of V14b and of the V14c/V14d verdict; their statistics are still computed and reported
(`V14Report.nuisance`).

Which sequences go into which statistic (V14c: walkers of one ensemble are not independent):
- R-hat: one chain per ensemble = its kept samples flattened step-major (all walkers of step t before step t + 1), split
  in halves by step, so 2 x ensembles half-chains.
- ESS: every walker of every ensemble is one sequence, split in halves (2 x ensembles x walkers sequences of
  kept_steps // 2). Geyer's truncation needs sequences ordered in time; a step-major flattened ensemble would place
  different walkers at lag 1 and stop the sum at once (ESS ~ n regardless of the autocorrelation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtri

# Default rule values (VERIFICATION.md V14; DESIGN.md section 4). The chain-length factor comes from the configuration
# (uncertainty.tau_factor).
BURN_TAUS = 5.0
TAU_CHANGE_MAX = 0.05
RHAT_MAX = 1.01
ESS_MIN = 400.0
DRIFT_MAX = 1.0  # nats
STUCK_MIN = 3.0  # nats
STUCK_MADS = 5.0
ACCEPT_RANGE = (0.2, 0.5)
SOKAL_C = 5.0

RULE_NAMES = {
    "V14b_length": "chain length < tau_factor x longest autocorrelation time",
    "V14b_tau_stable": "longest autocorrelation time changed by >= 5 % over the last quarter",
    "V14c_rhat": "rank-normalised split R-hat >= 1.01",
    "V14d_ess_bulk": "bulk ESS < 400",
    "V14d_ess_tail": "tail ESS < 400",
    "V14e_drift": "median -ln L of the second half differs from the first half by >= 1 nat",
    "V14e_stuck": "stuck walker(s)",
}


# ---------------------------------------------------------------------------------------------------------------------
# Autocorrelation time


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance (divided by N) along axis 0 for every column; x is (N,) or (N, K). FFT, zero-padded."""
    x = np.asarray(x, dtype=float)
    one = x.ndim == 1
    if one:
        x = x[:, None]
    n = x.shape[0]
    y = x - x.mean(axis=0)
    m = 2 * _next_pow2(n)
    f = np.fft.rfft(y, n=m, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=m, axis=0)[:n] / n
    return acov[:, 0] if one else acov


def _sokal_tau(rho: np.ndarray, c: float = SOKAL_C) -> float:
    taus = 2.0 * np.cumsum(rho) - 1.0  # tau(M) = 1 + 2 sum_{t=1}^{M} rho(t), rho(0) = 1
    ok = np.arange(taus.size) >= c * taus
    m = int(np.argmax(ok)) if ok.any() else taus.size - 1
    return float(taus[m])


def integrated_time(chain: np.ndarray, c: float = SOKAL_C) -> np.ndarray:
    """Integrated autocorrelation time per parameter of one ensemble.

    chain: (steps, walkers) or (steps, walkers, ndim). The normalised autocorrelation of each walker is averaged over
    the walkers, then summed with Sokal's window (module docstring). Constant parameters get tau = 1. Returns (ndim,)
    (a 1-element array for a 2-d input)."""
    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 2:
        chain = chain[:, :, None]
    steps, walkers, ndim = chain.shape
    out = np.ones(ndim)
    if steps < 2:
        return out
    for d in range(ndim):
        acov = autocovariance(chain[:, :, d])  # (steps, walkers)
        v = acov[0]
        good = v > 0
        if not good.any():
            continue
        rho = np.mean(acov[:, good] / v[good], axis=1)
        out[d] = max(_sokal_tau(rho, c), 1e-12)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Rank normalisation, R-hat, ESS


def rank_normalize(seqs: np.ndarray) -> np.ndarray:
    """z = Phi^-1((r - 3/8) / (S + 1/4)) with r the average ranks of the pooled draws (Vehtari et al. 2021, eq. 14)."""
    from scipy.stats import rankdata

    seqs = np.asarray(seqs, dtype=float)
    r = rankdata(seqs.ravel(), method="average").reshape(seqs.shape)
    return ndtri((r - 0.375) / (seqs.size + 0.25))


def split_halves(seqs: np.ndarray) -> np.ndarray:
    """(M, N) -> (2M, N // 2): first and second half of each sequence (the middle draw of an odd N is dropped)."""
    seqs = np.asarray(seqs, dtype=float)
    h = seqs.shape[1] // 2
    return np.concatenate([seqs[:, :h], seqs[:, seqs.shape[1] - h:]], axis=0)


def _rhat_basic(seqs: np.ndarray) -> float:
    m, n = seqs.shape
    if m < 2 or n < 2:
        return math.inf
    means = seqs.mean(axis=1)
    w = float(np.mean(seqs.var(axis=1, ddof=1)))
    b = n * float(means.var(ddof=1))
    if w <= 0.0:
        return 1.0 if b == 0.0 else math.inf
    var_plus = (n - 1) / n * w + b / n
    return math.sqrt(var_plus / w)


def rhat(chains: np.ndarray) -> float:
    """Rank-normalised split R-hat (Vehtari et al. 2021): max of the bulk and folded split R-hat.

    chains: (M, N), one row per chain (here: per ensemble, flattened step-major); each row is split in halves here."""
    chains = np.asarray(chains, dtype=float)
    if np.ptp(chains) == 0.0:
        return 1.0
    s = split_halves(chains)
    bulk = _rhat_basic(rank_normalize(s))
    folded = _rhat_basic(rank_normalize(np.abs(s - np.median(s))))
    return max(bulk, folded)


def ess(seqs: np.ndarray) -> float:
    """Multi-chain effective sample size of (M, N) sequences (already split if wanted), Geyer's initial monotone
    sequence (Stan reference manual; Vehtari et al. 2021 section 3.2)."""
    seqs = np.asarray(seqs, dtype=float)
    m, n = seqs.shape
    if n < 4:
        return float(m * n) if n > 0 and np.ptp(seqs) == 0.0 else 0.0
    if np.ptp(seqs) == 0.0:
        return float(m * n)
    acov = autocovariance(seqs.T).T  # (M, N)
    chain_var = acov[:, 0] * n / (n - 1.0)
    mean_var = float(np.mean(chain_var))
    var_plus = mean_var * (n - 1.0) / n
    if m > 1:
        var_plus += float(np.var(seqs.mean(axis=1), ddof=1))
    if var_plus <= 0.0:
        return float(m * n)
    mean_acov = acov.mean(axis=0)
    rho = np.zeros(n)
    rho[0] = 1.0
    rho_even = 1.0
    rho_odd = 1.0 - (mean_var - mean_acov[1]) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < n - 3 and rho_even + rho_odd > 0.0:
        rho_even = 1.0 - (mean_var - mean_acov[t + 1]) / var_plus
        rho_odd = 1.0 - (mean_var - mean_acov[t + 2]) / var_plus
        if rho_even + rho_odd >= 0.0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0.0 and max_t + 1 < n:
        rho[max_t + 1] = rho_even
    t = 1  # initial monotone sequence
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = 0.5 * (rho[t - 1] + rho[t])
            rho[t + 2] = rho[t + 1]
        t += 2
    total = float(m * n)
    tau = -1.0 + 2.0 * float(np.sum(rho[:max_t + 1])) + (rho[max_t + 1] if 0 <= max_t + 1 < n else 0.0)
    tau = max(tau, 1.0 / math.log10(total)) if total > 1 else max(tau, 1.0)
    return total / tau


def ess_bulk(seqs: np.ndarray) -> float:
    """Bulk ESS: ESS of the rank-normalised split sequences. seqs: (M, N) sequences ordered in time."""
    return ess(rank_normalize(split_halves(seqs)))


def ess_tail(seqs: np.ndarray) -> float:
    """Tail ESS: min of the ESS of the 5 % and 95 % quantile indicators, on split sequences."""
    s = split_halves(seqs)
    q05, q95 = np.quantile(s, [0.05, 0.95])
    return min(ess((s <= q05).astype(float)), ess((s <= q95).astype(float)))


# ---------------------------------------------------------------------------------------------------------------------
# Drift, stuck walkers


def drift(nll: np.ndarray) -> float:
    """|median -ln L of the second half - median of the first half| of one ensemble's kept chain (steps, walkers),
    halves by step (V14e)."""
    nll = np.asarray(nll, dtype=float)
    h = nll.shape[0] // 2
    if h == 0:
        return math.inf
    return abs(float(np.median(nll[nll.shape[0] - h:])) - float(np.median(nll[:h])))


def stuck_walkers(nll: np.ndarray) -> tuple[list[int], float]:
    """Walkers whose median -ln L lies farther than max(3 nats, 5 x MAD of the walkers' medians) from the median of
    the other walkers' medians (V14e). MAD = median |m - median(m)| (not scaled). Returns (indices, threshold)."""
    med = np.median(np.asarray(nll, dtype=float), axis=0)
    mad = float(np.median(np.abs(med - np.median(med))))
    thr = max(STUCK_MIN, STUCK_MADS * mad)
    out = []
    for w in range(med.size):
        others = np.delete(med, w)
        if others.size and abs(med[w] - float(np.median(others))) > thr:
            out.append(w)
    return out, thr


# ---------------------------------------------------------------------------------------------------------------------
# The V14 report


@dataclass
class V14Report:
    passed: bool
    failed: list[str]  # rule keys of RULE_NAMES
    burn: int  # steps discarded
    kept: int  # steps kept per walker
    tau: float  # longest integrated autocorrelation time on the kept chain
    tau_3q: float  # the same on the first 3/4 of the kept chain
    tau_per_param: list[float] = field(default_factory=list)
    rhat: dict = field(default_factory=dict)  # quantity -> R-hat
    ess_bulk: dict = field(default_factory=dict)
    ess_tail: dict = field(default_factory=dict)
    drift: list[float] = field(default_factory=list)  # per ensemble
    stuck: list[list[int]] = field(default_factory=list)  # per ensemble
    stuck_threshold: list[float] = field(default_factory=list)
    acceptance: list[float] = field(default_factory=list)  # per ensemble
    warnings: list[str] = field(default_factory=list)
    failed_quantities: dict = field(default_factory=dict)  # rule -> quantities that failed it
    quantity_failed: dict = field(default_factory=dict)  # quantity -> rules it fails on its own (decision #17)
    tau_quantity: dict = field(default_factory=dict)  # quantity -> its own longest autocorrelation time
    nuisance: list = field(default_factory=list)  # declared nuisance names, not part of the verdict (decision #18)

    def to_dict(self) -> dict:
        def f(v):
            return None if (isinstance(v, float) and not math.isfinite(v)) else v

        return {"passed": self.passed, "failed": list(self.failed),
                "failed_text": [RULE_NAMES[k] for k in self.failed], "burn": self.burn, "kept": self.kept,
                "tau": f(float(self.tau)), "tau_3q": f(float(self.tau_3q)),
                "tau_per_param": [f(float(t)) for t in self.tau_per_param],
                "rhat": {k: f(float(v)) for k, v in self.rhat.items()},
                "ess_bulk": {k: f(float(v)) for k, v in self.ess_bulk.items()},
                "ess_tail": {k: f(float(v)) for k, v in self.ess_tail.items()},
                "drift": [f(float(v)) for v in self.drift], "stuck": [list(s) for s in self.stuck],
                "stuck_threshold": [float(v) for v in self.stuck_threshold],
                "acceptance": [float(a) for a in self.acceptance], "warnings": list(self.warnings),
                "failed_quantities": {k: list(v) for k, v in self.failed_quantities.items()},
                "quantity_failed": {k: list(v) for k, v in self.quantity_failed.items()},
                "tau_quantity": {k: f(float(v)) for k, v in self.tau_quantity.items()},
                "nuisance": list(self.nuisance)}


def burn_in(chains: np.ndarray, burn_taus: float = BURN_TAUS) -> int:
    """Steps to discard: burn_taus x the longest autocorrelation time, estimated on the second half of the chain
    (the part least affected by the start). chains: (E, steps, walkers, ndim)."""
    steps = chains.shape[1]
    half = chains[:, steps // 2:]
    if half.shape[1] < 2:
        return steps
    tau = max(float(np.max(integrated_time(half[e]))) for e in range(chains.shape[0]))
    return min(steps, int(math.ceil(burn_taus * tau)))


def _ensemble_chains(q: np.ndarray) -> np.ndarray:
    """(E, steps, walkers) -> (E, steps * walkers), step-major."""
    return q.reshape(q.shape[0], -1)


def _walker_seqs(q: np.ndarray) -> np.ndarray:
    """(E, steps, walkers) -> (E * walkers, steps)."""
    return np.transpose(q, (0, 2, 1)).reshape(-1, q.shape[1])


def quantity_stats(q: np.ndarray) -> tuple[float, float, float]:
    """(R-hat, bulk ESS, tail ESS) of one quantity on the kept chain q: (E, steps, walkers)."""
    return rhat(_ensemble_chains(q)), ess_bulk(_walker_seqs(q)), ess_tail(_walker_seqs(q))


def _tau_of(q: np.ndarray) -> tuple[float, float]:
    """(tau, tau over the first 3/4) of one quantity's kept chain q (E, steps, walkers): the longest over the
    ensembles."""
    n3 = max(2, (3 * q.shape[1]) // 4)
    tau = max(float(integrated_time(q[i])[0]) for i in range(q.shape[0]))
    tau3 = max(float(integrated_time(q[i, :n3])[0]) for i in range(q.shape[0]))
    return tau, tau3


def check_v14(chains: np.ndarray, nll: np.ndarray, *, tau_factor: float = 50.0, quantities: dict | None = None,
              acceptance=None, burn: int | None = None, names: list[str] | None = None,
              nuisance=None) -> V14Report:
    """Apply the rules V14b-e to a run.

    chains: (E, steps, walkers, ndim), the sampling coordinates (autocorrelation time, burn-in);
    nll: (E, steps, walkers), -ln L of each sample (drift, stuck walkers);
    quantities: {name: (E, steps, walkers)} on the SAME steps as `chains`, the quantities for R-hat and ESS (reported
    parameters in canonical form and derived quantities); default: the columns of `chains` named by `names`;
    acceptance: per-ensemble acceptance fractions (reported, warning outside 0.2-0.5);
    burn: steps to discard (default `burn_in(chains)` over the columns not declared nuisance);
    nuisance: names (of columns of `chains` and of quantities) left out of the verdict (module docstring).
    """
    chains = np.asarray(chains, dtype=float)
    nll = np.asarray(nll, dtype=float)
    e, steps, walkers, ndim = chains.shape
    names = list(names) if names is not None else [f"p{i}" for i in range(ndim)]
    nuis = set(nuisance or ())
    if quantities is None:
        quantities = {names[d]: chains[..., d] for d in range(ndim)}
    req = [d for d in range(ndim) if names[d] not in nuis] or list(range(ndim))
    b = burn_in(chains[..., req]) if burn is None else int(burn)
    kept = steps - b
    failed: list[str] = []
    fq: dict[str, list[str]] = {}
    warnings: list[str] = []
    acc = [float(a) for a in np.atleast_1d(acceptance)] if acceptance is not None else []
    for k, a in enumerate(acc):
        if not (ACCEPT_RANGE[0] <= a <= ACCEPT_RANGE[1]):
            warnings.append(f"ensemble {k}: acceptance fraction {a:.3f} outside {ACCEPT_RANGE[0]}-{ACCEPT_RANGE[1]}")
    nuis_list = sorted(n for n in nuis if n in names or n in quantities)
    if kept < 8:
        return V14Report(False, list(RULE_NAMES), b, kept, math.inf, math.inf, acceptance=acc, warnings=warnings,
                         quantity_failed={n: list(RULE_NAMES) for n in quantities},
                         tau_quantity={n: math.inf for n in quantities}, nuisance=nuis_list)

    kc = chains[:, b:]
    taus = np.array([integrated_time(kc[i]) for i in range(e)])  # (E, ndim)
    tau = float(taus[:, req].max())
    n3 = max(2, (3 * kept) // 4)
    tau3 = max(float(np.max(integrated_time(kc[i, :n3][..., req]))) for i in range(e))
    if kept < tau_factor * tau:
        failed.append("V14b_length")
    if abs(tau - tau3) >= TAU_CHANGE_MAX * tau:
        failed.append("V14b_tau_stable")

    rh, eb, et, tq = {}, {}, {}, {}
    qfail: dict[str, list[str]] = {}
    for name, q in quantities.items():
        q = np.asarray(q, dtype=float)[:, b:]
        rh[name], eb[name], et[name] = quantity_stats(q)
        tq[name], t3 = _tau_of(q)
        own = []
        if kept < tau_factor * tq[name]:
            own.append("V14b_length")
        if abs(tq[name] - t3) >= TAU_CHANGE_MAX * tq[name]:
            own.append("V14b_tau_stable")
        qfail[name] = own
    for rule, test in (("V14c_rhat", lambda n: not rh[n] < RHAT_MAX),
                       ("V14d_ess_bulk", lambda n: not eb[n] >= ESS_MIN),
                       ("V14d_ess_tail", lambda n: not et[n] >= ESS_MIN)):
        bad = [n for n in quantities if test(n)]
        for n in bad:
            qfail[n].append(rule)
        bad = [n for n in bad if n not in nuis]
        if bad:
            failed.append(rule)
            fq[rule] = bad

    kn = nll[:, b:]
    dr = [drift(kn[i]) for i in range(e)]
    if any(not d < DRIFT_MAX for d in dr):
        failed.append("V14e_drift")
    st, thr = [], []
    for i in range(e):
        s, t = stuck_walkers(kn[i])
        st.append(s)
        thr.append(t)
    if any(st):
        failed.append("V14e_stuck")
    run_rules = [r for r in ("V14e_drift", "V14e_stuck") if r in failed]
    for n in qfail:
        qfail[n].extend(run_rules)
    return V14Report(not failed, failed, b, kept, tau, tau3, list(taus.max(axis=0)), rh, eb, et, dr, st, thr, acc,
                     warnings, fq, qfail, tq, nuis_list)
