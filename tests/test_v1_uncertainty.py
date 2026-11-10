"""V1l (UNC-5): a toy likelihood with a known profile gives that profile.

Toy: cost = 0.5 (x - m)^T C^-1 (x - m). The profile of parameter i (the other parameters re-optimised) is the parabola
0.5 (x_i - m_i)^2 / C_ii, and the re-optimised others follow the conditional mean
m_rest + C_rest,i (x_i - m_i) / C_ii.
"""

import numpy as np
import pytest

from zeemap import uncertainty as U
from zeemap.config import defaults

pytestmark = pytest.mark.tier1

COV = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, -0.3], [0.2, -0.3, 0.5]])
MEAN = np.array([1.0, -2.0, 3.0])


class Quadratic:
    def __init__(self):
        self.Ci = np.linalg.inv(COV)
        self.Lt = np.linalg.cholesky(self.Ci).T
        self.dim = 3
        self.names = ["a", "b", "c"]
        self.free = np.ones(3, dtype=bool)
        self.lo = np.full(3, -50.0)
        self.hi = np.full(3, 50.0)
        self.step = np.sqrt(np.diag(COV))
        self.value = np.zeros(3)
        self.field_slice = None

    def cost(self, X):
        D = np.atleast_2d(np.asarray(X, dtype=float)) - MEAN
        return 0.5 * np.einsum("ki,ij,kj->k", D, self.Ci, D)

    def residuals(self, x):
        return self.Lt @ (np.asarray(x, dtype=float) - MEAN)

    def canonical(self, x):
        return np.array(x, dtype=float)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_v1l_profile_of_quadratic_is_the_parabola(index):
    p = Quadratic()
    sig = np.sqrt(COV[index, index])
    grid = MEAN[index] + sig * np.linspace(-3.0, 3.0, 13)
    out = U.profile(p, MEAN + 0.1, index, grid, defaults())
    expect = 0.5 * (grid - MEAN[index]) ** 2 / COV[index, index]
    assert np.allclose(out["neg_log_l"], expect, rtol=0, atol=1e-6)
    assert np.allclose(out["x"][:, index], grid, rtol=0, atol=0)  # the profiled parameter stays fixed
    rest = [j for j in range(3) if j != index]
    cond = MEAN[rest][None, :] + np.outer(grid - MEAN[index], COV[rest, index]) / COV[index, index]
    assert np.allclose(out["x"][:, rest], cond, atol=1e-4)
    assert not np.any(np.asarray(p.free) == False)  # noqa: E712  the problem itself is untouched
