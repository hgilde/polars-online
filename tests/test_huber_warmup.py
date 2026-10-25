"""The Huber warm-up refit prototype (``huber_warm_rows``, branch
``huber-warmup``; docs/HUBER.md §3): each target's first rows refit by IRLS to
convergence, then the stream goes on online. Held here: the refit lands on
statsmodels' converged ``RLM`` on the same rows, chunk invariance (hard rule
3) and a save and load mid-buffer. A prototype behind a default-off option."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

import polars_online as po
from test_huber_scale import _frame, stream

TIER = "essential"


@pytest.mark.parametrize("share", [0.05, 0.125])
def test_the_refit_is_statsmodels_converged_fit_on_the_buffer(share):
    """The stream ends on the row the buffer fills, so the fit read after it
    is the refit's: statsmodels' ``RLM`` with the MAD scale on the same rows
    (an odd count, where the weighted median is ``np.median``), to the ridge's
    size. The plain one-pass fit is far from it at these rows, so the refit
    is what closes the distance."""
    n = 401
    df = stream(n, 2 * share, 0, level=1000.0)
    X = sm.add_constant(df["x"].to_numpy())
    rlm = sm.RLM(df["y"].to_numpy(), X, M=HuberT(1.345))
    oracle = rlm.fit(scale_est="mad", conv="coefs", tol=1e-12, maxiter=200).params

    def coef(**kw):
        spec = po.spec.huber("m", targets=["y"], features=["x"], half_life=float("inf"), **kw)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        return np.array(bank.coef("m")["coef"].to_list())

    warm = coef(huber_warm_rows=n)
    plain = coef()
    assert np.abs(warm - oracle).max() < 1e-5, (warm, oracle)
    assert np.abs(plain - oracle).max() > 1e-3, (plain, oracle)


def _spec():
    return po.spec.huber(
        "m",
        targets=["y0", "y1"],
        features=["x"],
        clock="t",
        gap_cap=50.0,
        half_life=300.0,
        coef_every=1,
        weight="w",
        huber_warm_rows=200,
    )


def test_one_chunk_or_many_give_one_run():
    """Hard rule 3: the buffer ends on a count of learned rows, never at a
    chunk's end, so chunks of 1, 7 and 600 rows give the one chunk's output
    to the bit."""
    df = _frame()
    whole = po.ModelBank([_spec()]).fit_predict(df)
    for rows in (1, 7, 600):
        bank = po.ModelBank([_spec()])
        parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
        assert pl.concat(parts).equals(whole, null_equal=True), rows


def test_a_bank_saved_mid_buffer_resumes_to_the_bit():
    """The buffer is state: saved and loaded before, at and after the
    refit's row, a bank goes on as the one that never stopped."""
    df = _frame()
    whole = po.ModelBank([_spec()]).fit_predict(df)
    for cut in (1, 37, 150, 220, 1001):
        bank = po.ModelBank([_spec()])
        head = bank.fit_predict(df.slice(0, cut))
        loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[_spec()])
        tail = loaded.fit_predict(df.slice(cut))
        assert pl.concat([head, tail]).equals(whole, null_equal=True), cut


def test_the_refit_moves_the_output():
    """The activation check: with the warm-up on, the predictions part from
    the plain run's after the buffer fills, and not before it; the emitted
    weight does not move."""
    df = _frame()
    spec = _spec()
    plain_spec = {
        **spec,
        "model": {k: v for k, v in spec["model"].items() if k != "huber_warm_rows"},
    }
    wa = po.ModelBank([spec]).fit_predict(df)["m"]
    wb = po.ModelBank([plain_spec]).fit_predict(df)["m"]
    # Hard rule 8: the weight counts raw weights, whatever the refit does.
    assert wa.struct.field("weight_sum").equals(wb.struct.field("weight_sum"))
    a = wa.struct.field("pred_y0").to_numpy()
    b = wb.struct.field("pred_y0").to_numpy()
    differ = np.flatnonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))
    assert differ.size > 0
    assert differ[0] > 150, differ[0]
