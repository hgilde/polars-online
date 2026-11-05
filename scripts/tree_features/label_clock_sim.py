"""v3, review finding 3: label_clock=, simulated.

A frozen tree model's residual stream r = y - trees is tracked by a model
of the intercept alone: an EW mean of r on the clock (half-life h, each
step's clock capped at gap_cap, as the bank's lam), its residuals' EW
spread sigma, and the bank's conformal tracker (radius q grows by
rate*sigma*coverage on a miss, shrinks by rate*sigma*(1 - coverage) on a
hit). Two semantics on the same events:

- doubled: the doubled stream the bank runs today. A label is learned at its
  arrival, and every diagnostic scores it against the prediction the model
  makes *then*, not the one served at the row's time;
- label_clock: the design. The prediction and interval served at the row's
  time are held until the label arrives; the label is scored against them,
  then learned.

The model learns the same labels at the same clock under both, so the
served predictions are identical; only what the diagnostics score differs.
The doubled simulation is checked against the bank's own run of the same
doubled stream (the review's e2: served 0.8803, learn rows 0.8996)."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from leafrefit import synth, train_trees  # noqa: E402


def simulate(t, avail, r, h, gap_cap, rate=0.05, cov=0.9, mode="label_clock", scale=None):
    n = len(t)
    ev = sorted(
        [(t[i], 0, i) for i in range(n)]
        + [(avail[i], 1, i) for i in range(n) if math.isfinite(avail[i])]
    )
    ln2 = math.log(2.0)
    W = 0.0  # weight of the mean
    m = 0.0  # EW mean of r
    Ws = 0.0  # weight of the spread
    s2 = 0.0  # EW mean square of the scored residuals
    q = None
    last = None
    served = np.full((n, 3), np.nan)  # pred, q at serve time, and q read at the learn event
    in_flight = 0
    for clock, role, i in ev:
        if last is not None:
            lam = math.exp(-ln2 * min(clock - last, gap_cap) / h)
            W *= lam
            Ws *= lam
        last = clock
        sigma = math.sqrt(s2) if Ws > 0 else math.nan
        if role == 0:
            in_flight += 1
            served[i, 0] = m if W > 0 else math.nan
            served[i, 1] = q if q is not None else math.nan
            continue
        in_flight -= 1
        pred_now = m if W > 0 else math.nan
        if mode == "doubled":
            p, qq = pred_now, q
        else:
            p, qq = served[i, 0], served[i, 1]
        served[i, 2] = q if q is not None else math.nan
        if not math.isnan(p):
            e = r[i] - p
            if Ws > 0 and q is not None and not math.isnan(qq):
                step = rate
                if scale == "sqrt":
                    step = rate / math.sqrt(max(in_flight, 1))
                elif scale == "linear":
                    step = rate / max(in_flight, 1)
                q += step * sigma * (cov if abs(e) > qq else -(1 - cov))
                q = max(q, 0.0)
            s2 = (Ws * s2 + e * e) / (Ws + 1.0)
            Ws += 1.0
            if q is None and Ws >= 2:
                q = 1.645 * math.sqrt(s2)
        m = (W * m + r[i]) / (W + 1.0)
        W += 1.0
    return served


def rolling(t, avail, e, te, window, cov=0.9, every=20):
    """The rolling split-conformal radius: the cov-quantile of |e| over the
    labels known in the last `window` of clock, read at each row's time
    (one test row in `every`)."""
    order = np.argsort(avail, kind="stable")
    known, a = avail[order], np.abs(e[order])
    hit, rad = [], []
    for i in np.where(te & ~np.isnan(e))[0][::every]:
        lo, hi = np.searchsorted(known, t[i] - window), np.searchsorted(known, t[i])
        seg = a[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            q = np.quantile(seg, cov)
            hit.append(abs(e[i]) <= q)
            rad.append(q)
    return float(np.mean(hit)), float(np.mean(rad))


def report(name, t, avail, r, te, h, gap_cap, window):
    out = {}
    served_e = None
    for mode, rate, scale in (
        ("doubled", 0.05, None),
        ("label_clock", 0.05, None),
        ("label_clock", 0.01, None),
        ("label_clock", 0.002, None),
        ("label_clock", 0.05, "sqrt"),
        ("label_clock", 0.5, "sqrt"),
    ):
        s = simulate(t, avail, r, h, gap_cap, rate=rate, mode=mode, scale=scale)
        ok = te & ~np.isnan(s[:, 0]) & ~np.isnan(s[:, 1])
        e = r[ok] - s[ok, 0]
        key = f"{mode}, rate {rate}" + (f" / {scale}(in flight)" if scale else "")
        out[key] = {
            "served coverage": float(np.mean(np.abs(e) <= s[ok, 1])),
            "served half-width": float(np.mean(s[ok, 1])),
            "served RMSE": float(np.sqrt(np.mean(e**2))),
        }
        if mode == "doubled":
            # the interval the doubled stream reads at the learn row, against the learn row's own
            # prediction, which is what its tracker targets
            ok2 = ok & ~np.isnan(s[:, 2])
            out[key]["coverage of the learn-row interval"] = float(
                np.mean(np.abs(r[ok2] - s[ok2, 0]) <= s[ok2, 2])
            )
            served_e = r - s[:, 0]
        print(
            f"{name} {key:24s}: " + ", ".join(f"{k} {v:.4f}" for k, v in out[key].items()),
            flush=True,
        )
    for what, e in (("raw residual", r), ("residual after the online bias", served_e)):
        c, w = rolling(t, avail, e, te, window)
        out[f"rolling split-conformal on the {what}"] = {
            "served coverage": c,
            "served half-width": w,
        }
        print(
            f"{name} rolling split-conformal, {what}: coverage {c:.4f}, half-width {w:.4f}",
            flush=True,
        )
    return out


print(C.load_line(), flush=True)
res = {}
df = C.load_flights()
t = df["t"].to_numpy()
av = df["avail"].to_numpy()
yf = df["y"].to_numpy()
tr, va, te = t < 243 * 1440.0, (t >= 243 * 1440.0) & (t < 273 * 1440.0), t >= 273 * 1440.0
K = [C.codes(df[c]).astype(float) for c in ("carrier", "origin", "dest")]
X = np.column_stack([df[c].to_numpy() for c in ("dep_min", "arr_min", "wday", "distance")] + K)
b = train_trees(X, yf, tr, va, 31, 1000)
res["flights"] = report(
    "flights",
    t,
    av,
    yf - b.predict(X, num_iteration=b.best_iteration),
    te,
    7 * 1440.0,
    600.0,
    7 * 1440.0,
)
# planted drift, each label arriving after an exponential delay of mean 20 clock steps
for seed in (0, 1, 2):
    clock, Xs, ys = synth(seed)
    S = clock.max() + 1
    tr, va, te = clock < 0.4 * S, (clock >= 0.4 * S) & (clock < 0.5 * S), clock >= 0.5 * S
    bs = train_trees(Xs, ys, tr, va, 15, 300)
    delay = np.random.default_rng(100 + seed).exponential(20.0, len(clock))
    res[f"planted {seed}"] = report(
        f"planted {seed}, delay mean 20",
        clock,
        clock + delay,
        ys - bs.predict(Xs, num_iteration=bs.best_iteration),
        te,
        20.0,
        10.0,
        20.0,
    )
(C.OUT / "label_clock_sim.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(C.load_line(), flush=True)
