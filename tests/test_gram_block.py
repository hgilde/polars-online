"""`gram_block_rows` on `ewridge` (docs/ENHANCEMENTS.md E51, docs/PLAN.md task 71).

The block is an implementation of the same model, not a different one: the
four scalars (`n_eff` among them) run the per-row recursion unchanged, and
only the `k x k` matrix waits, to be merged once per block by one matrix
product. So the claims are that a blocked fit is the per-row fit to rounding
and `n_eff` to the bit; that chunk invariance, a save mid-block and the JSON
export all hold with rows in flight; that `gram()` read mid-block sees the
held rows without moving the model's block boundary; that a state written
before the field existed loads with blocking off; and that the settings
under which a block cannot pay, or cannot be read, are refused with the
reason. `test_semantics_all_models` runs the common invariants over a blocked
`ewridge` as well.
"""

from __future__ import annotations

import json
import re

import numpy as np
import polars as pl
import pytest

import polars_online as po

K = 6
HL = 100.0


def stream(n: int = 700, seed: int = 0) -> pl.DataFrame:
    """Nulls, zero weights, a gap of five halflives and a session change:
    everything a held row has to carry into the merge."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, K))
    y = X @ np.linspace(1.0, -1.0, K) + 0.5 + 0.1 * rng.standard_normal(n)
    y[5::17] = np.nan
    w = 0.5 + rng.random(n)
    w[::23] = 0.0
    t = np.arange(float(n))
    t[300:] += 5 * HL
    df = pl.DataFrame({f"x{j}": X[:, j] for j in range(K)})
    return df.with_columns(
        y=pl.Series(y).fill_nan(None),
        w=pl.Series(w),
        t=pl.Series(t),
        s=pl.Series(["a"] * 450 + ["b"] * (n - 450)),
    )


def spec(block: int | None, **kw):
    opts = dict(
        targets=["y"],
        features=[f"x{j}" for j in range(K)],
        clock="t",
        max_dclock=5 * HL,
        halflife=HL,
        weight="w",
        ridge=1e-3,
        min_periods=8.0,
        solve_every=7.0,
        max_rows_between_solves=30,
        coef_every=1,
        gram_block_rows=block,
    )
    opts.update(kw)
    return po.spec.ewridge("m", **opts)


def fields(out: pl.DataFrame) -> pl.DataFrame:
    return out.select("m").unnest("m")


def run(df: pl.DataFrame, block: int | None, chunk: int | None = None, **kw):
    bank = po.ModelBank([spec(block, **kw)])
    if chunk is None:
        return fields(bank.fit_predict(df)), bank
    return fields(pl.concat([bank.fit_predict(c) for c in df.iter_slices(chunk)])), bank


def ridge_state(bank: po.ModelBank) -> dict:
    """The one `ewridge` instance's state in the JSON export: spec 0, its
    one group, model instance 0."""
    doc = json.loads(bank.to_json())
    return doc["states"][0][0][1]["models"][0]["model"]["EwRidge"]


def assert_same_fit(plain: pl.DataFrame, blocked: pl.DataFrame, rel: float = 1e-9) -> None:
    """The scalars to the bit, the fit to rounding."""
    assert plain["n_eff"].equals(blocked["n_eff"]), "n_eff is not on the blocked path"
    for col in ("pred_y", "resid_y"):
        a = plain[col].to_numpy().astype(float)
        b = blocked[col].to_numpy().astype(float)
        both_nan = np.isnan(a) & np.isnan(b)
        ok = both_nan | (np.abs(a - b) <= rel * (1.0 + np.abs(a)))
        assert ok.all(), f"{col}: {np.sum(~ok)} rows differ, worst {np.nanmax(np.abs(a - b))}"
    ca = plain["coef"].to_list()
    cb = blocked["coef"].to_list()
    for i, (p, q) in enumerate(zip(ca, cb, strict=True)):
        assert (p is None) == (q is None), f"coef at row {i}: one side unsolved"
        if p is not None:
            assert np.allclose(p, q, rtol=rel, atol=rel), f"coef at row {i}: {p} vs {q}"


@pytest.mark.parametrize("block", [3, 16, 256])
def test_a_blocked_fit_is_the_per_row_fit(block):
    df = stream()
    plain, _ = run(df, 0)
    blocked, _ = run(df, block)
    assert_same_fit(plain, blocked)


def test_zero_is_the_untouched_path():
    df = stream()
    absent, _ = run(df, None)
    zero, _ = run(df, 0)
    assert absent.equals(zero, null_equal=True)


@pytest.mark.parametrize("chunk", [1, 7, 13, 100, 333])
def test_chunking_cannot_move_a_block(chunk):
    """The block fills by the model's own row count and drains at its own
    solves, never at a chunk end: every chunking is the same stream to the
    bit, coefficients included."""
    df = stream()
    one, _ = run(df, 16)
    many, _ = run(df, 16, chunk=chunk)
    assert one.equals(many, null_equal=True)


def test_a_save_mid_block_resumes_bit_for_bit(tmp_path):
    df = stream()
    whole, _ = run(df, 16)
    a = po.ModelBank([spec(16)])
    a.fit_predict(df.slice(0, 101))
    assert ridge_state(a)["cov"]["pending"]["lam"], "row 101 should sit mid-block"
    a.save(tmp_path / "blocked.state")
    b = po.ModelBank.load(tmp_path / "blocked.state")
    rest = fields(b.fit_predict(df.slice(101)))
    assert rest.equals(whole.slice(101), null_equal=True)


class _Raw(bytes):
    """A msgpack float's eight bytes, kept verbatim so a NaN payload or a
    signed zero survives the trip."""


def _msgpack_read(b: bytes, i: int = 0):
    """A msgpack reader for the subset a bank file uses. Maps come back as
    lists of `(key, value)` pairs, in file order, so nothing is hashed or
    reordered. Returns `(value, next_index)`."""
    c = b[i]
    i += 1
    if c <= 0x7F:
        return c, i
    if c >= 0xE0:
        return c - 0x100, i
    if 0x80 <= c <= 0x8F:
        return _msgpack_map(b, i, c & 0x0F)
    if 0x90 <= c <= 0x9F:
        return _msgpack_array(b, i, c & 0x0F)
    if 0xA0 <= c <= 0xBF:
        n = c & 0x1F
        return b[i : i + n].decode(), i + n
    if c == 0xC0:
        return None, i
    if c == 0xC2:
        return False, i
    if c == 0xC3:
        return True, i
    if c in (0xC4, 0xC5, 0xC6):
        w = 1 << (c - 0xC4)
        n = int.from_bytes(b[i : i + w], "big")
        i += w
        return bytes(b[i : i + n]), i + n
    if c == 0xCA:
        raise AssertionError("no f32 in a bank file")
    if c == 0xCB:
        return _Raw(b[i : i + 8]), i + 8
    if 0xCC <= c <= 0xCF:
        w = 1 << (c - 0xCC)
        return int.from_bytes(b[i : i + w], "big"), i + w
    if 0xD0 <= c <= 0xD3:
        w = 1 << (c - 0xD0)
        return int.from_bytes(b[i : i + w], "big", signed=True), i + w
    if c in (0xD9, 0xDA, 0xDB):
        w = 1 << (c - 0xD9)
        n = int.from_bytes(b[i : i + w], "big")
        i += w
        return b[i : i + n].decode(), i + n
    if c in (0xDC, 0xDD):
        w = 2 << (c - 0xDC)
        n = int.from_bytes(b[i : i + w], "big")
        return _msgpack_array(b, i + w, n)
    if c in (0xDE, 0xDF):
        w = 2 << (c - 0xDE)
        n = int.from_bytes(b[i : i + w], "big")
        return _msgpack_map(b, i + w, n)
    raise AssertionError(f"msgpack byte {c:#x} is not one a bank file writes")


def _msgpack_array(b: bytes, i: int, n: int):
    out = []
    for _ in range(n):
        v, i = _msgpack_read(b, i)
        out.append(v)
    return out, i


def _msgpack_map(b: bytes, i: int, n: int):
    out = []
    for _ in range(n):
        k, i = _msgpack_read(b, i)
        v, i = _msgpack_read(b, i)
        out.append((k, v))
    return ("map", out), i


def _msgpack_write(v) -> bytes:
    """The inverse of `_msgpack_read`; any valid encoding will do for the
    loader, so the widths are the smallest that fit."""
    if v is None:
        return b"\xc0"
    if v is True:
        return b"\xc3"
    if v is False:
        return b"\xc2"
    if isinstance(v, _Raw):
        return b"\xcb" + bytes(v)
    if isinstance(v, int):
        if 0 <= v <= 0x7F:
            return bytes([v])
        if -32 <= v < 0:
            return bytes([v + 0x100])
        if v >= 0:
            for tag, w in ((0xCC, 1), (0xCD, 2), (0xCE, 4), (0xCF, 8)):
                if v < 1 << (8 * w):
                    return bytes([tag]) + v.to_bytes(w, "big")
        for tag, w in ((0xD0, 1), (0xD1, 2), (0xD2, 4), (0xD3, 8)):
            if -(1 << (8 * w - 1)) <= v < 1 << (8 * w - 1):
                return bytes([tag]) + v.to_bytes(w, "big", signed=True)
        raise AssertionError(f"{v} does not fit an int64")
    if isinstance(v, str):
        raw = v.encode()
        n = len(raw)
        if n < 32:
            return bytes([0xA0 | n]) + raw
        if n < 256:
            return b"\xd9" + bytes([n]) + raw
        return b"\xda" + n.to_bytes(2, "big") + raw
    if isinstance(v, bytes):
        n = len(v)
        head = b"\xc4" + bytes([n]) if n < 256 else b"\xc5" + n.to_bytes(2, "big")
        return head + v
    if isinstance(v, tuple) and v[0] == "map":
        items = v[1]
        n = len(items)
        head = bytes([0x80 | n]) if n < 16 else b"\xde" + n.to_bytes(2, "big")
        return head + b"".join(_msgpack_write(k) + _msgpack_write(x) for k, x in items)
    if isinstance(v, list):
        n = len(v)
        head = bytes([0x90 | n]) if n < 16 else b"\xdc" + n.to_bytes(2, "big")
        return head + b"".join(_msgpack_write(x) for x in v)
    raise AssertionError(f"cannot write {type(v)}")


def _without_key(v, key: str):
    """The same tree with every map entry named `key` removed: the file a
    writer from before the field existed would have written."""
    if isinstance(v, tuple) and v[0] == "map":
        return ("map", [(k, _without_key(x, key)) for k, x in v[1] if k != key])
    if isinstance(v, list):
        return [_without_key(x, key) for x in v]
    return v


def test_a_state_written_before_the_field_loads_with_blocking_off(tmp_path):
    """The field is additive under `SCHEMA_VERSION` 6: a bank file names its
    fields, so the key is simply absent from an older `ewridge` file, in the
    spec and in every instance's config. Such a file -- made here by
    stripping the key from a current one, since the loader cannot tell the
    two apart -- loads, continues to the bit as the unblocked model, and
    re-saves carrying the key."""
    df = stream()
    head, rest = df.slice(0, 101), df.slice(101)
    whole, _ = run(df, None)
    _, bank = run(head, None)
    bank.save(tmp_path / "now.state")
    now = (tmp_path / "now.state").read_bytes()
    tree, end = _msgpack_read(now)
    assert end == len(now)
    assert _msgpack_write(tree) == now, "the reader and writer are not each other's inverse"
    assert b"gram_block_rows" in now
    before = _msgpack_write(_without_key(tree, "gram_block_rows"))
    assert b"gram_block_rows" not in before
    (tmp_path / "before.state").write_bytes(before)
    old = po.ModelBank.load(tmp_path / "before.state")
    assert old.specs[0]["model"].get("gram_block_rows") is None
    assert fields(old.fit_predict(rest)).equals(whole.slice(101), null_equal=True)
    old.save(tmp_path / "again.state")
    assert b"gram_block_rows" in (tmp_path / "again.state").read_bytes()


def test_the_held_rows_are_in_the_json_and_legible():
    """A state saved mid-block carries the rows the matrix has not seen, as a
    `pending` block a reader can recognise; an unblocked fit carries nothing."""
    df = stream()
    _, blocked = run(df.slice(0, 101), 16)
    held = ridge_state(blocked)["cov"]["pending"]
    assert held["block_rows"] == 16
    n = len(held["lam"])
    assert 0 < n < 16
    assert len(held["w"]) == n
    assert len(held["x"]) == n * (K + 1), "one z row per held row, the intercept slot included"
    assert held["w_open"] > 0
    _, plain = run(df.slice(0, 101), 0)
    assert "pending" not in ridge_state(plain)["cov"]


def test_gram_mid_block_reads_the_held_rows_and_moves_nothing():
    """`gram()` is the matrix *with* the held rows -- the same accumulators
    the unblocked model has at that row -- and reading it does not merge the
    model's own block: the stream after the read is the stream without it."""
    df = stream()
    head, rest = df.slice(0, 101), df.slice(101)
    _, plain = run(head, 0)
    _, read = run(head, 16)
    _, unread = run(head, 16)
    g0, g1 = plain.gram("m")[0], read.gram("m")[0]
    assert g0["n_eff"] == g1["n_eff"]
    assert np.allclose(g0["means"], g1["means"], rtol=1e-9, atol=1e-12)
    scale = np.abs(g0["comoments"]).max()
    assert np.allclose(g0["comoments"], g1["comoments"], rtol=0, atol=1e-9 * scale)
    assert np.allclose(g0["cross_moments"], g1["cross_moments"], rtol=1e-9, atol=1e-12)
    after_read = fields(read.fit_predict(rest))
    after_none = fields(unread.fit_predict(rest))
    assert after_read.equals(after_none, null_equal=True)


def test_a_session_change_blends_the_held_block_first():
    """`session_shrink` reads both matrices in full at the boundary, so the
    held rows go in first, and the twin keeps its block afterwards."""
    df = stream()
    kw = dict(session="s", session_gap=0.0, session_shrink=0.5, long_halflife=1e4)
    plain, _ = run(df, 0, **kw)
    blocked, bank = run(df, 16, **kw)
    assert_same_fit(plain, blocked)
    state = ridge_state(bank)
    assert state["cov"]["pending"]["block_rows"] == 16
    assert state["slow"]["cov"]["pending"]["block_rows"] == 16


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"window": 50.0}, "gram_block_rows and window do not combine"),
        ({"solve_every": 0.0}, "needs a solve cadence"),
        ({"max_rows_between_solves": 1}, "max_rows_between_solves = 1"),
        # The default cadence is halflife / 50, which is every row for an
        # infinite halflife and for `lam`: the option then needs its own.
        ({"halflife": float("inf"), "solve_every": None}, "solve_every = 0"),
        ({"halflife": None, "lam": 0.99, "solve_every": None}, "0 for `lam`"),
        ({"gram_block_rows": 1 << 30}, "over the 256 MiB budget"),
    ],
)
def test_a_block_that_cannot_pay_is_refused(kw, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        spec(64, **kw)


def test_the_budget_names_the_size():
    with pytest.raises(ValueError, match=r"would hold \d+\.\d GiB of rows \(\d+ × \d+ features\)"):
        spec(1 << 30)
    with pytest.raises(ValueError, match="twice for the slow twin"):
        spec(1 << 30, session="s", session_gap=0.0, session_shrink=0.5, long_halflife=1e4)
