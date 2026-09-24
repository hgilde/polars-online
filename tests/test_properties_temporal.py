"""Property tests for durations and temporal clocks (docs/PLAN.md task 88).

``test_properties.py`` generates adversarial streams on a numeric clock. This
file does the same for what task 88 added: a clock parameter written as a
duration, and a ``Datetime``, ``Date`` or ``Duration`` column as the clock.
Review, not a test, found the two subtle bugs of that task, both in the
duration text: ``"1e37w"`` passed as about 218 years through an unchecked
sum, and whitespace in the text reached a grid's field names. Run against a
copy of either bug, the property that covers it fails. Each property states
what must hold for *every* input its strategy draws, and hypothesis shrinks
a failure to a small one.

Durations:

- ``format_duration`` then ``parse_duration`` is the identity on every
  length a clock can hold, and the text is written largest unit first.
- polars' own parser is an oracle for the text.
- A length past an i64 of nanoseconds is refused, and never wraps, in each
  of the three forms a duration takes.
- Whitespace around a text is trimmed from what is stored and named;
  whitespace inside it is refused.
- The three forms of one length store one text.

Temporal clocks, on every kind of temporal column:

- Chunking never changes the numbers (hard rule 3), nor does a save and
  load at any row.
- Labels held back by ``label_delay`` across chunk boundaries change no
  number either.
- Gaps are taken in integer nanoseconds, so ``n_eff`` is the exact
  recursion even centuries after the stream's first instant.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import polars_online as po
from polars_online._polars_online import format_duration, parse_duration
from test_temporal_clock import START, UNIT_KEYWORD, UNIT_NS, _column

#: The longest duration, and the latest instant, an i64 of nanoseconds holds.
MAX_NS = 2**63 - 1
YEAR_NS = 31_557_600 * 10**9

#: Every spelling of a unit the text takes: ``UNIT_NS``'s nine, whose ``µs``
#: is the micro sign (U+00B5), and the Greek mu (U+03BC) the parser also reads.
TEXT_UNITS = {**UNIT_NS, "\u03bcs": 1_000}

#: Unicode's White_Space: what Rust's ``trim`` and ``char::is_whitespace``
#: and Python's ``str.strip`` all take as whitespace. A space first, so a
#: shrunk counterexample shows one.
WHITESPACE = [
    " ",
    "\t",
    "\n",
    "\x0b",
    "\x0c",
    "\r",
    "\x85",
    "\xa0",
    "\u1680",
    *(chr(c) for c in range(0x2000, 0x200B)),
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
]

#: Pure functions: many examples cost little.
TEXT = settings(max_examples=300, deadline=None)
#: A spec per example.
SPECS = settings(max_examples=150, deadline=None)
#: A bank fitted on a stream per example, per model.
STREAMS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# --- durations ----------------------------------------------------------------


def _counts(low: int, high: int):
    """Counts from ``low`` to ``high``, spread evenly over their number of
    digits. Left to itself, hypothesis draws an integer in a wide range
    with few digits: in 300 examples up to 10**40 it drew none past 21, so
    it never reached a count of 24 to 38 digits, the ones whose product
    with a unit overflows an i128 while the count fits one (``"1e37w"``).
    Here a count has ``k`` digits, ``k`` drawn first."""

    def with_digits(k: int):
        return st.integers(low if k == 1 else max(low, 10 ** (k - 1)), min(high, 10**k - 1))

    return st.integers(len(str(low)), len(str(high))).flatmap(with_digits)


@st.composite
def fixed_texts(draw, most=MAX_NS, signs=("", "-", "+"), positive=False):
    """A duration text in fixed units and its exact length, in Python ints.

    One to six parts, each a count and a unit, in any order, repeated, with
    zero counts and leading zeros: the language polars reads, less its
    calendar units. With ``most`` the length is at most that; with
    ``most=None`` a count reaches 10**40, far past what a clock holds.
    ``positive`` makes the first count at least 1."""
    sign = draw(st.sampled_from(signs))
    units = draw(st.lists(st.sampled_from(list(TEXT_UNITS)), min_size=1, max_size=6))
    parts, total = [], 0
    for i, unit in enumerate(units):
        per = TEXT_UNITS[unit]
        high = 10**40 if most is None else (most - total) // per
        n = draw(_counts(1 if positive and i == 0 else 0, high))
        zeros = "0" * draw(st.integers(0, 2))
        parts.append(f"{zeros}{n}{unit}")
        total += n * per
    return sign + "".join(parts), -total if sign == "-" else total


@st.composite
def boundary_texts(draw):
    """A text whose length is within 2 us of the longest a clock holds, on
    either side, split over random units with the rest in nanoseconds."""
    total = MAX_NS + draw(st.integers(-2_000, 2_000))
    units = draw(st.lists(st.sampled_from(list(TEXT_UNITS)), max_size=5))
    parts, left = [], total
    for unit in units:
        n = draw(st.integers(0, left // TEXT_UNITS[unit]))
        parts.append(f"{n}{unit}")
        left -= n * TEXT_UNITS[unit]
    parts.append(f"{left}ns")
    sign = draw(st.sampled_from(["", "-", "+"]))
    return sign + "".join(parts), -total if sign == "-" else total


@st.composite
def by_parts(draw):
    """A length built from its written parts, each zero or not, so every
    pattern of present and absent units is reached."""

    def part(most):
        return draw(st.one_of(st.just(0), st.integers(1, most)))

    days = part(106_750)  # the most whole days, with any rest, below MAX_NS
    total = days
    for most, carry in [(23, 24), (59, 60), (59, 60), (999, 1000), (999, 1000), (999, 1000)]:
        total = total * carry + part(most)
    return total or 1


#: A written duration: whole counts, largest unit first, no zero parts.
CANONICAL = re.compile(
    r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+)s)?(?:(\d+)ms)?(?:(\d+)us)?(?:(\d+)ns)?"
)
#: Each written part's length, and the most it can be before it carries.
PARTS = [
    (UNIT_NS["d"], 106_751),
    (UNIT_NS["h"], 23),
    (UNIT_NS["m"], 59),
    (UNIT_NS["s"], 59),
    (UNIT_NS["ms"], 999),
    (UNIT_NS["us"], 999),
    (UNIT_NS["ns"], 999),
]


def _halflife(value):
    """The text a spec stores for ``value`` given as a halflife."""
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0"], clock="t", max_dclock="inf", halflife=value
    )
    return spec["halflife"]


class TestDurationText:
    @TEXT
    @given(ns=st.one_of(_counts(1, MAX_NS), by_parts()))
    @example(ns=1)
    @example(ns=MAX_NS)
    @example(ns=86_400 * 10**9)
    def test_format_then_parse_is_the_identity(self, ns):
        text = format_duration(ns)
        assert parse_duration(text) == ns, text
        assert parse_duration(f"-{text}") == -ns and format_duration(-ns) == f"-{text}"
        # Written largest unit first, zero parts left out, each part below
        # the count that would carry it into the next unit.
        m = CANONICAL.fullmatch(text)
        assert m is not None and text, text
        written = [(g, per, most) for g, (per, most) in zip(m.groups(), PARTS, strict=True) if g]
        assert written, text
        for g, _, most in written:
            assert g[0] != "0" and 1 <= int(g) <= most, text
        assert sum(int(g) * per for g, per, _ in written) == ns, text

    @TEXT
    @given(case=fixed_texts())
    def test_the_text_means_what_polars_reads_it_as(self, case):
        """polars' own parser is the oracle: the offset ``dt.offset_by``
        applies to the epoch on a zone-free nanosecond clock. ``d`` and
        ``w`` are calendar units there, but a zone-free clock has no summer
        time, so they are 24 and 168 hours (the test below shows both). The
        lengths stay within an i64, past which polars itself wraps."""
        text, exact = case
        assert parse_duration(text) == exact
        assert _polars_offset(text) == exact, text

    def test_days_and_weeks_are_fixed_on_the_oracles_zone_free_clock(self):
        """Why the oracle is zone-free: across New York's spring change a
        day is 23 hours on a zoned clock, and 24 on a zone-free one."""
        start = datetime(2024, 3, 9, 12)
        for zone, day_hours in [(None, 24), ("America/New_York", 23)]:
            t = pl.Series([start]).cast(pl.Datetime("ns")).dt.replace_time_zone(zone)
            for unit, hours in [("1d", day_hours), ("1w", 6 * 24 + day_hours)]:
                gap = (t.dt.offset_by(unit) - t).dt.total_nanoseconds().item()
                assert gap == hours * UNIT_NS["h"], (zone, unit)
        assert _polars_offset("1d") == _polars_offset("24h")
        assert _polars_offset("1w") == _polars_offset("168h")

    @TEXT
    @given(case=st.one_of(fixed_texts(most=None), boundary_texts(), fixed_texts()))
    def test_a_length_past_an_i64_is_refused_by_name_and_never_wraps(self, case):
        """The unchecked sum once read ``"1e37w"`` as about 218 years. A text
        is its exact length or a refusal, never another length."""
        text, exact = case
        if abs(exact) <= MAX_NS:
            assert parse_duration(text) == exact, text
            return
        with pytest.raises(ValueError) as e:
            parse_duration(text)
        said = str(e.value)
        assert said.startswith(f'"{text}" is not a duration: '), said
        assert "longer than 292 years, the most a clock can hold" in said or re.search(
            r": \d+ is too large$", said
        ), said
        with pytest.raises(ValueError, match=re.escape(f'spec "m": halflife "{text}" is not')):
            _halflife(text)

    @SPECS
    @given(case=fixed_texts(signs=("", "+"), positive=True), data=st.data())
    def test_padding_is_trimmed_from_the_stored_text_and_the_field_names(self, case, data):
        text, exact = case
        lead = data.draw(st.text(WHITESPACE, max_size=3), label="lead")
        trail = data.draw(st.text(WHITESPACE, max_size=3), label="trail")
        padded = lead + text + trail
        assert parse_duration(padded) == exact
        other = format_duration(2 if exact == 1 else 1)
        # The Python side strips it ...
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            max_dclock="inf",
            halflife=[padded, other],
        )
        assert spec["halflife"] == [text, other]
        assert f"pred_y@h{text}" in po.spec.output_fields(spec)
        # ... and the Rust side, which a config or a hand-written dict reaches
        # without it, trims it too; the state keeps the trimmed text.
        bank = po.ModelBank([{**spec, "halflife": [padded, other]}])
        assert f"pred_y@h{text}" in bank.output_fields()["m"]
        assert po.ModelBank.load_bytes(bank.save_bytes()).specs[0]["halflife"] == [text, other]

    @SPECS
    @given(case=fixed_texts(), space=st.sampled_from(WHITESPACE), data=st.data())
    def test_whitespace_inside_a_duration_is_refused(self, case, space, data):
        """Anywhere inside, a sign included: the text names a grid's fields,
        and polars refuses it too."""
        text, _ = case
        at = data.draw(st.integers(1, len(text) - 1), label="at")
        spaced = text[:at] + space + text[at:]
        with pytest.raises(ValueError, match="has a space in it"):
            parse_duration(spaced)
        with pytest.raises(ValueError, match='spec "m": halflife .*has a space in it'):
            _halflife(spaced)


@st.composite
def keyword_lengths(draw):
    """A positive length a clock holds, as counts of distinct units that both
    keyword forms name, in a random order: ``{unit: count}``."""
    units = draw(st.lists(st.sampled_from(list(UNIT_KEYWORD)), min_size=1, max_size=8, unique=True))
    parts, total = {}, 0
    for i, unit in enumerate(units):
        n = draw(_counts(1 if i == 0 else 0, (MAX_NS - total) // UNIT_NS[unit]))
        parts[unit] = n
        total += n * UNIT_NS[unit]
    return parts


class TestTheThreeForms:
    @SPECS
    @given(parts=keyword_lengths())
    def test_one_length_is_one_text_in_every_form(self, parts):
        """polars' text is kept as written; ``pl.duration`` and ``timedelta``
        are written by the formatter, so they store the canonical text, which
        the text form stores as it is too."""
        exact = sum(n * UNIT_NS[u] for u, n in parts.items())
        canonical = format_duration(exact)
        text = "".join(f"{n}{u}" for u, n in parts.items())
        assert _halflife(text) == text and parse_duration(text) == exact
        assert _halflife(canonical) == canonical
        assert _halflife(pl.duration(**{UNIT_KEYWORD[u]: n for u, n in parts.items()})) == canonical
        if not parts.get("ns"):  # a timedelta holds nothing finer than a microsecond
            td = timedelta(**{UNIT_KEYWORD[u]: n for u, n in parts.items() if u != "ns"})
            assert _halflife(td) == canonical

    @SPECS
    @given(data=st.data())
    def test_a_pl_duration_past_an_i64_is_refused_by_name_and_never_wraps(self, data):
        """``pl.duration`` holds microseconds, so it reaches a thousand times
        past what nanoseconds hold; that far is refused, as its text is."""
        unit = data.draw(st.sampled_from([u for u in UNIT_KEYWORD if u != "ns"]), label="unit")
        per_us = UNIT_NS[unit] // 1_000
        n = data.draw(_counts(MAX_NS // UNIT_NS[unit] + 1, MAX_NS // per_us), label="n")
        _assert_refused_as_too_long(pl.duration(**{UNIT_KEYWORD[unit]: n}), n * UNIT_NS[unit])

    @SPECS
    @given(data=st.data())
    def test_a_timedelta_past_an_i64_is_refused_by_name(self, data):
        unit = data.draw(st.sampled_from([u for u in UNIT_KEYWORD if u != "ns"]), label="unit")
        most = timedelta.max // timedelta(**{UNIT_KEYWORD[unit]: 1})
        n = data.draw(_counts(MAX_NS // UNIT_NS[unit] + 1, most), label="n")
        _assert_refused_as_too_long(timedelta(**{UNIT_KEYWORD[unit]: n}), n * UNIT_NS[unit])


def _polars_offset(text: str) -> int:
    """The offset polars' own parser reads ``text`` as, in nanoseconds: from
    the epoch, on a zone-free nanosecond clock. polars spells a microsecond
    ``us`` only."""
    text = text.replace("\u00b5s", "us").replace("\u03bcs", "us")
    t = pl.Series([0]).cast(pl.Datetime("ns"))
    return t.dt.offset_by(text).cast(pl.Int64).item()


def _assert_refused_as_too_long(value, exact: int) -> None:
    """``value``, ``exact`` nanoseconds long, is refused as a halflife by
    name and for its length, and never stored as some other length."""
    try:
        stored = _halflife(value)
    except ValueError as e:
        said = str(e)
        assert said.startswith('spec "m": halflife') and "292 years" in said, said
    else:
        raise AssertionError(
            f"{value!r}, {exact} ns long, was stored as {stored!r}, "
            f"which is {parse_duration(stored)} ns"
        )


# --- temporal clocks ----------------------------------------------------------

#: Zones with summer time (New York), a half-hour change (Lord Howe) and an
#: offset in quarter hours (Chatham); none of them moves an instant.
ZONES = ["UTC", "America/New_York", "Australia/Lord_Howe", "Pacific/Chatham"]


@st.composite
def clock_dtypes(draw):
    """Every kind of column a clock can be: a ``Datetime`` in each unit,
    zone-free or zoned, a ``Date``, and a ``Duration`` in each unit."""
    kind = draw(st.sampled_from(["Datetime", "Date", "Duration"]))
    if kind == "Date":
        return pl.Date
    unit = draw(st.sampled_from(["ms", "us", "ns"]))
    if kind == "Duration":
        return pl.Duration(unit)
    return pl.Datetime(unit, draw(st.one_of(st.none(), st.sampled_from(ZONES))))


def _step(dtype) -> int:
    """The smallest step the column takes, in nanoseconds."""
    return UNIT_NS["d"] if dtype == pl.Date else UNIT_NS[dtype.time_unit]


#: Where a stream starts, in nanoseconds since 1970: 1970 itself, 2024
#: (``START``), and 2**61 either side of 1970, which is 1897 and 2043. Drawn
#: across that range, an instant would sit near 1970: hypothesis draws an
#: integer with few digits.
ANCHORS = [0, START * 10**9, -(2**61), 2**61]


def _instants(step: int):
    """An instant within a thousand steps of an anchor, in whole steps."""
    anchor = st.sampled_from(ANCHORS).map(lambda a: a // step * step)
    return st.builds(lambda a, k: a + k * step, anchor, st.integers(-1_000, 1_000))


#: The models with clock parameters of their own besides the shared ones.
CLOCK_MODELS = [
    "ewridge",
    "lasso",
    "kalman",
    "holt",
    "ew_cov",
    "ew_class",
    "rls",
    "huber",
    "quantile",
]


@st.composite
def clock_specs(draw, model: str, step: int, cap: int):
    """One ``model`` spec on clock ``t``, grouped by ``g``, each clock
    parameter a duration in whole steps of the column, ``max_dclock``
    ``cap`` steps, the optional ones drawn in or left out."""

    def steps(lo, hi):
        return format_duration(draw(st.integers(lo, hi)) * step)

    def maybe(key, lo, hi):
        return {key: steps(lo, hi)} if draw(st.booleans()) else {}

    kw: dict = dict(clock="t", group="g", max_dclock=format_duration(cap * step))
    if draw(st.booleans()):
        kw["weight"] = "w"
    xy = dict(targets=["y"], features=["x0", "x1"], halflife=steps(1, 30))
    if model == "ewridge":
        kw |= xy | maybe("window", 1, 60) | maybe("solve_every", 1, 5) | maybe("label_delay", 1, 5)
        if draw(st.booleans()):
            gap = draw(
                st.one_of(
                    st.just("reset"), st.integers(1, cap).map(lambda k: format_duration(k * step))
                )
            )
            kw |= dict(session="s", session_gap=gap)
            # The slow twin is a second history, which a window would cut.
            if gap != "reset" and "window" not in kw and draw(st.booleans()):
                kw |= dict(session_shrink=0.5, long_halflife=steps(1, 120))
    elif model == "lasso":
        kw |= xy | dict(lasso_path=[0.1, 0.0], select_halflife=steps(1, 60))
        kw |= maybe("window", 1, 60) | maybe("solve_every", 1, 5)
    elif model == "kalman":
        kw |= xy | dict(coef_halflife=steps(1, 60)) | maybe("revert_halflife", 1, 600)
    elif model == "holt":
        kw |= dict(targets=["y"], level_halflife=steps(1, 30), trend_halflife=steps(1, 60))
    elif model == "ew_cov":
        kw |= dict(features=["x0", "x1"], halflife=steps(1, 30)) | maybe("window", 1, 60)
    elif model == "ew_class":
        kw |= dict(label="lab", classes=["a", "b"], precision_prior=1.0, features=["x0", "x1"])
        kw |= dict(halflife=steps(1, 30)) | maybe("window", 1, 60)
    elif model in ("huber", "quantile"):
        kw |= xy | maybe("solve_every", 1, 5) | ({"quantile": 0.5} if model == "quantile" else {})
    else:
        assert model == "rls", model
        kw |= xy | maybe("label_delay", 1, 5)
    return getattr(po.spec, model)("m", **kw)


@st.composite
def temporal_streams(draw, model: str, min_rows=1, max_rows=40):
    """A stream on a temporal clock and one ``model`` spec to fit it with.

    Each group's clock is non-decreasing (disorder is its own test) in whole
    steps of the column: ties, gaps of a few steps and gaps past
    ``max_dclock``, from an instant near one of ``ANCHORS``. Features,
    targets and weights are seeded, with nulls, and zero weights, which
    advance the clock and teach nothing."""
    dtype = draw(clock_dtypes(), label="dtype")
    step = _step(dtype)
    cap = draw(st.integers(1, 8), label="max_dclock in steps")
    n = draw(st.integers(min_rows, max_rows), label="rows")
    n_groups = draw(st.integers(1, 3))
    groups = draw(st.lists(st.integers(0, n_groups - 1), min_size=n, max_size=n), label="groups")
    gap = st.one_of(st.sampled_from([0, 1, 2, 3]), st.integers(cap + 1, 4 * cap + 4))
    last = dict.fromkeys(range(n_groups), draw(_instants(step), label="start"))
    instants = []
    for g in groups:
        last[g] += draw(gap) * step
        instants.append(last[g])
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1), label="seed"))

    def nulled(values, p):
        return [None if u < p else v for v, u in zip(values.tolist(), rng.random(n), strict=True)]

    x = rng.normal(size=(n, 2))
    y = 0.5 + x @ np.array([1.0, -2.0]) + rng.normal(0.0, 0.3, n)
    w = np.where(rng.random(n) < 0.1, 0.0, rng.uniform(0.1, 2.0, n))
    df = pl.DataFrame(
        {
            "g": [f"g{g}" for g in groups],
            "t": _column(dtype, np.array(instants, dtype=np.int64)),
            "x0": nulled(x[:, 0], 0.08),
            "x1": nulled(x[:, 1], 0.08),
            "y": nulled(y, 0.15),
            "w": nulled(w, 0.05),
            "s": [f"s{k}" for k in np.cumsum(rng.random(n) < 0.15)],
        },
        schema_overrides={c: pl.Float64 for c in ("x0", "x1", "y", "w")},
    ).with_columns(lab=pl.when(pl.col("y") > 0.5).then(pl.lit("a")).otherwise(pl.lit("b")))
    df = df.with_columns(lab=pl.when(pl.col("y").is_null()).then(None).otherwise(pl.col("lab")))
    spec = draw(clock_specs(model, step, cap), label="spec")
    return df, spec


def _edges(draw, n: int) -> list[int]:
    """Chunk boundaries over ``n`` rows: up to ten cuts anywhere, empty
    chunks included, or a chunk per row, the "1000 chunks" of hard rule 3,
    which puts a boundary between every two rows a delayed label spans."""
    if draw(st.booleans(), label="a chunk per row"):
        return list(range(n + 1))
    cuts = draw(st.lists(st.integers(0, n), max_size=10), label="cuts")
    return [0, *sorted(cuts), n]


def _assert_same_numbers(one: pl.Series, many: pl.Series) -> None:
    """Every field of the one-chunk output equals the chunked one's. The
    lists (``coef``, ``support_coef``) are emitted on each group's last row
    of each chunk, by design, so they are compared on the rows the one-chunk
    run emits them, which are such a row in every chunking."""
    assert one.len() == many.len()
    for f in one.dtype.fields:
        a, b = one.struct.field(f.name), many.struct.field(f.name)
        if isinstance(f.dtype, pl.List):
            emitted = a.is_not_null()
            a, b = a.filter(emitted), b.filter(emitted)
        assert a.equals(b, null_equal=True), f.name


@pytest.mark.parametrize("model", CLOCK_MODELS)
class TestTemporalClockStreams:
    # A tiny or stale fit gets the readiness notice; it is beside the point.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @STREAMS
    @given(data=st.data())
    def test_chunking_never_changes_the_numbers(self, model, data):
        df, spec = data.draw(temporal_streams(model), label="stream")
        one = po.ModelBank([spec]).fit_predict(df)["m"]
        bank = po.ModelBank([spec])
        edges = _edges(data.draw, df.height)
        many = pl.concat([bank.fit_predict(df.slice(a, b - a))["m"] for a, b in pairwise(edges)])
        _assert_same_numbers(one, many)

    # A tiny or stale fit gets the readiness notice; it is beside the point.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @STREAMS
    @given(data=st.data())
    def test_a_save_and_load_at_any_row_is_transparent(self, model, data):
        df, spec = data.draw(temporal_streams(model, min_rows=2), label="stream")
        split = data.draw(st.integers(1, df.height - 1), label="split")
        kept = po.ModelBank([spec])
        head = kept.fit_predict(df.slice(0, split))["m"]
        loaded = po.ModelBank.load_bytes(kept.save_bytes())
        assert loaded.specs == [spec]  # the durations survive as their text
        rest = df.slice(split)
        tail = loaded.fit_predict(rest)["m"]
        assert tail.equals(kept.fit_predict(rest)["m"], null_equal=True)
        _assert_same_numbers(po.ModelBank([spec]).fit_predict(df)["m"], pl.concat([head, tail]))


#: The models a delayed label is fitted with, and what each needs besides.
DELAYED_MODELS = {
    "ewridge": {},
    "rls": {},
    "kalman": {"coef_halflife": "1h"},
    "huber": {},
    "quantile": {"quantile": 0.5},
}


@st.composite
def delayed_streams(draw):
    """A stream whose spec holds each row's label back ``label_delay``, a
    few steps, so several labels are in flight across every chunk boundary.
    The column steps in fractions of a second, whose sums a double rounds
    by their order; whole seconds and days sum exactly in any order."""
    unit = draw(st.sampled_from(["ms", "us", "ns"]), label="unit")
    dtype = draw(st.sampled_from([pl.Datetime(unit), pl.Datetime(unit, "UTC"), pl.Duration(unit)]))
    step = UNIT_NS[unit]
    cap = draw(st.integers(1, 8), label="max_dclock in steps")
    n = draw(st.integers(2, 30), label="rows")
    gaps = draw(st.lists(st.integers(0, cap + 2), min_size=n, max_size=n), label="gaps")
    instants = draw(_instants(step), label="start") + np.cumsum(gaps) * step
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1), label="seed"))
    x = rng.normal(size=n)
    y = x + rng.normal(0.0, 0.3, n)
    df = pl.DataFrame(
        {
            "t": _column(dtype, instants.astype(np.int64)),
            "x0": x,
            "y": [None if u < 0.15 else v for v, u in zip(y.tolist(), rng.random(n), strict=True)],
        },
        schema_overrides={"y": pl.Float64},
    )
    model = draw(st.sampled_from(sorted(DELAYED_MODELS)), label="model")
    spec = getattr(po.spec, model)(
        "m",
        targets=["y"],
        features=["x0"],
        clock="t",
        halflife=format_duration(draw(st.integers(1, 30)) * step),
        max_dclock=format_duration(cap * step),
        label_delay=format_duration(draw(st.integers(1, 12), label="delay in steps") * step),
        **DELAYED_MODELS[model],
    )
    return df, spec


class TestADelayedLabel:
    # A tiny or stale fit gets the readiness notice; it is beside the point.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @settings(STREAMS, max_examples=150)
    @given(data=st.data())
    def test_labels_in_flight_across_chunks_change_no_number(self, data):
        """Hard rule 3 where the stream properties above rarely reach it:
        rows parked for their label at a chunk boundary, carried in the
        state and released in a later chunk. Drawn among the other clock
        parameters there, a case that tells came up once in 40 to 370
        examples; here, once in 4 to 50 (measured 2026-09-24)."""
        df, spec = data.draw(delayed_streams(), label="stream")
        one = po.ModelBank([spec]).fit_predict(df)["m"]
        bank = po.ModelBank([spec])
        edges = _edges(data.draw, df.height)
        many = pl.concat([bank.fit_predict(df.slice(a, b - a))["m"] for a, b in pairwise(edges)])
        _assert_same_numbers(one, many)


#: Every model that emits ``n_eff``, with what each needs besides a halflife.
N_EFF_MODELS = {
    "ewridge": {},
    "rls": {},
    "lasso": {"lasso_path": [0.1, 0.0]},
    "kalman": {"coef_halflife": float("inf")},
    "huber": {},
    "quantile": {"quantile": 0.5},
    "ftrl": {},
    "sgd": {"learning_rate": 0.01},
    "pa": {},
    "holt": {"features": None},
    "ew_cov": {"targets": None},
    "ew_class": {"targets": None, "label": "lab", "classes": ["a", "b"], "precision_prior": 1.0},
    "marginal": {},
}


@st.composite
def late_streams(draw):
    """A stream's first instant, then ticks a few halflives apart from one
    to two hundred years later, with the halflife, the gaps and the cap in
    whole steps of the column. The age is drawn in years: drawn in steps,
    it would sit near the first instant, where a double is still exact."""
    dtype = draw(clock_dtypes(), label="dtype")
    step = _step(dtype)
    halflife = draw(st.integers(1, 50), label="halflife in steps") * step
    cap = draw(st.one_of(st.none(), st.integers(1, 200)), label="max_dclock in steps")
    gaps = draw(st.lists(st.integers(0, 4 * halflife // step), min_size=1, max_size=60))
    gaps = [g * step for g in gaps]
    first = draw(_instants(step), label="first instant")
    # Four years to spare: the extra thousand steps are 2.7 years of days.
    years = draw(st.integers(1, min(200, (MAX_NS - first - sum(gaps)) // YEAR_NS - 4)))
    age = years * YEAR_NS // step * step + draw(st.integers(0, 1_000)) * step
    instants = [first, *(first + age + np.cumsum(gaps)).tolist()]
    assert instants[-1] <= MAX_NS
    return dtype, instants, halflife, None if cap is None else cap * step


class TestNanosecondsAreExact:
    # A tiny or stale fit gets the readiness notice; it is beside the point.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @pytest.mark.parametrize("model", sorted(N_EFF_MODELS))
    @STREAMS
    @given(case=late_streams(), data=st.data())
    def test_n_eff_is_the_exact_recursion_at_any_age(self, model, case, data):
        """``w <- w * 2**(-min(gap, cap) / h) + 1`` on the exact integer gaps,
        ``n_eff`` being ``w`` before the row's own update (hard rule 8), in
        every model and fed in any chunks. Read as a double of seconds, even
        from the stream's first instant, the gaps would be rounded to the
        double's resolution at that age: 1.4 us two centuries in."""
        dtype, instants, halflife, cap = case
        n = len(instants)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, 2))
        df = pl.DataFrame(
            {
                "t": _column(dtype, np.array(instants, dtype=np.int64)),
                "x0": x[:, 0],
                "x1": x[:, 1],
                "y": (x[:, 0] > 0).astype(float),
                "lab": np.where(x[:, 1] > 0, "a", "b"),
            }
        )
        kw = dict(
            clock="t", halflife=format_duration(halflife), targets=["y"], features=["x0", "x1"]
        )
        kw |= N_EFF_MODELS[model]
        kw["max_dclock"] = float("inf") if cap is None else format_duration(cap)
        spec = getattr(po.spec, model)("m", **{k: v for k, v in kw.items() if v is not None})
        bank = po.ModelBank([spec])
        edges = _edges(data.draw, n)
        got = pl.concat([bank.fit_predict(df.slice(a, b - a))["m"] for a, b in pairwise(edges)])
        got = got.struct.field("n_eff").to_numpy()
        want, w = [], 0.0
        for i in range(n):
            want.append(w)
            gap = instants[i] - instants[i - 1] if i else 0
            w = w * 2.0 ** (-(gap if cap is None else min(gap, cap)) / halflife) + 1.0
        want = np.array(want)
        assert got[0] == 0.0
        assert np.all(np.abs(got - want) <= 1e-12 * want), np.max(
            np.abs(got - want) / np.maximum(want, 1)
        )
