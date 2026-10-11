"""A name renamed after 1.0 is read as its new one, with a warning, until the
next major version refuses it (docs/PLAN.md task 198; review round 4, D2, AP9,
CI4 and DA16).

The shared table is built and tested and empty: a rename made before 1.0
stays refused by name (task 144's rule, ``tests/test_renames.py``), and one
made after goes in the forwarding table -- ``_DEPRECATED`` for a builder's
keyword and a spec dict's key, and its twin ``online_polars::DEPRECATED`` for
a TOML file, which the command line forwards the same way
(``crates/online-cli/src/main.rs``). Each entry is removed at the next major,
when the old name moves to the refusal table.

The names standardized for 1.0 (task 244) are forwarded already, so a spec
written for 0.13 runs on 1.0 with a warning: one model's own keys and words
in ``_DEPRECATED_IN`` and ``_DEPRECATED_WORDS_IN``, with twins of the same
names on the Rust side, and ``bocpd``'s ``emission`` and ``corrchange``'s
``reset_on_flag``, which became other keys with values of their own
(``tests/test_builder_keywords.py``).
"""

from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import polars as pl
import pytest

import polars_online as po
from polars_online import _warnings
from polars_online._spec import _RENAMED
from polars_online._warnings import _DEPRECATED, forward_deprecated

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]


def _rust_table() -> dict[str, str]:
    """``online_polars::DEPRECATED`` as spec.rs writes it."""
    text = (REPO / "crates/online-polars/src/spec.rs").read_text(encoding="utf-8")
    body = re.search(r"pub const DEPRECATED: &\[\(&str, &str\)\] = &\[(.*?)\];", text, re.S)
    assert body, "spec.rs keeps its forwarding table as DEPRECATED"
    return dict(re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', body.group(1)))


def _rust_model_table() -> dict[str, dict[str, str]]:
    """``online_polars::DEPRECATED_IN`` as spec/names.rs writes it."""
    text = (REPO / "crates/online-polars/src/spec/names.rs").read_text(encoding="utf-8")
    body = re.search(
        r"pub const DEPRECATED_IN: &\[\(&str, &str, &str\)\] = &\[(.*?)\];", text, re.S
    )
    assert body, "spec/names.rs keeps its per-model forwarding table as DEPRECATED_IN"
    out: dict[str, dict[str, str]] = {}
    for kind, old, new in re.findall(r'\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)', body.group(1)):
        out.setdefault(kind, {})[old] = new
    return out


def _rust_word_table() -> dict[str, dict[str, dict[str, str]]]:
    """``online_polars::DEPRECATED_WORDS_IN`` as spec/names.rs writes it."""
    text = (REPO / "crates/online-polars/src/spec/names.rs").read_text(encoding="utf-8")
    body = re.search(
        r"pub const DEPRECATED_WORDS_IN: &\[\(&str, &str, &str, &str\)\] = &\[(.*?)\];", text, re.S
    )
    assert body, "spec/names.rs keeps its per-model word table as DEPRECATED_WORDS_IN"
    out: dict[str, dict[str, dict[str, str]]] = {}
    rows = re.findall(r'\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)', body.group(1))
    for kind, key, old, new in rows:
        out.setdefault(kind, {}).setdefault(key, {})[old] = new
    return out


def test_the_per_model_tables_agree_and_name_current_keys():
    """The renames made at 1.0 in one model alone (``kmeans``' ``k`` is
    ``n_clusters``, ``hmm``'s ``n_components``): the two sides keep one
    table, each new name is a keyword of the model's builder, and no old
    name is still one."""
    assert _rust_model_table() == _warnings._DEPRECATED_IN
    assert _rust_word_table() == _warnings._DEPRECATED_WORDS_IN
    for kind, table in _warnings._DEPRECATED_IN.items():
        params = inspect.signature(getattr(po.spec, kind)).parameters
        for old, new in table.items():
            assert new in params and old not in params, (kind, old, new)


def test_no_test_and_no_recipe_meets_a_deprecated_name():
    """A deprecated name is an error under the suite, so no test, and no
    recipe of the README, a guide or a docstring that the suite runs, reads
    an old name without saying so with ``pytest.warns``."""
    config = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"error::polars_online.PolarsOnlineDeprecationWarning"' in config
    with pytest.raises(po.PolarsOnlineDeprecationWarning):
        po.spec.kmeans("m", features=["x0"], half_life=10.0, k=2)


def test_the_warning_is_a_deprecation_warning_and_exported():
    assert issubclass(po.PolarsOnlineDeprecationWarning, DeprecationWarning)
    assert "PolarsOnlineDeprecationWarning" in po.__all__


def test_the_tables_are_empty_before_1_0_and_agree():
    """No rename made so far is forwarded: each is refused by name. The two
    sides keep the same table, and no name is both refused and forwarded."""
    if int(po.__version__.split(".")[0]) < 1:
        assert _DEPRECATED == {}
    assert _rust_table() == _DEPRECATED
    assert not set(_DEPRECATED) & set(_RENAMED)


def test_an_old_name_is_forwarded_with_the_warning():
    with pytest.warns(
        po.PolarsOnlineDeprecationWarning, match="old is deprecated: it was renamed new"
    ):
        got = forward_deprecated("spec 'm'", {"old": 1, "other": 2}, {"old": "new"})
    assert got == {"new": 1, "other": 2}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert forward_deprecated("spec 'm'", {"new": 1}, {"old": "new"}) == {"new": 1}


def test_an_old_name_beside_its_new_one_is_refused_naming_both():
    with pytest.raises(TypeError, match="old was renamed new, and both are given"):
        forward_deprecated("spec 'm'", {"old": 1, "new": 2}, {"old": "new"})


@pytest.fixture
def renamed(monkeypatch):
    """A table with one entry, as a rename after 1.0 would add: the model's
    ``ridge`` once called ``rigde``, and the spec's ``half_life`` once
    ``half_lyfe``."""
    monkeypatch.setattr(_warnings, "_DEPRECATED", {"rigde": "ridge", "half_lyfe": "half_life"})


def test_a_builder_reads_an_old_keyword_as_the_new_one(renamed):
    new = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0, ridge=0.5)
    with pytest.warns(po.PolarsOnlineDeprecationWarning) as caught:
        old = po.spec.ewridge("m", targets=["y"], features=["x"], half_lyfe=10.0, rigde=0.5)
    assert old == new
    messages = sorted(str(w.message) for w in caught)
    assert any("half_lyfe is deprecated: it was renamed half_life" in m for m in messages)
    assert all("next major version" in m for m in messages)


def test_a_spec_dict_reads_an_old_key_as_the_new_one(renamed):
    """At any depth: the spec's own key and its model's."""
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0, ridge=0.5)
    old = dict(spec, half_lyfe=spec["half_life"], model=dict(spec["model"]))
    del old["half_life"]
    old["model"]["rigde"] = old["model"].pop("ridge")
    with pytest.warns(po.PolarsOnlineDeprecationWarning):
        bank = po.ModelBank([old])
    assert bank.specs == po.ModelBank([spec]).specs


def test_without_an_entry_an_unknown_name_is_still_refused():
    with pytest.raises(TypeError, match="unexpected keyword argument 'rigde'"):
        po.spec.ewridge("m", targets=["y"], features=["x"], rigde=0.5)


def test_a_like_spec_reads_an_old_key_as_the_new_one(monkeypatch):
    """``with_windows(like=)`` reads its clock policy from a spec dict, so a
    spec key renamed after 1.0 is read as its new name there too, with the
    warning: the policy names it among what the table forwards (review round
    5, D4)."""
    monkeypatch.setattr(_warnings, "_DEPRECATED", {"gap_kap": "gap_cap"})
    df = pl.DataFrame({"t": [0.0, 1.0, 9.0, 10.0], "x": [1.0, 2.0, 3.0, 4.0]})
    like = {"name": "m", "features": ["x"], "clock": "t", "gap_cap": 2.0}
    f = po.ewm_mean("x", half_life=2.0)
    want = po.stream.with_windows(df, f=f, like=like)
    old = {**{k: v for k, v in like.items() if k != "gap_cap"}, "gap_kap": 2.0}
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="gap_kap is deprecated"):
        got = po.stream.with_windows(df, f=f, like=old)
    assert got.equals(want)
    # The cap reached the numbers: without it the 8-unit gap is read whole.
    wide = po.stream.with_windows(df, f=f, like={**like, "gap_cap": 100.0})
    assert not got.equals(wide)
