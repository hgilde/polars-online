"""The bank's file handling on the paths no other test took (task 252): a
``save``, ``write_json`` or ``load`` handed something that is neither a path
nor a file object of the right mode, and ``write_json`` into a file object
that takes text alone. Each refusal is named and leaves the bank as it was."""

from __future__ import annotations

import io
import json

import polars as pl
import pytest

import polars_online as po

TIER = "essential"


@pytest.fixture
def bank() -> po.ModelBank:
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0)
    b = po.ModelBank([spec])
    b.fit_predict(pl.DataFrame({"x": [1.0, 3.0, 5.0, 2.0], "y": [1.0, 2.0, 4.0, 1.5]}))
    return b


def test_save_refuses_what_is_neither_a_path_nor_a_writer(bank):
    with pytest.raises(
        TypeError,
        match="ModelBank.save takes a path or a file object opened for binary writing, got int",
    ):
        bank.save(123)  # type: ignore[arg-type]
    # The refusal wrote nothing and left the bank whole: it saves after it.
    buf = io.BytesIO()
    bank.save(buf)
    assert buf.getvalue() == bank.save_bytes()


class _TextOnly:
    """A file object that is not an ``io.TextIOBase`` and takes ``str``
    alone, as a hand-written writer or a logging adapter may."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.refused = 0

    def write(self, s: object) -> int:
        if not isinstance(s, str):
            self.refused += 1
            raise TypeError("text only")
        self.parts.append(s)
        return len(s)


def test_write_json_gives_a_text_only_writer_the_text(bank):
    """Bytes are tried first; a writer that refuses them with ``TypeError``
    is handed the same text as a string."""
    w = _TextOnly()
    assert bank.write_json(w) is None
    assert w.refused == 1, "the bytes were offered first"
    assert "".join(w.parts) == bank.write_json()
    assert json.loads("".join(w.parts)) == json.loads(bank.write_json())


def test_write_json_refuses_what_is_neither_a_path_a_writer_nor_none(bank):
    with pytest.raises(
        TypeError, match="ModelBank.write_json takes a path, a file object or None, got int"
    ):
        bank.write_json(123)  # type: ignore[call-overload]


def test_load_refuses_a_text_mode_file_and_what_is_not_a_source(bank):
    with pytest.raises(
        TypeError,
        match=r"ModelBank.load: the file object gave str; open the file in binary mode \('rb'\)",
    ):
        po.ModelBank.load(io.StringIO("not msgpack"))  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match="ModelBank.load takes a path, bytes or a file object opened for binary reading, "
        "got int",
    ):
        po.ModelBank.load(123)  # type: ignore[arg-type]
    # The binary file object of the same state loads.
    back = po.ModelBank.load(io.BytesIO(bank.save_bytes()))
    assert back.save_bytes() == bank.save_bytes()


def test_a_keyword_once_positional_is_refused_past_its_count_or_given_twice(bank):
    """``fit``'s ``closed_groups`` and ``chunk_size`` were positional once
    and are read by position with a deprecation warning; a call with more
    positional arguments than that, or with one of them given both ways, is
    refused as Python refuses it for any function."""
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    before = bank.save_bytes()
    with pytest.raises(
        TypeError, match=r"ModelBank.fit\(\) takes 2 positional arguments but 5 were given"
    ):
        bank.fit(df, None, 10, "extra")  # type: ignore[misc]
    with pytest.raises(
        TypeError, match=r"ModelBank.fit\(\) got multiple values for argument 'closed_groups'"
    ):
        bank.fit(df, None, closed_groups=None)  # type: ignore[misc]
    assert bank.save_bytes() == before, "neither call read a row"


def test_one_spec_or_a_string_in_place_of_the_list_is_refused_by_name():
    """A dict and a string are iterable, so either read as a list was its
    keys or its characters, refused as an invalid spec of type string; the
    refusal now names the mistake and the fix."""
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0)
    with pytest.raises(
        ValueError,
        match=r"ModelBank takes a list of specs, and got one spec: pass \[spec\] for a bank of one",
    ):
        po.ModelBank(spec)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"ModelBank takes a list of specs, such as \[po.spec.ewridge\(...\)\], got str",
    ):
        po.ModelBank("m")  # type: ignore[arg-type]
    assert po.ModelBank([spec]).specs[0]["name"] == "m"
