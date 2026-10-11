"""Every word a spec parameter takes, written down once and refused one way.

A parameter that takes one of a few words (``closed``, ``loss``,
``target_gaps``...) is annotated with ``Literal[...]`` naming them, so an
editor completes the words and a type checker finds a misspelt one. The
builders' own check reads the same annotation, so a word outside it is
refused in Python, before anything is serialized. And every surface refuses
it in one shape, naming the spec, the parameter and the words:

    spec "m": <param> must be one of "a", "b"; got "x"

A builder in Python, a spec dict at the bank's door and a TOML file read by
the same Rust validation all say that. Each of serde's own enums once said
``invalid spec: model.target_gaps: unknown variant `x`, expected ...`` with
no spec name, and the hand-written refusals said it five other ways.

The words are probed at the Rust door the way the API snapshot reads them
(``tests/test_api_surface.py``'s ``ENUM_PROBES``), so the annotation, the
builder's refusal and the Rust side's refusal are held to one list.
"""

from __future__ import annotations

import re
import typing

import pytest

import polars_online as po
from polars_online import _spec
from test_api_surface import BOGUS, ENUM_PROBES, _registry_spec, _shaped, _spec_refusal, taken_words

TIER = "essential"

#: The probes of parameters whose value is words alone. `session_gap` is a
#: duration or the word "reset": its refusal says what a duration is.
WORD_PROBES = [p for p in ENUM_PROBES if p[0] != "spec.session_gap"]


def _hint(label: str) -> object:
    """The annotation of the parameter a probe label names: a builder's
    own, or ``_common``'s for the shared ones."""
    owner, param = label.rsplit(".", 1)
    fn = _spec._common if owner == "spec" else getattr(po.spec, owner)
    return typing.get_type_hints(fn)[param]


def _literal_words(hint: object) -> tuple[str, ...]:
    """The words of the ``Literal`` anywhere in ``hint``."""
    if typing.get_origin(hint) is typing.Literal:
        return typing.get_args(hint)
    for arg in typing.get_args(hint):
        if words := _literal_words(arg):
            return words
    return ()


def _keyword(path: str) -> str:
    return path.removeprefix("model.")


#: `spec "Probe": <param> must be [an int or ]one of "a", "b"; got "zz_bogus"`,
#: and for `rcov`'s `kernel` the reason it is the only one after a colon.
_SHAPE = (
    r'^spec "Probe": {param} must be (an int or )?one of "[^"]+"(, "[^"]+")*; '
    r'got "zz_bogus"(: .+)?$'
)


@pytest.mark.parametrize("probe", WORD_PROBES, ids=[p[0] for p in WORD_PROBES])
def test_the_rust_side_refuses_a_word_in_one_shape(probe):
    """A spec dict at the bank's door, as a TOML spec meets the same
    validation: the spec, the parameter, the words and the word given."""
    label, name, needs, path, shape, *_ = probe
    spec, _ = _registry_spec(name, needs)
    said = _spec_refusal(spec, path, _shaped(BOGUS, shape))
    param = re.escape(_keyword(path))
    assert re.fullmatch(_SHAPE.format(param=param), said, flags=re.DOTALL), said


@pytest.mark.parametrize("probe", WORD_PROBES, ids=[p[0] for p in WORD_PROBES])
def test_the_annotation_names_the_words_the_rust_side_takes(probe):
    """The ``Literal`` a builder's parameter is annotated with is the list
    the Rust side takes, word for word: a word added on one side alone is a
    failure here, not a refusal a caller meets."""
    label, name, needs, path, shape, *per_word = probe
    spec, _ = _registry_spec(name, needs)
    beside: dict[str, dict[str, object]] = per_word[0] if per_word else {}

    def refuse(word: str) -> str:
        return _spec_refusal(spec, path, _shaped(word, shape), beside.get(word))

    words = _literal_words(_hint(label))
    assert words, f"{label} is not annotated with a Literal"
    assert sorted(words) == sorted(taken_words(label, refuse))


@pytest.mark.parametrize("probe", WORD_PROBES, ids=[p[0] for p in WORD_PROBES])
def test_a_builder_refuses_a_word_in_python_as_the_rust_side_does(probe):
    """The builder's own check reads the annotation, so the word is refused
    before the spec is serialized, with the Rust side's words in the Rust
    side's order (``rcov``'s ``kernel`` adds its reason there)."""
    label, name, needs, path, shape, *_ = probe
    spec, _ = _registry_spec(name, needs)
    rust = _spec_refusal(spec, path, _shaped(BOGUS, shape)).replace('"Probe"', '"m"')
    with pytest.raises(ValueError) as caught:
        _registry_spec(name, {**needs, _keyword(path): _shaped(BOGUS, shape)})
    python = str(caught.value)
    assert rust.startswith(python), (python, rust)
    assert python.endswith(f'; got "{BOGUS}"'), python


def test_a_word_parameter_given_a_value_of_another_kind_is_a_type_error():
    """Not a word at all is a type error naming the words."""
    with pytest.raises(TypeError, match=r'mode must be one of "pa", "pa1", "pa2", got int 3$'):
        po.spec.pa("m", targets=["y"], features=["x"], half_life=10.0, mode=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r'stats must be a list of strs, each one of "mean", '):
        po.spec.ew_cov("m", features=["x", "z"], half_life=10.0, stats="mean")  # type: ignore[arg-type]


def test_every_enum_string_parameter_of_a_builder_is_a_literal():
    """No builder parameter that takes a fixed set of words is typed as a
    bare ``str``: each one the snapshot probes carries its words, and no
    other ``str`` parameter of a builder names a word list in its
    refusal."""
    for label, *_ in WORD_PROBES:
        assert _literal_words(_hint(label)), label
