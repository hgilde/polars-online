"""The suite's two tiers (docs/TESTING.md, "Two tiers").

The *essentials* gate each commit while a task is in progress
(``scripts/gate.sh``); the *extended* tier, which is everything, runs before
every push (``scripts/gate.sh --extended``), in CI and before a release. A
test is extended by ``pytest.mark.extended(reason="...")``, on the test, on
one of its parameters, or as its module's ``pytestmark``. Every test module
says which it is in a module-level ``TIER``: ``"essential"`` (no test in it
is extended), ``"extended"`` (its ``pytestmark`` is), ``"mixed"`` (some
tests are) or ``"soak"`` (opt-in, ``pytest -m soak``).
``tests/test_tiers.py`` holds each module's ``TIER`` to its marks.

The property tests run in both tiers; the tier sets how many examples each
draws. ``HYPOTHESIS_PROFILE`` picks the Hypothesis profile: ``extended``, the
default, so a plain ``pytest`` is the full run, or ``essential``, which the
essentials gate sets. Both profiles start from the one Hypothesis loaded
itself, ``ci`` on a CI runner (derandomized, no example database) and
``default`` elsewhere. A property draws the count its file names under
``extended``, and at most ``ESSENTIAL_EXAMPLES`` (or the smaller count its
file names for the essentials) under ``essential``. Locally, Hypothesis's
example database (``.hypothesis/``) replays a failure either tier found
before it draws anything new.
"""

from __future__ import annotations

import os

from hypothesis import settings

#: The environment variable that picks the profile.
PROFILE_VARIABLE = "HYPOTHESIS_PROFILE"
PROFILES = ("essential", "extended")
#: The count of a property whose file names none, as before the tiers.
EXTENDED_EXAMPLES = 30
#: The essentials' count: a property still draws its explicit ``@example``
#: rows and this many streams, at a third of the full tier's time. Ten is
#: also the floor of streams `test_properties.py`'s out-of-sample property
#: must compare a prediction on (hard rule 2), which five did not meet.
ESSENTIAL_EXAMPLES = 10


def profile() -> str:
    """The profile ``HYPOTHESIS_PROFILE`` names, ``extended`` when unset."""
    name = os.environ.get(PROFILE_VARIABLE, "extended")
    if name not in PROFILES:
        raise ValueError(f"{PROFILE_VARIABLE}={name!r}: expected one of {', '.join(PROFILES)}")
    return name


def register() -> None:
    """Register both profiles and load the one the environment names."""
    base = settings.default
    settings.register_profile("extended", parent=base, max_examples=EXTENDED_EXAMPLES)
    settings.register_profile("essential", parent=base, max_examples=ESSENTIAL_EXAMPLES)
    settings.load_profile(profile())


def examples(full: int, essential: int = ESSENTIAL_EXAMPLES) -> int:
    """How many examples a property draws: ``full`` in the extended tier,
    and no more than ``essential`` in the essentials."""
    return full if profile() == "extended" else min(full, essential)
