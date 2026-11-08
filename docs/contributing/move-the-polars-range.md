<!-- kind: how-to -->
# Move the Polars range

This how-to is for a contributor who must change the range of py-polars
that polars-online declares, `polars>=1.37.0,<3` in `pyproject.toml`: cap
it when a new Polars breaks the library, move its floor up, or move its
ceiling up to a new major.

**When a new Polars breaks this library, a patch release caps the range at
the last Polars that passed, so that no resolver hands anyone the broken
pair.** A fix then widens the range again. [The Polars
range](../VERSIONS.md#the-polars-range) says which release each kind of
move needs, and [Polars interfaces](../explanation/polars-interfaces.md)
why a break on a new Polars is expected maintenance.

## How a break is found

**Two kinds of run look for a break, in Polars and in NumPy, the one
optional dependency** (`polars-online[numpy]`, `numpy>=1.26`). The canary
reports one, and the legs every release runs before it publishes withhold
the release:

| run | when | what it resolves to | a red run |
|---|---|---|---|
| `release.yml`'s `next-polars`, the newest in-range leg | every dispatched release, before any tag exists | the newest stable inside `<3` | withholds the release until the range is capped or the code fixed |
| `release.yml`'s `floor-polars` | every dispatched release, as above | exactly the floor `pyproject.toml` declares, 1.37.0, with its runtime package | withholds the release until the code is fixed or the floor moved up |
| `release.yml`'s `next-polars`, the next-major leg | every dispatched release, as above | unpinned, prereleases allowed | is advisory, and withholds nothing |
| `release.yml`'s `next-numpy`, the newest NumPy | every dispatched release, as above | the newest NumPy | withholds the release |
| `release.yml`'s `next-numpy`, NumPy's next release candidate | every dispatched release, as above | its next release candidate | is advisory, and withholds nothing |
| `polars-canary.yml`, the canary | weekly, on Mondays, and on demand | the newest py-polars, prereleases included, with the range dropped, for the Rust tests and the Python suite | is the notification, and withholds nothing |
| the canary's `next-numpy` | weekly, on Mondays, and on demand | NumPy's next release candidate | is the notification, and withholds nothing |
| the canary's `floor-polars` | the 1st of each month, and on demand | the floor, 1.37.0, as every release runs it | is the notification, and withholds nothing |

**The canary ([`polars-canary.yml`](../../.github/workflows/polars-canary.yml))
drops the range from `pyproject.toml` and installs the newest py-polars,
prereleases included.** It builds the wheel as CI does, then runs the
suite, all but the opt-in soak tests and the checks of this repository's
own pins. Only Polars moves in that run, so a red canary means Polars
broke this library. Only NumPy moves in the NumPy runs, the release's and
the canary's, so a red one names NumPy.

**The newest in-range leg blocks the publish, because `<3` admits every
1.x and 2.x.** A resolver can hand someone a Polars newer than the one the
wheel was built against, the day after that Polars ships. A pass on the
pinned version alone would not show that the range holds. The next-major
leg is early warning for [a new major](#move-the-ceiling-up-to-a-new-major).

**The floor leg blocks the publish too, because a run on the newest Polars
says nothing about the oldest one the range admits.** A resolver keeps a
user's Polars while it is inside the range, so a change that needs a newer
Polars than the floor would break that user's install. The newest-Polars
runs prove the ceiling, and only the floor runs prove the floor.

## Prerequisites

Each of the moves below needs these:

- The run that found the break, or the reason to move the floor, read to
  its failing test.
- A checkout of `main`, and the steps of [Cut a release](cut-a-release.md)
  for shipping the change.

## Cap the range below a broken Polars

A cap ships first, and the fix follows it.

1. **Read the red run's Polars version** in its install step. In the
   canary and the release's Polars legs only Polars moves, so that version
   is the one that broke the library.
2. **Cap the range in `pyproject.toml`** below that version, and let the
   lock resolve.
3. **Ship the cap as a patch release,** with [Cut a
   release](cut-a-release.md).
4. **Fix the code, and widen the range again** in the release after the
   cap.

## Move the floor up

**Move the floor up when the library needs a Polars newer than the
floor.**
`scripts/polars_floor.py` reads the floor from `pyproject.toml` for the
release's floor leg and the canary's, so a new floor moves both runs with
it.

1. **Move the floor up in `pyproject.toml`'s `polars` range,** and let the
   lock resolve.
2. **Set `SUPPORTED_FLOOR` in `tests/test_scaffold.py` to the same
   version.** The test pins the declared floor, so a change to either has
   to change both. It also holds every guide that states the floor to the
   declared one.
3. **Remove each `needs_polars` mark at or below the new floor.**
   `needs_polars` in `tests/polars_version.py` skips a test on a Polars
   older than the version it names, and says why. It refuses a version at
   or below the floor, where it would skip nothing, and one above the pin,
   where it would hide the test from CI's own runs.
4. **Run the floor leg locally, in a scratch environment,** with only
   polars and its runtime package at the new floor. The release's floor
   leg runs the same suite, and refuses to go on unless the installed
   py-polars is the floor.
5. **Ship the change as a minor release.** A resolver that installs it
   upgrades Polars with it, inside the same Polars major, so no user's code
   has to change.

## Move the ceiling up to a new major

**Five steps, in order, move the ceiling to a new major.** The ceiling is
`<3`, moved up from `<2` in 0.5.1, so a py-polars 3.0 is excluded until they
are done again.

1. **Read the last run of the canary and of the release's advisory leg.**
   Both unpin, and both allow prereleases, so a release candidate of the
   next major is exercised the week it appears. Moving the ceiling up
   is also what moves that major under the *blocking* leg, since that leg
   tests the range as declared.
2. **Check the Rust side separately.** py-polars' major and the `polars`
   crate's version are independent. On 2026-10-08, with py-polars at
   2.0.0, the newest crate was 0.55.2, the one `Cargo.toml` pins. A Python
   major on its own needs no Rust change. The wheel carries its own
   statically linked copy, and the two never meet: data crosses on the
   Arrow C Data Interface.
3. **Widen the range in `pyproject.toml`**, today `polars>=1.37.0,<3`, and
   let the lock resolve. That is the only required source change if steps
   1 and 2 are clean.
4. **Regenerate `docs/VALIDATION.md`** with
   `uv run python scripts/validate.py > docs/VALIDATION.md`. Its header
   records the Polars version it was generated with, which is why its test
   is marked `pins`.
5. **Ship it as a minor**, not a patch. Widening the Polars range is a
   minor release by this package's own rule ([The Polars
   range](../VERSIONS.md#the-polars-range)).

## Check the result

The move is done when the release that ships it passes its `next-polars`
and `floor-polars` legs, and the canary's next run is green on the range
as declared.
