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
why a break on a new Polars is expected maintenance. [The runs that watch
the dependencies](../RELEASE-READINESS.md#the-runs-that-watch-the-dependencies)
says which runs find a break, and what each withholds.

## Prerequisites

Each of the three moves needs these:

- The run that found the break, read to its failing test; or the reason
  to move the floor; or, for the ceiling, the canary's run on the new
  major.
- A checkout of `main`, and the steps of [Cut a release](cut-a-release.md)
  for shipping the change.

## Cap the range below a broken Polars

**A cap ships first, and the fix follows it.** If the red run's Polars
is outside the range already, such as a 3.0 prerelease under `<3`, no cap
is needed: the range keeps it from every user, and the work is [moving
the ceiling up](#move-the-ceiling-up-to-a-new-major) when the code is
fixed.

1. **Read the red run's Polars version** in its install step. In the
   canary and the release's Polars legs only Polars moves, so that version
   is the one that broke the library.
2. **If the cap is not a major, widen the two readers of the range
   first.** `scripts/polars_floor.py` reads only the form
   `polars>=X.Y.Z,<N`, a floor and a major ceiling, and refuses any other.
   `test_the_declared_range_brackets_what_we_build_against`
   (`tests/test_scaffold.py`) reads the ceiling as `<N` too. Widen its
   `RANGE` and that test's parse to take a ceiling such as `<2.3`, and add
   the new form as a case of `test_the_floor_script_reads_the_range_it_is_given`
   (`tests/test_release_workflow.py`).
3. **Cap the range in `pyproject.toml`** below that version, and run
   `uv lock`.
4. **Check that the floor still reads,** with
   `python3 scripts/polars_floor.py`. It prints `version=` and the floor.
5. **Ship the cap as a patch release,** with [Cut a
   release](cut-a-release.md).
6. **Fix the code, and widen the range again** in the release after the
   cap.

## Move the floor up

**Move the floor up when the library needs a Polars newer than the
floor.**
`scripts/polars_floor.py` reads the floor from `pyproject.toml` for the
release's floor leg and the canary's, so a new floor moves both runs with
it.

1. **Move the floor up in `pyproject.toml`'s `polars` range,** and run
   `uv lock`.
2. **Set `SUPPORTED_FLOOR` in `tests/test_scaffold.py` to the same
   version.** The test pins the declared floor, so a change to either has
   to change both.
3. **Change the floor in every guide that states it,** `docs/VERSIONS.md`,
   `docs/RUNNER.md` and `docs/RELEASE-READINESS.md` among them.
   `test_every_guide_that_states_the_floor_states_the_declared_one`, in
   the same file, names any you miss.
4. **Remove each `needs_polars` mark at or below the new floor.**
   `needs_polars` in `tests/polars_version.py` skips a test on a Polars
   older than the version it names, and says why. It refuses a version at
   or below the floor, where it would skip nothing, and one above the pin,
   where it would hide the test from CI's own runs.
5. **Run the suite on the new floor locally,** in a scratch copy of the
   checkout, as [Cut a release](cut-a-release.md) step 11 runs it: only
   polars and its runtime package move to the floor. The release's floor
   leg runs the same suite, and refuses to go on unless the installed
   py-polars is the floor.
6. **Ship the change as a minor release.** A resolver that installs it
   upgrades Polars with it, inside the same Polars major, so no user's code
   has to change.

## Move the ceiling up to a new major

**Five steps, in order, move the ceiling to a new major.** The ceiling is
`<3`, moved up from `<2` in 0.5.1, so a py-polars 3.0 is excluded until
these five steps are done for it.

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
   run `uv lock`. That is the only required source change if steps 1 and
   2 are clean.
4. **Decide whether the development environment moves to the new
   major.** Step 3's `uv lock` keeps the locked py-polars while it is
   inside the range. To move it, run
   `uv lock --upgrade-package polars --upgrade-package polars-runtime-32`,
   then `uv sync` and
   `uv run maturin develop --release -m crates/online-py/Cargo.toml`.
   Then regenerate `docs/VALIDATION.md` with
   `uv run python scripts/validate.py > docs/VALIDATION.md`, and set
   `BUILT_AGAINST` in `tests/test_scaffold.py` to the version the lock
   holds. The page's header records the Polars version it was generated
   with, which is why its test is marked `pins`. If the environment stays,
   `BUILT_AGAINST` and `docs/VALIDATION.md` stay as they are.
5. **Ship it as a minor**, not a patch. Widening the Polars range is a
   minor release by this package's own rule ([The Polars
   range](../VERSIONS.md#the-polars-range)).

## Check the result

The move is done when the release that ships it passes its `next-polars`
and `floor-polars` legs, and the canary's next weekly run and its next
floor run are green.
