<!-- kind: how-to -->
# Update the oracle ledger

This how-to is for a contributor whose change moved this library's
numbers against a third-party oracle, who wants to rerun the oracle grids
the change touches and record their errors in the ledger, which only
tightens.

**The oracle ledger, `oracles.toml`, holds each oracle grid's error as
measured, and every run is held to those fixed numbers.** An oracle grid
is a test that holds a model to a third-party oracle over many seeds,
orders or datasets, and carries the `oracle_grid` mark. No local gate runs
the grids. The gate runs a cheap guard instead, which hashes this
library's outputs on each grid's inputs and fails when a hash or an oracle
library's version moves. [The oracle ledger](../TESTING.md#the-oracle-ledger)
describes the files, how a case is judged and why the numbers are CI's.

## Prerequisites

The steps start from a guard that has failed:

- A branch whose change makes the guard in `tests/test_oracle_ledger.py`
  fail, naming the groups whose hashes or oracle versions moved.
  `scripts/quick.sh` runs that guard for the groups a change can move.
- The project's environment (`uv sync`), and the network for any grid that
  reads a downloaded dataset.

## Steps

The steps take the change from the failed guard to a ledger CI agrees with:

1. **Rerun the groups the guard names,** with `uv run python
   scripts/oracles.py rerun`, or `uv run python scripts/oracles.py rerun
   <group>` for those you name. It runs those groups' oracle grids with
   recording on.
2. **Read its verdict.** It fails if a test fails, or if a case is past the
   ledger or was not measured. Otherwise it applies the run to the ledger
   (new cases, versions, and on the ledger's own platform tightened
   numbers) and writes the groups' hashes to `oracle_hashes.toml`. So a
   hash changes only after a passing run of its oracles.
3. **If a case is past the ledger, find what moved it.** If the change is
   worth having at that error, record the higher number with a `[[raised]]` entry in
   `oracles.toml`. The entry names the `key`, the old value as `from` and
   the new as `to`, the `date`, the `reason`, and a `changelog` text that
   `CHANGELOG.md` contains. A worse error against an oracle is a *Changed*
   or *Fixed* matter, so any section of the CHANGELOG serves.
4. **Commit `oracles.toml` and `oracle_hashes.toml` with the change.**
5. **After the push, read the conclusion of `oracles.yml`.** Its last job,
   `the oracle ledger`, holds every case to the ledger on CI's platform,
   Linux x86-64.
6. **If it asks to tighten, or names a new case or a moved version, apply
   CI's measurements.** Download the run's `oracle-ledger-measured`
   artifact, which holds `measured.json`, and run `python
   scripts/oracles.py apply measured.json`. `apply` never writes a higher number.

To check the result, run `python scripts/oracles.py raises`, which fails on
any number in the file's history that moved away from its ideal without a
`[[raised]]` entry. The next push's `oracles.yml` then passes.

```
uv run python scripts/oracles.py rerun            # every group whose hash or versions moved
uv run python scripts/oracles.py rerun <group>    # those named
```

**A `rerun` on another platform than the ledger's writes none of its
numbers.** The ledger holds CI's numbers, so a run on a Mac is held to
`tolerance_elsewhere` (5%) past the ledger. A gain it sees is reported,
not applied. A case new to the ledger gets its table without numbers,
which `apply` on CI's next `measured.json` fills. The hash file holds one
platform's hashes, and a `rerun` on another platform rewrites that
platform. So it refuses unless every group is re-run, since the hashes it
kept would be the other platform's, and the guard, where they were
written, would report a moved hash instead of failing on it.

## Act on each verdict

**`scripts/oracles.py check` fails on five things, each with its remedy:**

| verdict | when | what to do |
|---|---|---|
| past the ledger | a gap more than the tolerance, 2%, past the ledger's | find what moved it; if the change is worth having at that error, record the higher number with a `[[raised]]` entry |
| tighten | a gap more than 2% under the ledger's | `python scripts/oracles.py apply measured.json`, so the gain is kept |
| version | an oracle library at another version than the ledger's | `apply`, once the numbers held on it |
| not measured | a case of the ledger that a complete run did not record | a skip (offline), or a `record` call gone |
| fewer values | a case measured over fewer values than the ledger's `n`: its worst seed over fewer seeds is no gain, so a grid cut short cannot read as a tightening | restore the seeds; cutting them is a `[[raised]]` entry for the case's `n` |

A case new to the ledger is reported and fails nothing. A number the
history holds and the file no longer has, a case or group dropped or
renamed, fails until a `[[retired]]` entry names it (a case's key or a
group's), with the date and the reason.

## Register a new grid

**A grid registers in two calls.** At module level, `ledger.guard` names
a group, the libraries whose answer it takes as given (distribution names,
as `uv.lock` spells them), its dataset, and a callable that returns this
library's outputs on the group's inputs, with no oracle in it. In the test,
`ledger.record` takes each quantity the grid bounds, one value a seed:

```python
ledger.guard("kmeans_blobs", lambda: [fit_kmeans(blobs(s)[0], 5) for s in SEEDS],
             oracles=["scikit-learn", "scipy"])
...
ledger.record("kmeans_blobs", "raw_ari", aris, ideal=1.0)
```

1. **Call `ledger.guard` at module level** with the group's name, its
   `oracles=` and the callable.
2. **Call `ledger.record` in the test** for each quantity the grid bounds,
   with its `ideal`: 0 for an error, 1 for a score such as an ARI, a share
   or an F1. Pass `exact=True` for a count.
3. **Mark the test `@pytest.mark.oracle_grid(reason="...")`.** `rerun` and
   CI run nothing else.
4. **Run `uv run python scripts/oracles.py rerun <group>`,** which writes the
   group's hash and its new cases.

A group's id is the module's name without `test_`, a dot and the name:
`truth_recovery.kmeans_blobs`. With `POLARS_ONLINE_ORACLES_OUT` naming a
directory, `record` appends what it measured there as JSON lines, one file
a process, so pytest-xdist's workers do not share one. Without it, `record`
checks its arguments and writes nothing. A module with a grid that cannot
register yet is listed in `tests/ledger.py`'s `PENDING` with the reason it
waits, and the rules test in `tests/test_oracle_ledger.py` fails when a
module with a grid is in neither.
