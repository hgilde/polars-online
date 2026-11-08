<!-- kind: how-to -->
# Register an oracle grid

This how-to is for a contributor adding a test that holds a model to a
third-party oracle over many seeds, orders or datasets, who wants its
error held to the oracle ledger.

**A grid registers in two calls, so the ledger holds its error and the
test gate's guard hashes its outputs.** An oracle grid is a test that
carries the `oracle_grid` mark, which CI runs on every push to `main` and
no local test gate runs. The ledger, `oracles.toml`, keeps each case's
error as measured, and [The oracle
ledger](../TESTING.md#the-oracle-ledger) describes how a case is judged.

## Prerequisites

The steps need a test that already compares the model with its oracle:

- A test module whose test computes, for each seed, the quantity the
  oracle bounds: an error against the oracle, or a score such as an ARI.
- The project's environment (`uv sync`), and the network for any grid that
  reads a downloaded dataset.

## Steps

The steps turn that test into a registered grid:

1. **Call `ledger.guard` at module level.** It names the group, the
   libraries whose answer it takes as given (distribution names, as
   `uv.lock` spells them) in `oracles=`, the group's `dataset=`, and a
   callable that returns this library's outputs on the group's inputs, with
   no oracle in it.
2. **Call `ledger.record` in the test** for each quantity the grid bounds,
   one value a seed, with its `ideal`: 0 for an error, 1 for a score such
   as an ARI, a share or an F1. Pass `exact=True` for a count.
3. **Mark the test `@pytest.mark.oracle_grid(reason="...")`.** `rerun` and
   CI run nothing else.
4. **Run `uv run python scripts/oracles.py rerun <group>`,** where
   `<group>` is the group's full id, its module's name without `test_`, a
   dot and the group's name, as `truth_recovery.kmeans_blobs`. It runs the
   grid, writes the group's new cases to `oracles.toml` and its hash to
   `oracle_hashes.toml`. Off the ledger's platform, Linux x86-64, the new
   cases' tables carry no numbers, and a case without numbers holds
   nothing.
5. **Commit the test, `oracles.toml` and `oracle_hashes.toml` together.**
6. **After the push, apply CI's numbers,** as [Update the oracle
   ledger](update-the-oracle-ledger.md#steps), steps 8 to 11, describes:
   read `oracles.yml`'s conclusion, download its `oracle-ledger-measured`
   artifact, run `uv run python scripts/oracles.py apply measured.json`, and
   commit `oracles.toml`. From then on, the ledger holds the grid's error.

This code registers a group and records one quantity:

```python
ledger.guard("kmeans_blobs", lambda: [fit_kmeans(blobs(s)[0], 5) for s in SEEDS],
             oracles=["scikit-learn", "scipy"])
...
ledger.record("kmeans_blobs", "raw_ari", aris, ideal=1.0)
```

To check the result, run `uv run pytest tests/test_oracle_ledger.py`. Its
rules test fails when a module with a grid is neither registered nor listed
in `tests/ledger.py`'s `PENDING`. Its guard fails when the group's hash is
missing, on the development Mac where the hashes were written; elsewhere
it reports.

**`record` writes only when asked.** With `POLARS_ONLINE_ORACLES_OUT` naming a
directory, `record` appends what it measured there as JSON lines, one file
a process, so pytest-xdist's workers do not share one. Without it, `record`
checks its arguments and writes nothing.

**A module with a grid that cannot register yet goes in `tests/ledger.py`'s
`PENDING`,** with the reason it waits. The rules test fails when a module
with a grid is in neither.
