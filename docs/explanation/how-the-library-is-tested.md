<!-- kind: explanation -->
# How the library is tested

This page explains, for anyone deciding whether to trust polars-online's
numbers, how each of its guarantees is held by a test, and where the tests
run.

**Each guarantee in the README's [What you can rely
on](../../README.md#what-you-can-rely-on) is held by a test, against a
reference or an invariant.** A reference is a computation the model cannot
share a bug with, such as scikit-learn's `Ridge` for `ewridge`. An
invariant is a property every run must keep. The invariants are checked
over streams that Hypothesis, a library for property-based testing,
generates with mixed nulls, clocks that repeat or jump, values at ±1e8,
zero weights and tiny groups.

## What holds each guarantee

**Each of the main guarantees has a test that would fail if it stopped
holding:**

| guarantee | held by |
|---|---|
| out-of-sample predictions | changing a row's own target never changes that row's prediction, for every regression model (`tests/test_properties.py`); each model's prediction while learning equals the one it makes from the row's features alone, without its target (`crates/online-core/tests/model_contract.rs`) |
| bounded memory | a 10-million-row run whose memory and state stay bounded (`tests/test_soak.py`, opt-in with `pytest -m soak`, weekly) |
| any chunking | a stream fed in chunks of 1 to 13 rows gives one chunk's output, on generated streams for every regression model (`tests/test_properties.py`) and on a fixed stream for every model, which is also saved and loaded halfway (`tests/test_semantics_all_models.py`) |
| any thread count | a bank run on 1 thread against 8 threads, its output compared bit for bit (`tests/test_portability.py`) |
| zero weights and one `weight_sum` | every model held to the same `weight_sum` recursion (`crates/online-core/tests/model_contract.rs`) |
| each model's numbers | a reference written from its update equations, and another library wherever one computes the same quantity, such as statsmodels, scipy and [river](https://riverml.xyz) |

## Invariants every model keeps

**Every model is held to the same invariants, because a property that
holds for one model and not another would make a bank's output depend on
which models it runs.** The bank runs each model over one chunk, seven
chunks, a hundred chunks, and one row at a time, with a save and load in the middle. It runs
each at 1 thread against 8, and the numbers must not move. A group's numbers must
not depend on what else is in the bank. The command line must give the
bank's output, and the Arrow output the Polars one, field for field.
`predict` must give what `fit_predict` would for the next row. Each model
must keep the same `weight_sum` recursion. [What the README's guarantees
are held to](../TESTING.md#what-the-readmes-guarantees-are-held-to)
tabulates every invariant and the test that checks it.

**The strongest invariant is that changing a row's own target never
changes that row's own prediction.** Hypothesis generates adversarial
streams to test it, and in the Rust tests `proptest` drives generated
streams through every model against the same contract. An information
coefficient (IC), the correlation of a prediction with its target, of
about 0 on pure-noise targets checks the same thing from the other side.

## Fixed numbers

**Fixed output catches a change that no invariant would see, such as a
divergence in Polars' vectorized paths on another CPU.** Each model has
one stream whose output is stored in the Rust tests. The whole pipeline,
from reading the columns through the models and the diagnostics to
assembling each spec's output, is pinned to fixed output too, and compared
on every operating system.

## What the suite does to a bank on purpose

**The suite attacks a bank the ways a user's data or code might, because
the failures a user meets are rarely the ones a well-formed stream
shows.** It scales every weight by 1e±6, sets `half_life` from `1e-3` to
`inf`, and flips any byte of a state file. It calls `fit_predict` from two
threads at once, copies a bank with `pickle` and `copy.deepcopy`, and runs
10 million rows in one go. Each attack has what must hold: a clean error
where a call is misused, and the same numbers where it is not. The contributors' guide to the tests,
[TESTING.md](../TESTING.md#what-the-readmes-guarantees-are-held-to),
lists every attack and what must survive it.

## Contracts held to files

**What the repository promises in a file is held by a test that reads
it, so a promise cannot go stale unnoticed.** The public API, every name,
default, signature and output field name, is compared with a checked-in
copy, so a change shows as a diff in review. Every python block in the
README, the tutorials, the how-tos and the explanations runs, as does
every example in the API reference and everything under `examples/`. The
state files every release from 0.10.0 on wrote are loaded: a 0.x
release's refused, and a 1.x release's continued as the CHANGELOG says.
[VALIDATION.md](../VALIDATION.md), where the defaults were chosen, is
regenerated and compared, so the numbers behind the defaults cannot
silently stop being true. And no data file may be tracked: the tests generate or download their own
data. TESTING.md's [What the README's guarantees are held
to](../TESTING.md#what-the-readmes-guarantees-are-held-to) names each of
these tests.

## Where the suite runs

**The suite runs on Linux, macOS and Windows at every push to `main` and
every pull request.** It has 2,653 Rust tests and 7,211 pytest cases,
counted on 2026-10-24. It runs on Linux at Python 3.12, 3.13 and 3.14, and
on macOS and two Windows builds (Server 2025 and Server 2022) at 3.12 and
3.14. Windows runs twice because its math library is part of the operating
system, so each build's last bits can differ.
[docs/TESTING.md](../TESTING.md) is the ledger of what each part proves.
It lists every reference and its agreement, every invariant, and [where
the suite runs](../TESTING.md#where-the-suite-runs).
