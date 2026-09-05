# The state workflow on the plan surface — research, 2026-09-03

**Status: decided and implemented, 2026-09-03 (PLAN task 20; the decisions
are recorded in §7).** `lf.online.fit_predict(.., save_state=)` is the
syntax proposed in §4, with rules R1–R7; §5's checks are
`tests/test_frame.py`'s E35 tests. The facts about polars it rests on were
measured, not recalled (§2, `scripts/io_source_semantics.py`), on polars
1.34.0 (the floor), 1.38.1 and 1.44.1 (the pin). The memory side — a plan
that updates a `ModelBank` object, or takes one as `load_state` — was
decided against: the file is the state's one form on the plan surface,
and a bank object stays the loop's (§1, §3 B/E).

## 0. The ask, and the answer in one paragraph

Four steps: **(1)** fit a model online in bounded memory; **(2)** export the
model state, optionally to disk; **(3)** load the state and predict online in
bounded memory *without* updating it; **(4)** load the state and update it
with new data. The requirement is that these read naturally in the
polars-native syntax, `lf.online.*`.

Every step already exists on the `ModelBank`, `po.run` and CLI surfaces, and
steps (1), (3) and the load half of (4) exist on the plan surface. The one
thing the plan surface cannot do is **let state out**: `lf.online.fit_predict`
was made *pure* on purpose (ENHANCEMENTS E33, PLAN §11a) — a fresh bank per
execution, nothing saved. The research question was therefore narrow: *is
there a way for a plan to carry state out that survives the way polars
actually executes a Python source?* The measurements say yes, with exactly
one way: **`save_state=`, the runner's keyword, on the plan**, written
atomically when the source has fed the bank its last row. It is safe
precisely *because* the plan is pure: polars runs a plan's source once per
execution and twice — concurrently — when a query uses the plan twice, and
a pure plan's end state is the same every time, so the write is idempotent.
Two things must change with it for the semantics to be exact: the source
must feed the bank only the rows a `head(n)` asked for (today it feeds the
whole chunk and trims the output), and `load_state` must be read when the
plan is built, not when it runs. One thing cannot be had: the source does not
learn whether the *query* succeeded, so a node after it failing still leaves
the state written (polars drains the source first, in every engine); where
"state only if the output landed" is required, `po.run` is that call
and stays so.

## 1. What existed before the decision, per step and surface

| step | `ModelBank` | `po.run` / TOML / CLI | plan `lf.online.*` (and `df.online.*`, `po.fit_predict`) |
|---|---|---|---|
| (1) fit online, bounded | `bank.fit_predict(chunk)` in a `collect_batches` loop; `fit_predict_batches(iter)` | `po.run(input=path\|lf\|df\|iter, output=path, specs=)` — polars reads in chunks, the bank fits, a writer thread writes; O(state + chunk) | `lf.online.fit_predict(specs).sink_parquet(..)` / `.collect_batches()` — O(chunk) |
| (2) export state | `bank.save(path)` (atomic), `bank.save_bytes()`, `pickle`; inspect with `groups()`, `coef()`, `last_row()`, `gram()` | `save_state=` / `[save_state]` / `--save-state`, written after the output is committed | **none** — the bank was dropped when the source ended (E33: "no `save_state`"); now `save_state=` (§4) |
| (3) load, predict, no update | `ModelBank.load(path).predict(df)`; `load_bytes` | `po.run(load_state=, predict=True)`; `--resume p --predict` | `lf.online.predict(bank_or_path)` — pure, the bank does not move |
| (4) load, update | `ModelBank.load(p).fit_predict(df)` then `save` | `po.run(load_state=, save_state=)`; `--resume p --save-state p` | `lf.online.fit_predict(load_state=p)` learns on from `p`, **could not save**; now `load_state=p, save_state=p` |

So the gap was one cell, twice: getting state *out* of a streamed plan. The
Rust side (`crates/online-polars/src/runner.rs`, the CLI) needs nothing: it
has both keywords and saves after the writer commits.

The state file itself is one msgpack blob for the whole bank
(`crates/online-polars/src/bank.rs` `BankFile`: magic, `format_version`,
`schema_version`, package version, the specs, per spec a sorted list of
(group key, stream state — the models, the diagnostic accumulators and,
since 0.2.0, the output row of the last learned row that `last_row()`
reads), `rows_fed`), written through `atomic.rs` — a
temporary sibling, fsync, rename over the destination (named `.{file}.tmp{pid}`
at the time of the research; `.{file}.tmp{pid}-{seq}` since decision 4, §7).
Serialization is deterministic (groups sorted), so two saves of the same
state are byte-identical; §5 relies on that to compare states.

## 2. How polars executes a Python IO source — measured

`lf.online.fit_predict(specs)` is a `register_io_source` source. Nothing in
polars' documentation says how often such a source runs per query, whether
two runs of it overlap, what it sees when a later node fails, or what
happens to a run the caller abandons — and each of those decides whether a
side effect inside the source is sound. `scripts/io_source_semantics.py`
measures them with a toy source that logs its own life; the same numbers on
1.34.0, 1.38.1 and 1.44.1 unless noted.

| | fact | measured |
|---|---|---|
| **F1** | **One run per execution, and no sharing.** A plan used twice in one query — `lf.join(lf)`, `pl.concat([lf, lf])`, `pl.collect_all([lf.sink_parquet(a, lazy=True), lf.select(..).sink_parquet(b, lazy=True)])` — runs its source twice. Common-subplan elimination does not apply to a Python source, on or off. | 2 runs in every case |
| **F2** | **Those two runs are concurrent**, on two threads, in both engines. | second start before the first end, every case |
| **F3** | **Building or inspecting a plan runs nothing**: `collect_schema()`, `explain()`, `explain(engine="streaming")`. | 0 runs |
| **F4** | **`head(n)` reaches the source as `n_rows`** (when it is pushed — not through a `sort`), and the source ends its own run by returning; a filter reaches it as `predicate`; the streaming engine hands `batch_size=100_000`, the in-memory engine `None`. | as `_frame.py` already assumes |
| **F5** | **A node after the source failing does not stop the source.** With a source that takes 0.5 s and a strict cast that fails on its second chunk, every engine (`collect()`, `collect(engine="streaming")`, `sink_parquet()`) drains the source to its natural end and raises *afterwards*. A failed `sink_parquet` leaves its file behind (0 bytes on 1.34.0, 323 bytes on 1.38.1/1.44.1). | raised after 0.58 s, source ended, in all three |
| **F6** | An exception inside the source surfaces as `ComputeError: caught exception during execution of a Python source, exception: ...`. | as `_frame.py` documents |
| **F7** | **An abandoned run is not ended.** `next()` on `collect_batches` then dropping the iterator, or `for .. break`, leaves the generator suspended after it ran 4–5 chunks ahead; on 1.38.1/1.44.1 it is closed (`GeneratorExit`, `finally`) only when the *plan object* is dropped, on 1.34.0 when the iterator is. | no `end`, `finally` later or much later |
| **F8** | `sink_batches` exists from 1.34.0 (a callback per batch). A callback returning `True` stops the run on 1.34.0/1.38.1 and drains it on 1.44.1. Not needed by anything below; recorded because it differs. | |

Two consequences drive everything in §3–§4:

- **Any state a plan mutates is fed the stream twice, at once, whenever the
  plan is used twice in a query** (F1+F2). That is E33's objection to a
  `bank=` the plan would learn into, now with the failure mode measured: not
  a wrong count but a data race.
- **A side effect can only be tied to the source's own end** (F5, F7): the
  source never learns whether the query succeeded, and `finally` runs at a
  version-dependent moment that can be long after the user moved on. So a
  write happens when the source has delivered its last row, or not at all.

## 3. The candidates

| | form | verdict |
|---|---|---|
| **A** | **`lf.online.fit_predict(specs, save_state=path)`** — the runner's keyword on the plan; the state is written when the source ends, atomically. | **Recommended.** Sound under F1/F2 because the plan is pure: both concurrent runs write the same bytes (§5 C3). Needs the two changes in §4 to be exact under `head(n)` and under a file that changes between build and run. One documented gap (F5): a downstream failure still writes. |
| B | `lf.online.fit_predict(bank)` — a caller's bank the plan learns into. | **Rejected**, as in E33, now on measurement: `lf.join(lf)` or `collect_all` feeds it the stream twice concurrently (F1+F2). A "used once" guard cannot distinguish the second run inside one query from a legitimate second `collect()`, and would fail the query from inside the source. |
| C | State as data: `lf.online.fit(specs)` → a plan of one row, `state: Binary`, that a user sinks or `ModelBank.load_bytes` reads; `load_state` accepting a frame. | Pure and polars-native, but **predictions and state come from two plans, so one pass of the input becomes two** (F1: `collect_all` does not share the source), the bank runs twice, and a user who wants both pays double. Nothing single-pass is possible without a side effect. Keep as an idea if a "state is a frame" use case appears (a join of states? none known). |
| C′ | State on the output's last row: a `state: Binary` column, null except on the last row the source delivers. | Pure and single-pass, but the consumer must find that row after a sink (re-scan the output), a filter after the bank can drop it, and every row carries a null cell. Judged heavier than A for no gain in safety: it too is written before the query's outcome is known. |
| D | Keep the plan pure; state comes from `po.run` / `ModelBank` (status quo). | What the README says today. It answers the four steps, but not in the polars-native syntax the user asked for; and `po.run(input=lf, output=path)` has no lazy output — it is the terminal op. Stays as the *transactional* call (§4 R6). |
| E | `save_state=callable`: `fit_predict(specs, save_state=lambda bank: ...)` — in-memory export from a plan, in the shape of `sink_batches`. | Possible later on top of A (same trigger, same F1/F2 caveat: called once per run, twice in a self-join, on polars' threads). Not needed for the four steps: in-process state is `ModelBank`'s job (`fit_predict_batches`, `save_bytes`). Not recommended now. |
| F | `lf.online.run(specs, output=path, save_state=p)` — the runner as a namespace method, for a uniform reading of the four steps. | A one-line alias of `po.run(input=lf, ..)`; adds surface, no capability. Not recommended; `po.run` takes a `LazyFrame` already. |

## 4. The proposal: A, made exact

The four steps in the polars-native syntax, as they would read:

```python
import polars as pl
import polars_online as po

spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"],
                       clock="t", halflife=600.0, max_dclock=300.0)

# (1) + (2): fit online in O(chunk); the state is written when the stream ends
(pl.scan_parquet("2025.parquet")
   .online.fit_predict([spec], save_state="ridge.state")
   .sink_parquet("2025_scored.parquet"))

# (2) as bytes, or to inspect: the file is a ModelBank
bank = po.ModelBank.load("ridge.state")
blob = bank.save_bytes()

# (3): score new rows against the state, learning nothing (exists today)
(pl.scan_parquet("2026-01.parquet")
   .online.predict("ridge.state")
   .sink_parquet("2026-01_scored.parquet"))

# (4): learn on from the state; the new state replaces the old when the stream ends
(pl.scan_parquet("2026-01.parquet")
   .online.fit_predict(load_state="ridge.state", save_state="ridge.state")
   .sink_parquet("2026-01_scored.parquet"))
```

`df.online.fit_predict(specs, save_state=)` and `po.fit_predict(frame, specs,
save_state=)` are the eager and typed twins (`bank.fit_predict(df)` then
`bank.save`). `load_state` also accepts a `ModelBank`, copied at build time
(R3), so step (4) has an in-process form: `lf.online.fit_predict(load_state=bank,
save_state=...)`. The vocabulary is then the same on every surface:
`load_state` / `save_state` on the plan, in `po.run`, in the TOML, on the CLI.

The rules — each one checked in §5:

- **R1 — the write is tied to the source's end.** `save_state` is written
  when the source has fed the bank its last row: the input's natural end, or
  the `n_rows` a pushed `head(n)` asked for. Never in `finally` (F7), never
  on an error the bank raised (the run ends without reaching the write: the
  file, if any, is untouched), never on an abandoned run. Atomic, through
  `ModelBank.save`, so a reader sees the old file or the new one.
- **R2 — idempotent under re-execution.** Every run of a pure plan ends in
  the same state, so the two concurrent runs of a self-join or a
  `collect_all` write the same bytes twice (C3); `collect()` twice writes
  twice. Two in-process writers need distinct temporaries — `atomic.rs`
  named its temporary by pid only, so either the Python side serialises the
  writes with a lock (the prototype) or the temporary name gains a thread id
  / counter. One of the two is required, not optional (F2). *Taken: the
  counter, in `atomic.rs` (§7, decision 4).*
- **R3 — `load_state` is read when the plan is built.** The plan carries
  the state it was built from (the bytes; each run deserialises them), the
  way `df.lazy()` carries the frame — not re-read at run time, the way
  `scan_parquet` re-reads a file. Without this, `load_state=p, save_state=p`
  used twice in one query races the second run's load against the first
  run's write (F2), and a plan collected twice would not be the same frame
  if the file changed in between (C8b). The probe that computes the output
  schema already opens the file at build time, so this costs nothing. Cost:
  a plan built against `p` does not see a later `p`; build it again.
  `predict(path)` follows the same rule for consistency (its docstring
  said "loaded each time the plan runs" before this); `predict(bank_object)`
  stays by reference, as documented, since `predict` never moves the bank.
- **R4 — the bank is fed exactly the rows the query pulled.** Today a
  pushed `head(n)` feeds the bank the whole first chunk and trims the
  *output* to `n`; with state observable that would make "the state after
  `head(5)`" mean "after 100,000 rows". The source truncates the *input*
  chunk to the rows still wanted before the bank sees it. The delivered rows
  are bit-identical (out-of-sample by construction: row *i* depends on rows
  < *i*), except that `coef` — a reporting cadence, emitted on each chunk's
  last row — now appears on the last delivered row (C1). Less work for a
  `head(n)`, and the README's existing sentence "`head(n)` learns from the
  first `n` rows and no more" becomes true of the bank, not only of the
  output. `sort().head(n)` is not pushed (F4): the bank sees every row and
  the state is the whole stream's (C4b), which is right, since the query did
  read every row.
- **R5 — `predict` refuses `save_state`**, as `po.run(predict=True)` does:
  it learns nothing, so there is nothing to save.
- **R6 — the one gap, stated.** Because polars drains the source before
  surfacing a later node's error (F5), a query that fails *after* the bank —
  a full disk under `sink_parquet`, a bad cast — leaves `save_state` written
  with the complete, valid state of the whole stream while the query's own
  output is missing. With `load_state=p, save_state=p` a rerun then learns
  the data twice. `po.run` saves only after the writer has committed the
  output and is the call for "state only if the output landed"; the
  plan's docstring and the README say so, and recommend a dated
  `save_state` per batch of data (`ridge-2026-01.state`) for the in-place
  pattern, which also keeps an audit trail. This is polars' own precedent:
  `sink_parquet` too leaves its file when the query fails (F5).
- **R7 — building a plan writes nothing** (F3): the schema probe runs the
  bank on zero rows and drops it.

## 5. The prototype's checks (real bank, 3 groups × 1,000 rows, chunks of 250)

A stand-alone `register_io_source` source with R1–R4 built in, against
`po.ModelBank` and `po.run` (scratchpad `proto_save_state.py`; these become
`tests/test_frame.py` tests when implemented).

| | claim | result |
|---|---|---|
| C1 | `head(613)` with the input truncated (R4) delivers the same rows as today's output-side trim, except `coef` on the last delivered row | identical bar `coef`; `coef` present on row 613 with R4, absent today |
| C2 | the state a plan writes == `po.run(save_state=)`'s == a `ModelBank` fed the same chunks | byte-identical, all three |
| C3 | self-join, `pl.concat([plan, plan])`, `collect_all([sink, sink])`: two writes, two threads, file == reference | 2 writes on 2 threads in each case, file byte-identical to C2 |
| C4 | `head(613)` writes the state of a bank fed `df.head(613)` | byte-identical (`rows_seen=613`, 1 group) |
| C4b | `sort().head(5)` writes the whole stream's state | byte-identical to C2 (`rows_seen=3000`) |
| C5 | `for .. break` over `collect_batches` writes nothing | no file |
| C6 | a bank error mid-stream (a null clock at row 700) writes nothing | `ComputeError`, no file |
| C7 | a failing node after the bank still writes (R6) | file present, byte-identical to C2 |
| C8 | `save_state=p` over rows 0–599, then `load_state=p, save_state=p` over 600– : the resumed rows equal one continuous stream's (bar `coef`'s cadence at the chunk boundary, as the README states for any resume) and the final state is byte-identical to the continuous run's | both hold |
| C8b | the plan built against `p` re-run after `p` was overwritten gives the same frame (R3) | same frame |

## 6. Usability, performance, stability — checked, not assumed

- **Usability.** One keyword, already known from the runner, in the place a
  user looks for it; the four steps read in one syntax (§4). The eager and
  typed twins get it for free. Nothing existing changes meaning except
  `predict(path)`'s read time (R3) and `coef` on a `head(n)`'s last row (R4),
  both visible only in cases that were unobservable before.
- **Performance.** `save_state` costs one serialization at the end of a run
  (milliseconds; the state is O(specs × groups × p²)). R3 holds the state
  bytes in the plan for its lifetime — the same order as one bank. R4 does
  strictly less work for a `head(n)`. Nothing on the per-row path moves;
  the 12M-row numbers in `docs/PERFORMANCE.md` §11 are unaffected.
- **Stability.** The design depends on none of the version-specific facts:
  it is correct whether the source runs once or twice (R2), whether polars
  drains the source or stops it on error (R6 narrows if polars ever stops
  it), and whenever `finally` runs (R1 never uses it). The write is atomic
  and idempotent; the only new failure mode is R6, which is documented and
  has a supported alternative. The floor stays 1.34.0.
- **Rules kept.** Hard rules 2 and 3 untouched (C1, C8); `n_eff` and zero
  weight untouched; no new Rust surface unless the temporary's name changes
  (R2 — a `crates/online-polars/src/atomic.rs` change, no linkage, no
  `SCHEMA_VERSION` bump: the file format does not change).

## 7. Decisions (taken 2026-09-03)

1. **Add `save_state=` to the plan (A)** — yes, with R1–R7. The user's
   framing: the file saves are good and easier for a user to understand;
   the memory side is not worth its problems.
2. **R3, `load_state` read at build time** — yes, including for
   `predict(path)`; the alternative (re-read per run, like `scan_parquet`)
   keeps a race in `load_state=p, save_state=p` when a plan is used twice in
   one query. Taken with decision 1, as the rule set proposed; it is the one
   change in observable behaviour for existing code — a plan built before
   the file changed keeps the frame it had (`tests/test_frame.py`,
   `test_load_and_save_the_same_path_resumes_in_place`).
3. **`load_state` accepting a `ModelBank`** — no. The memory side is out of
   scope; a bank object is the loop's, the file is the plan's. Nothing on
   the plan surface mutates a `ModelBank`.
4. **R2's mechanism** — the root, not a lock: the concern was parallel
   writers colliding on one file, and the collision was in
   `crates/online-polars/src/atomic.rs`, whose temporary was named by pid
   alone, so two threads saving one path in one process created and wrote
   *the same temporary* and the rename published whichever mixture resulted.
   The temporary is now `.{name}.tmp{pid}-{seq}` with a process-wide
   counter, so no two writers — threads or processes — share one; the
   destination is the old file or one writer's whole file. This also closes
   the same hole for `ModelBank.save` called from two threads, which existed
   before the plan could write anything, and for `po.run`'s output file
   (`AtomicFile::create` in `runner.rs`). Held by
   `two_writers_of_one_destination_do_not_share_a_temporary` (50 rounds ×
   2 threads × 200 kB, every read is one writer's bytes; it fails under the
   pid-only name). No Python-side lock: the plan's writes are idempotent
   (R2) and now collision-free.
5. **Candidates C / E / F** — none; C is the one to revisit if state as a
   frame ever has a consumer.

## 8. Where it landed

`python/polars_online/_frame.py` (`_source`: R1, R4 and the write;
`_bank`/`_read_state`: R3 for `load_state` and `predict(path)`;
`_save_path`: the directory checked at build; `fit_predict` on the lazy,
eager and typed forms; `predict` R5 by having no such keyword),
`crates/online-polars/src/atomic.rs` (decision 4), `tests/test_frame.py`
(C1–C8b as the E35 tests, `predict(save_state=)` refused, the R3 snapshot,
the same-path double use), `README.md` ("As a plan": the keyword, purity as
what makes the write safe, the R6 note and the dated-file recommendation),
`docs/PLAN.md` §11a, `docs/ENHANCEMENTS.md` (E35), `CHANGELOG.md`,
`tests/api_surface.txt`.

## 9. Reproduce

```sh
uv run python scripts/io_source_semantics.py        # §2, the installed polars
uv venv /tmp/v && uv pip install --python /tmp/v/bin/python 'polars==1.34.0' \
  && /tmp/v/bin/python scripts/io_source_semantics.py   # the floor
```
