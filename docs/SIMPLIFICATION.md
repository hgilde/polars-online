# Simplification review

Status as of 2026-08-31: **proposed, none implemented.** A read of the codebase
after the P1–P8 performance work, looking for complexity that can go without
costing features, performance, stability, or any stated goal.

The bar each item has to clear:

- **No feature loss.** Nothing in `docs/ENHANCEMENTS.md` stops working.
- **No measured regression.** `docs/PERFORMANCE.md`'s numbers hold, verified
  the same way they were produced.
- **No golden number moves.** `golden.rs` and `test_golden_pipeline.py` are the
  arbiter; if a "simplification" changes a number it is a rewrite, not a
  simplification.
- **Net less code, or net less to hold in your head.** Shuffling complexity
  from one file to another does not count.

Ordered by value, which is not the same as size. S1 removes a whole *class* of
bug; the rest mostly remove typing.

---

## S1 — One field descriptor instead of two orderings *(highest value)*

**Problem.** The output schema is written out twice, in two functions, in the
same order, by hand:

- `output_fields()` (bank.rs, ~102 lines) builds the field *names*, which the
  expression plugin uses to declare its return dtype;
- `assemble()` (bank.rs, ~318 lines) builds the *Series*, re-deriving the same
  names with the same `format!` strings in the same nested loops.

There are 33 `format!` calls across the two. Every optional output —
`emit_sigma`, `emit_resid_z`, `emit_drift`, `emit_metrics`, `emit_autocorr`,
`resid_quantiles`, `emit_selected`, `emit_averaged`, lasso's `lam_selected` —
appears in both, guarded by the same condition twice.

**Why it matters beyond tidiness.** This exact duplication has already produced
a real defect: E23's declared-vs-realized schema divergence, where an output was
added to the assembler but not the declaration. The expression plugin takes its
dtype from the declaration, so the bank kept working while `.over()` broke.
`test_names_match_the_realized_struct_for_every_model` exists purely to catch
recurrences — a guard against a duplication we could delete instead.

**Shape of the fix.** One function returning an ordered descriptor:

```rust
enum Source { Pred, Resid, Sigma, ResidZ, Drift, Metric(usize), Quantile(usize),
              Autocorr, NEff, Coef, LamSelected, Selected, Averaged }
struct Field { name: String, mi: usize, slot: usize, source: Source }
fn output_schema(spec: &Spec) -> Vec<Field>;
```

`output_fields()` becomes `schema.map(|f| f.name)`. `assemble()` becomes a walk
over the same vector, switching on `source` to pick the buffer. The ordering
exists once.

**Risk.** Moderate — it is the widest-reaching change here, and `assemble()` is
delicate. Mitigated by the golden tests plus the 156-field kitchen sink in
`test_hardening.py`, which was written for exactly this kind of stride error.

**Expected:** ~120 fewer lines, one impossible-by-construction bug class, and
the schema guard becomes a redundant belt rather than the only brace.

---

## S2 — Delete `assemble_ew_cov` as a separate function

**Problem.** `assemble_ew_cov` (38 lines) does what `assemble` does — scatter
`ChunkOut` into per-column vectors, build a `StructChunked` — for a model whose
only real difference is that its slots are named statistics rather than
`pred_*`/`resid_*` pairs, and that it has no targets.

**Fix.** Falls out of S1 for free: `ew_cov`'s fields are just a different
`Vec<Field>` from the same descriptor function. The `if matches!(spec.model,
ModelKind::EwCov { .. }) { return assemble_ew_cov(...) }` special case at the
top of `assemble` disappears.

**Risk.** Low, once S1 exists. **Expected:** ~40 lines, one special case.

---

## S3 — A macro for `AnyModel`'s ten-arm dispatch

**Problem.** `AnyModel` has five methods that are each an identical ten-arm
match — `step`, `solve_failures`, `n_outputs`, `coefficients`, `state` — plus
`restore`. That is ~73 `AnyModel::` mentions for six behaviours. Adding an
eleventh model means touching six matches, and the compiler only catches it
because the enum is exhaustive.

**Fix.** A small local macro:

```rust
macro_rules! dispatch {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            AnyModel::EwRidge($m) => $body, AnyModel::Rls($m) => $body, /* ... */
        }
    };
}
```

Then `pub fn step(&mut self, ...) -> Step { dispatch!(self, m => m.step(x, y, d, w)) }`.

**Risk.** Low, and mechanical. The counter-argument is real and worth stating:
macros are harder to read than a match, and this trades ~60 lines of obvious
code for one clever construct. **I would take it only bundled with S1/S2**, not
on its own — a codebase this heavily commented gains little from saving typing
at the cost of grep-ability.

**Expected:** ~60 lines. Judgement call, weakest item here.

---

## S4 — Compute the spec's derived values once

**Problem.** `spec.decays()` allocates a `Vec<(String, Decay)>` and re-runs the
halflife/lam validation every call. It is called **four times** in `bank.rs`
per chunk plus twice in `stream.rs`; `combo_labels(spec)` allocates a `Vec<String>`
and is called three times; `slot_labels` once more. Every one of them is
`.expect("validated")`, i.e. re-doing validation that already passed at
construction.

**Fix.** A `SpecDerived { decays, combos, slot_labels, n_models, nc, m }`
computed once in `Bank::new` beside the existing `clock_cfgs`, and passed to
`assemble`. `Spec::decays()` stays for validation.

**Performance.** Small but real and free: it is per-chunk, not per-row, so at
the default 100k-row chunk it is noise — but a caller feeding 1000-row chunks
pays it 100× more often, and `ONLINE_TIMING` currently attributes it to
`assemble`. Worth doing because it *removes* work, not because it is hot.

**Risk.** Low. **Expected:** ~20 lines, plus one fewer `.expect()` per call site.

---

## S5 — Two `Vec<f64>` scratch buffers in `solve_spd`

**Problem.** Each solve does `Mat::from_fn` twice — once for the RHS, once per
jitter attempt for the matrix — then copies the solution out element-by-element
into a `Vec`. That is 2–3 heap allocations plus three copies per solve, on a
path called once per `solve_every` rows.

**Fix.** Take `&mut Vec<f64>` scratch from the caller (the models already own
per-instance scratch after P1), and write the solution into a caller-provided
buffer.

**Performance caveat, and why this is ranked low.** `core_bench` says
solve-every-row is 469k rows/s against 5.7M for solve-never — so solving is
expensive — but the default `solve_every = halflife/50` already amortizes it to
within 1.5× of never-solving. The allocations are a small share of a solve
that is dominated by the O(k³) factorization. **This is a simplification only
if it also reads better**; as a performance change it is not worth the
argument, and `docs/PERFORMANCE.md` §5 already rejects micro-optimizing here.

**Risk.** Low. **Expected:** marginal. Do it last or not at all.

---

## S6 — `Stream::save` / `restore` field-by-field cloning

**Problem.** `StreamState` mirrors nine of `Stream`'s fields, and `save()`
clones each by hand. Adding a piece of per-stream state means remembering to
add it in three places (struct, `save`, `restore`), and forgetting is silent —
the state just does not persist.

**Fix.** Make the persisted fields a single `#[derive(Serialize, Deserialize)]`
sub-struct that `Stream` holds by value, so `save()` is `self.persisted.clone()`
and the compiler enforces completeness.

**Risk.** **This changes the state file layout**, so it needs a
`SCHEMA_VERSION` bump and a v-previous loader per CLAUDE.md hard rule 5, and
`crates/online-polars/tests/state_v1.rs` gains a second frozen fixture. That is
real work for an internal tidy.

**Verdict: not now.** Worth doing the *next* time `SCHEMA_VERSION` has to move
for another reason, so the migration cost is already being paid. Recorded here
so the opportunity is not forgotten.

---

## Considered and rejected

- **Replacing `faer` with a hand-rolled Cholesky.** Solves are k ≤ ~65, and a
  50-line Cholesky is genuinely simple. But `faer` also supplies the pivoting
  and numerical care behind the jitter-retry ladder that `solve_spd` depends
  on, and `tests/test_edge_cases.py` covers exactly the degenerate matrices
  where hand-rolled code goes wrong. Trading a dependency for numerical risk in
  the one place the library must not silently produce NaN is a bad trade.
- **Flattening `Vec<Vec<f64>>` state (`resid_var`, `resid_w`, `r`) into one
  buffer.** Would touch every model. The per-instance split in P2 already gives
  each task contiguous output; the *state* vectors are small (n_slots doubles)
  and accessed per row, so they are in L1 regardless. Churn without a measurement
  behind it.
- **Merging `online-polars` and `online-py`.** They are separate so the CLI does
  not link pyo3 — a real constraint, not an accident.
- **Collapsing the eleven near-identical `po.spec.*` constructors in
  `_spec.py`.** They are the public API's discoverability: explicit keyword
  arguments per model are what make the signatures readable in an editor and
  in `help()`. A generic `spec(model="ewridge", **kw)` would be shorter and
  worse.

---

## Suggested order

1. **S4** first — smallest, independent, and makes S1's signature cleaner by
   handing `assemble` its derived values rather than re-deriving them.
2. **S1 + S2** together — S2 is free once S1 lands, and doing them apart means
   touching `assemble` twice.
3. **S3** only if S1/S2 leave the dispatch looking out of place beside them.
4. **S5** and **S6** deferred, with S6 waiting for a `SCHEMA_VERSION` bump that
   is happening anyway.

Each step: `./scripts/gate.sh` unpiped, then re-run the `ONLINE_TIMING` matrix
and `scripts/scaling_bench.py` to confirm no regression, and confirm the golden
suites did not move.
