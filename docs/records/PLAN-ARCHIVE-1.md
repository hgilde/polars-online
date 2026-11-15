<!-- kind: record -->
# The plan archive, part 1 of 8

Moved out of `docs/PLAN.md` on 2026-11-14 (task 398), text unchanged: the plan had
grown to 1.7 MB, more than a reader can hold, and most of it was finished work.
This part holds the full entry of each finished task and the text of each closed
section, in the order they stood in the plan. `docs/PLAN.md` keeps one line per
finished task, linking the entry here (`#task-N`), and every section heading with
a line saying where its text went (`#sNN`).

Later text is in [part 2](PLAN-ARCHIVE-2.md).

<a id="s11"></a>
## 11. Task list

<a id="task-1"></a>
- [x] 1. Scaffold: workspace, four crates, `pyproject.toml`, `uv`, `maturin develop` builds an
      empty plugin, CI runs lint on all three OSes.

<a id="task-2"></a>
- [x] 2. `tests/data.py`: synthetic generator + public download + caching + offline skip.
      `tests/reference.py`: numpy EW-ridge and RLS oracles.

<a id="task-3"></a>
- [x] 3. `online-core`: `OnlineModel` trait, `Step`, clock/decay helper (`halflife`/`lam`,
      `max_dclock`, `on_clock_reset`, session gap), `EwCov` primitive. Unit tests.

<a id="task-4"></a>
- [x] 4. EW-ridge (§4.1) single target, no grids; Cholesky via `faer`; solve schedule.

<a id="task-5"></a>
- [x] 5. Multi-target + `feature_sets` + `ridge` grid + standardization at solve time.

<a id="task-6"></a>
- [x] 6. `online-polars` model bank: column extraction, per-group state, rayon fan-out, chunk
      monotonicity check, save/load (msgpack, versioned).

<a id="task-7"></a>
- [x] 7. `online-py`: `ModelBank` class + `fit_predict`; pytest oracle, chunk-invariance, null,
      warmup, clock tests pass.

<a id="task-8"></a>
- [x] 8. Expression plugin (`online.ewridge`) + expression≡bank test.

<a id="task-9"></a>
- [x] 9. RLS (§4.2) + agreement test with EW-ridge at `solve_every`=1 row.

<a id="task-10"></a>
- [x] 10. Lasso path + online λ selection (§4.3).

<a id="task-11"></a>
- [x] 11. Kalman (§4.4) with halflife-derived q; per-factor halflife; `inf` pinning.

<a id="task-12"></a>
- [x] 12. Evaluation harness (§8); run the solve-schedule experiment and every **[validate]**
      item on public data; record results in `docs/VALIDATION.md` and fix defaults.

<a id="task-13"></a>
- [x] 13. Robust models (§4.5).

<a id="task-14"></a>
- [x] 14. Logistic / FTRL (§4.6).

<a id="task-15"></a>
- [x] 15. `online-cli`: TOML specs, streaming in/out (parquet, ipc, csv, ndjson — ENHANCEMENTS
      E32), progress, resume from state.

<a id="task-16"></a>
- [x] 16. CI: wheels + CLI binaries for macOS/Windows, cross-platform state test, Polars-latest
      canary job. Benchmark script + numbers in README.

<a id="task-17"></a>
- [x] 17. README with the three usage modes and the math per model.

<a id="task-18"></a>
- [x] 18. **Weekly native leak check in CI — after the repo is public.** Add
      `scripts/leakcheck.sh` to a scheduled Linux job (valgrind, with CPython's
      suppression file) and to the weekly macOS run (`leaks`). Deliberately
      deferred, not forgotten: it is worthless on a budget, because valgrind is
      roughly 50x slower than the suite and macOS bills at 10x, and the whole
      2,000-minute month went in a single day while the repo was private
      (`docs/RELEASE-READINESS.md`). Once Actions is unmetered the cost is
      irrelevant and the value is real — `tests/test_ffi_memory.py` can only
      see a leak large enough to move RSS, whereas `leaks`/valgrind find
      unreachable blocks of any size. **Trigger: the repo going public.** Both
      currently report clean, so the job starts from a known-good baseline;
      wire it as reported-not-gating first (like the benchmark job), since
      valgrind on CPython is noisy until the suppressions are tuned.
      **The trigger fired on 2026-09-02**: the repository is public and
      Actions is unmetered, so this is now doable work rather than a deferral.
      **Done 2026-09-03**, `.github/workflows/leakcheck.yml`: Mondays and on
      demand, ubuntu + macOS, nothing gates on it — a scheduled run that goes
      red is the report (GitHub mails the owner). Wiring it found that the
      "0 leaks" baseline was blindness, not cleanliness: pymalloc hands Python
      objects out of mmap'd arenas that `leaks` does not walk, and the script
      reported 0 leaks against a deliberate 128 MB refcount leak. With
      `PYTHONMALLOC=malloc` it sees Python objects — and the interpreter then
      leaves ~11k blocks (~700 KB) unreachable at exit whatever the workload,
      so the script became differential: 1 iteration against 1000, growth over
      500 blocks or 64 KiB is a leak (two runs of the same workload differ by
      <150). It still cannot see anything allocated through polars' allocator
      (mimalloc on macOS, jemalloc on Linux), which is every Rust-side
      allocation; 160 MB of deliberately leaked `Series` buffers did not move
      the count. That side stays with `test_ffi_memory.py` and RSS. Because a
      blind check reports clean forever, the job also runs a **control**
      (`LEAKCHECK_CONTROL=1`, one Python object leaked per iteration) that
      must fail. Real workload: growth 63 blocks / 3.7 KB, clean; control:
      +1968 blocks / 142 KB, caught. The valgrind path is written to the same
      contract and first runs on the runner — it cannot run on this machine.

<a id="task-19"></a>
- [x] 19. The expression plugin (task 8) is the in-memory form and says so: every
      `pl.col(..).online.<model>` call warns with `InMemoryExpressionWarning` naming the plan
      and the reason (§6); the README shows the two forms side by side in a closing note.
      (First committed as an off-by-default cargo feature that took it out of the wheel;
      reverted the same day — §11a.)

<a id="task-20"></a>
- [x] 20. **State out of a streamed plan — researched and implemented 2026-09-03.**
      The four-step workflow (fit online in bounded memory; export the state, optionally
      to disk; load it and predict without updating; load it and learn on) existed end to
      end on `ModelBank`, `po.run` and the CLI, and on the plan surface for every step
      but the export: `lf.online.fit_predict` is pure by decision (E33, §11a). The
      research — `docs/STATE-WORKFLOW.md`, with the engine facts measured on polars
      1.34.0/1.38.1/1.44.1 by `scripts/io_source_semantics.py` — found one sound
      form, `lf.online.fit_predict(specs, load_state=, save_state=)`: the runner's
      keywords on the plan, the state written atomically when the source has fed the
      bank its last row, idempotent under the two concurrent runs polars gives a plan
      used twice in one query. Implemented as proposed (§11a): the source feeds the bank
      only the rows a `head(n)` asked for, `load_state` and `predict(path)` are read when
      the plan is built, and the collision two concurrent writers had in `atomic.rs` is
      fixed at the root. The memory side (a plan mutating a `ModelBank`) is declined.

<a id="task-21"></a>
- [x] 21. **Gradient-boosted trees, as online as possible — investigated 2026-09-03.**
      Asked how far XGBoost's method can be pushed toward this library's contract, and
      what that does to parallel fitting and memory. `docs/BOOSTED-TREES.md` (§11g) is the
      answer: the XGBoost paper and source (`54155e3`) read with every claim cited by
      `file:line`, the streaming-tree literature and river/MOA/VW/LightGBM code compared,
      a design that keeps the contract (EW-decayed per-node gradient sums, histograms only
      on splittable leaves from a bounded pool, growth and collapse at checkpoints so the
      model is frozen between them — hence chunk-invariant and additive over threads — and
      a batch warm start on the warm-up buffer the bins need anyway), prototyped in numpy
      (`scripts/ogbt_proto.py`) and measured (`scripts/ogbt_experiments.py`) against
      XGBoost refits on synthetic drift. Nothing in the Rust crates; whether to build it is
      the user's call and §9 of the doc costs it. Research sources stay under the
      gitignored `.cache/research/`.

<a id="task-22"></a>
- [x] 22. **Online clustering, every family that can be made to fit — investigated
      2026-09-04, on the branch `online-clustering`.** The user asked for "all the
      clustering types that may be possible Online", so: every family in river 0.26.1,
      MOA, scikit-learn and Spark plus the two survey papers, each decided against the
      contract. `docs/CLUSTERING.md` (§11h) is the answer — the papers read with claims
      cited by line (DenStream's fading function, definitions and pruning thresholds;
      CluStream's micro-cluster; BIRCH's CF triple; DP-means; Cappé–Moulines eq. 15;
      Bottou–Bengio's `1/n_k` as the Newton rate), the four implementations read with
      `file:line` (none of which is both chunk-invariant and bounded, none of which
      reads a real clock, none of which labels a row before learning it), nine designs
      prototyped in numpy (`scripts/clustering_proto.py`) and measured
      (`scripts/clustering_experiments.py`) against Lloyd refits and scikit-learn's
      `MiniBatchKMeans`. Every guarantee holds bit-exactly. The one real defect of
      sequential k-means — two centres collapsing onto one component under drift, on 5
      of 20 streams — is fixed by a split–merge move on a slower clock. On seven
      hard geometries (§7.8) being online costs at most 0.04 ARI against batch
      Lloyd's; the family costs everything — every k-means and GMM scores 0.000
      on concentric rings, where `micro` (DenStream-style micro-clusters with a
      linkage macro step) reaches 0.998 / 0.999 / 0.998 on moons, rings and rows
      against DBSCAN's 1.000, with a measured rule for the threshold that decides
      it. `micro` is the design worth the build decision; §0 of the doc and
      ENHANCEMENTS §4 give the seven reasons and the structural limits. Merged to
      `main` 2026-09-04 as documentation and numpy prototypes. Nothing in the
      Rust crates; whether to build it is the user's call and §9 costs it.
      Research sources stay under the gitignored `.cache/research/`.

<a id="task-23"></a>
- [x] 23. **`kmeans` — online k-means in the crates.** `crates/online-core/src/cluster/`
      (`summary.rs`: the §6.1 mean-form summary and the diagonal feature moments the
      metric reads; `kmeans.rs`: seeding over a bounded warm-up buffer, the assignment,
      the checkpointed centre update, the split–merge move on its own clock), every
      step of `docs/EXTENDING.md`, a from-scratch numpy reference
      (`tests/reference_cluster.py`) held bit-exact, the prototype as a second oracle,
      large streams, and the edge cases §11a lists.
      Built 2026-09-05, on branch `clustering-build`: `ModelState::KMeans`, the spec
      variant with `k`, `warm_rows`, `seed_rule`, `seed`, `update_every`,
      `split_merge`, `split_merge_every`, `dead_frac`, `standardize`; outputs `cluster` (`i32`),
      `dist`, `dist2`, `n_eff`, centres as `coef` (`cluster{j}` slots, one per
      feature; `coef_index` and `unnest` follow); `ModelKind::is_unsupervised`
      refuses every residual diagnostic by name for it and for `ew_cov`. The far-row
      design §11a records replaced the doc's first split–merge move after its
      measurements failed (a seeding artefact had been read as recovery). Tests: 61
      in `tests/test_kmeans.py` (22 bit-exact against the oracle, large streams to
      40k rows, the null-row / zero-weight / constant-feature / min_periods /
      chunk-invariance / save-load edges, the stranded-centre and jumped-blob
      recoveries with their latencies pinned), the core golden and contract suites,
      the Python golden pipeline on three OSes.

<a id="task-24"></a>
- [x] 24. **`micro` — DenStream-style micro-clusters with a linkage macro step.**
      `cluster/micro.rs` on the same summary; ids monotone and never reused; the label
      is the id the row would be absorbed by, computed before the update; `macro_link`
      derived from the observed spacing at each checkpoint (§6.5) with the parameter
      as an override; accuracy measured in-test against a numpy DBSCAN on moons and
      rings.
      Built 2026-09-05, on branch `clustering-build`: `ModelState::Micro`, the spec
      variant with `eps` (required), `beta_mu`, `max_clusters`, `prune_every`,
      `macro_link`, `standardize`; outputs `cluster` (`i64` macro label), `dist`,
      `micro` (`i64` id), `outlier` (`bool`), `n_clusters`, `n_micro` (`i32`),
      `n_eff`, and a ragged `coef` (`Source::Id` and `Source::Flag` are new; the
      id and the label are separate columns, so §6.5's rule 2 holds for the id and
      the label is still a cluster). The admission rule, the derived link, the ξ
      semantics and the two failure regimes are in §11a below. Tests: 63 cases in
      `tests/test_micro.py` (17 bit-exact against the oracle over four geometries,
      seven knob settings, nulls/weights/clock gaps and `predict`; 20k-row shapes
      against an in-test DBSCAN ceiling, 200k rows in 4-D, a cluster born and one
      dying, 5% noise; ids, the cap, promotion and pruning traced row by row, a
      heavy row, the infinite halflife, standardization, chunk invariance,
      save/load, groups, the ragged `coef`, the expression and CLI paths, every
      refusal), 21 core unit tests, the golden and contract suites, the Python
      golden pipeline.

<a id="task-25"></a>
- [x] 25. **E36: adaptive conformal intervals** (`lo`/`hi`/`coverage` per slot) on every
      regression model, O(1) state per slot; oracle + large-data coverage tests.
      Built 2026-09-05, on branch `clustering-build`: `online_core::Conformal`
      (`conformal.rs`, with `norm_ppf` for the warm start), spec fields `conformal`
      (the coverage level) and `conformal_rate` (0.05, in units of the slot's
      `sigma`); `StreamState.conformal` as a `#[serde(default)]` per-slot vector
      (schema stays 2); `Source::Conformal`; fields `lo_<slot>`, `hi_<slot>`,
      `coverage_<slot>` after `resid_z`. The fields are not `pred_lo`/`pred_hi`
      as sketched above: `pred_` marks a prediction for `eval.unpack` and the
      README's field grammar, and a bound is not one. The update rule, the warm
      start and the guarantee are in §11a below. Tests: 70 cases in
      `tests/test_conformal.py` (a longhand replay bit-exact for every regression
      model over a grid, nulls, zero and varying weights, an irregular clock and
      groups; the telescoped `1/T` bound as a hard inequality and coverage
      within 0.01 of target on four 200k-row residual regimes; fields, validation,
      refusals, nulls, zero weights, warmup, out-of-sample-ness, chunk
      invariance, save/load, `predict`, a drift reset, the runner and the
      expression), 9 core unit tests, the golden and API-surface snapshots.

<a id="task-26"></a>
- [x] 26. **E37 + E38 on `ew_cov`:** Mahalanobis distance (`stats: "mahal"`) and EW-PCA at
      checkpoints (`pca`, `pca_every`); oracles via `gram()` and numpy.
      Built 2026-09-05, on branch `clustering-build`: `EwCovStat::Mahal` (one
      slot `mahal`, `solve_spd` on `C + s·prior·I` each row, so it needs
      `precision_prior`), `EwCovCfg::{mahal_quantiles, pca, pca_every}` (all
      `#[serde(default)]`, schema stays 2), `online_core::Pca` (faer
      `self_adjoint_eigen`, continuity-signed), fields `mahal`, `mahal_q<p>`,
      `pc<j>_var`, `pc<j>_share`, `pc<j>_<feature>`, `pc<j>_score`;
      `output_index` kinds `mahal`, `mahal_q`, `pc_var`, `pc_share`,
      `pc_score`, `pc_loading`. Decisions in §11a. Tests: 40 cases in
      `tests/test_ew_cov_scores.py` (a Welford replay of the stream — clock
      gaps, `max_dclock`, zero and null weights, skipped rows — held to the
      solve at 1e-9; numpy `eigh` with the continuity rule at every refresh
      for two cadences; χ² calibration and a 3-factor recovery at 200k rows;
      a covariance switch; chunk invariance, save/load mid-cadence, `predict`,
      the expression, the runner and a TOML config; fields, validation, edge
      cases), 31 core unit tests, the API-surface, error-message and
      output-index suites extended.

<a id="task-27"></a>
- [x] 27. **E39: class-conditional `ew_cov`** (`class` column, per-class moments).
      Built 2026-09-05, on branch `clustering-build`, as a model of its own:
      `po.spec.ew_class(name, features=, label=, classes=, covariance=,
      precision_prior=)` — one `EwCov` per declared class, scored by Bayes'
      rule over Gaussian classes with `covariance` = `full` (QDA, default),
      `shared` (LDA: the class-weighted pool, one factorization) or
      `diagonal` (naive Bayes). Outputs `class` (String), `p_<class>` per
      class, `n_eff`, `coef` = the class means (`coef_<class>_<feature>`).
      `online_core::{EwClass, EwClassCfg, Covariance}`,
      `solve::quad_forms_logdet` (all quadratic forms and the log-determinant
      off one Cholesky), `ModelState::EwClass` (schema stays 2). The label
      column rides as `targets[0]` through the bank and is read as a key
      (`label_column`: cast to String, mapped to the class index, an
      undeclared value is an error naming the row); `Source::Label` and
      `F64Column::finish_label` materialize the class name. Decisions in
      §11a. Tests: 51 cases in `tests/test_ew_class.py` (an in-file replay
      oracle — per-class weighted Welford in the core's operation order —
      holding weights, means and `n_eff` bit-exact and the posteriors to
      1e-9 for every shape over null labels, null features, zero and null
      weights and a capped irregular clock; 200k rows × 6 features × 3
      classes within 0.001 of the Bayes rate and calibrated to 0.01; the
      shapes told apart on data that separates them; a class swap relearned;
      an unseen class, late labels, integer/boolean/categorical labels, an
      undeclared label, the input bound, chunk invariance, save/load,
      `predict`, groups, the grid, `coef` and its index, the expression, the
      lazy path, the runner, the CLI and the refusals), 14 core unit tests
      plus 2 for the solve, the golden, contract, API-surface, error-message
      and registry suites extended.

<a id="task-28"></a>
- [x] 28. **E40: constrained coefficients** on `sgd` / `pa`: `coef_min` /
      `coef_max` (a number for every slope or one per feature, `inf` for no
      bound) and `coef_sum`, one projection (`online_core::Constraint`) for
      the box, the simplex and any box with a sum; intercept free; the
      constraint in the caller's units under `scale_features`. Verified by a
      Python replay of both models and the projection held bit-exact to the
      bank over six constraint sets × three schedules / three PA modes with
      nulls, zero and NaN weights and an irregular clock, an independent
      bisection-and-KKT check of the projection, 200k-row recovery on the
      simplex / a sign / a hyperplane / a box / a million-fold scale gap,
      and the edge cases (pinned slopes, list vs scalar, explicit ±inf,
      several targets, zero-weight and null-target rows, the input bound,
      chunk invariance, save/load, `predict`, groups, the expression, the
      lazy path, the runner, the CLI, every refusal by name); 9 core unit
      tests in `constraint.rs`, 7 in `sgd.rs`, 4 in `pa.rs`, two goldens.

<a id="task-29"></a>
- [x] 29. **E41: diagonal transition `φ^d`** on `kalman` (coefficient dynamics):
      `revert_halflife`, a scalar or one per slot (intercept first, `inf` =
      random walk, the default), `β ← Φβ`, `P ← ΦPΦ + Q·d` before each
      row's prediction, `predict` propagating by the same `Φ`. Verified by
      the Python replay held to 1e-9 against the bank over thirteen
      configurations (per-slot / shared `P`, explicit `q`/`obs_var`/`p0`,
      nulls, skipped features, zero weights, no intercept,
      `standardize=False`, a capped gap), the exact shrink over an irregular
      clock with skipped rows to 1e-11, the covariance bound
      `q/(1−φ²)` after 200k null rows, 300k-row tracking (sparse / dense /
      random-walk truth), and the edge cases (`inf` bit-identical in every
      spelling, scalar == list, zero-weight rows, `predict` over the clock
      distance, chunk invariance, save/load, groups, the expression, the
      lazy path, the CLI, every refusal by name); 7 core unit tests, the
      contract with and without reversion, two goldens.

<a id="task-30"></a>
- [x] 30. **E42: a sequential e-process test** — `seqtest`, a model in two
      modes: the sign of each target column, or, with `a`/`b`, whether one
      spec of the bank predicts closer than another (`|resid_b| − |resid_a|`).
      Two one-sided Kelly bettors per target at the Krichevsky–Trofimov
      stake on the counts before the row; `log_e_pos/neg`, `n_pos/neg`,
      `n_eff` (`log_e_a/b`, `wins_a/b` in compare mode), all before the
      row. The bank runs in two phases (every other spec, then the
      comparisons over the residuals just assembled) and returns columns in
      spec order. Verified by a scalar replay to 1e-12, the closed-form KT
      wealth `2ⁿ B(n⁺+½, n⁻+½)/π` through `math.lgamma`, `po.eval.seqtest`
      bit-identical to the bank on 1M rows (column) and 300k (compare), the
      guarantee as a crossing rate over 20k fair-coin and 20k dependent-null
      streams (2.2% at α = 5%, 0.4% at 1%), power at 60% / 55%, and the
      edge cases (ties, nulls, ±inf, beyond-bound, denormals, integers,
      warm-up with `n_eff` through it, per-target `min_periods`, chunk
      invariance incl. interleaved-group comparisons, save/load, pickle,
      groups and a null key, session and clock restarts, lazy path, runner,
      CLI incl. `--dry-run`, the expression and its `a`/`b` refusal, two
      comparisons in one bank, scoring, a refused chunk, every refusal by
      name); 14 core unit tests, the contract, two goldens, 9 bank tests.

<a id="task-31"></a>
- [x] 31. **Performance and parallel-performance deep dive** over the new models and
      enhancements (`docs/PERFORMANCE.md` §13, `benchmark.py`). Bit-exact
      throughout, two dumps against the Task 30 build say so; per 400k rows
      at k = 20 the default `ew_cov` 758 → 222 ms, full `ew_class` 1510 →
      808, PCA at `pca_every=100` 332 → 162, `micro` 37 → 24, the simplex
      179 → 157; at 14 threads over 64 groups `ew_cov` 276 → 152, `ew_class`
      324 → 191 per 800k rows. The finding: a wide model's speed depended on
      the caller's chunk size through the slot stride (a power-of-two chunk
      2–3× slower), fixed by processing in cache-sized runs with an odd-line
      stride.

<a id="task-32"></a>
- [x] 32. **Prepare 0.2.0**: version bump, CHANGELOG, README and VALIDATION numbers
      regenerated, gate and CI green. Tag and Release dispatch are the user's steps.

<a id="task-33"></a>
- [x] 33. **Diagonal standardizer for `kalman` and `sgd`** (`EwDiag`), the O(k²)
      → O(k) item Task 31 deferred; `SCHEMA_VERSION` 2 → 3 with schema-2 loaders
      for both models and a frozen schema-2 bank fixture
      (`crates/online-polars/tests/state_schema2.rs`). Bit-exact: the Task 31
      dump recipe against a build of the previous commit, plus `EwDiag` vs
      `EwCov` compared as bits. Per 400k rows at k = 20: `kalman` 223 → 179 ms,
      `sgd` with `scale_features` 109 → 62; at k = 50 1065 → 908 and 279 → 121.

<a id="task-34"></a>
- [x] 34. **Last-row diagnostics in the bank file**: the output-struct fields as
      of the last training row, per (spec, group), saved with the state and
      readable from a loaded bank without the output frame
      (`ModelBank.last_row()`, `Bank::last_row`). Additive field, no format
      bump (the `rows_fed` precedent).

<a id="task-35"></a>
- [x] 35. **The training data in the bank file**: per (spec, group), what the
      stream was fed -- row counts (fed, processed, skipped, learned, zero
      weight), the weight sum, the clock range, session changes, backwards
      clocks and resets, and per-column count / nulls / mean / std / min /
      max over every row fed -- saved with the state, read back as
      `ModelBank.summary()` and `ModelBank.describe()`. Undecayed, in row
      order (Welford), so chunking cannot move a bit. Additive field, no
      format bump; a 0.1.x file reports nulls. With it, the state-file tests
      made rigorous: every facet equal across save/load, a re-save byte
      for byte the first, truncated and bit-flipped files refused without a
      panic, and a frozen 0.2.0 fixture (`state_schema3.rs`).

Tasks 36–43 are `docs/ENHANCEMENTS.md` §9 (E43–E53): what the bank needs
to run at the width, target count and block count its design allows.

<a id="task-36"></a>
- [x] 36. **`ew_cov` with no per-row output** (E43): `stats=[]` is legal and
      means "accumulate only"; the spec emits `n_eff` and its value is its
      state (`gram()`, `describe()`, `summary()`).

<a id="task-37"></a>
- [x] 37. **`marginal` model** (E44): per-(feature, target) EW mean, variance,
      covariance, Σw and Σw² — `O(p·T)` per row, `n_eff` the only output —
      read as a long frame by `ModelBank.marginal()`.

<a id="task-38"></a>
- [x] 38. **Complete the Gram export** (E45): `gram()` gains per-target means
      and variances and Kish `n_kish` (features and per target); Σw² tracked
      beside Σw. Additive fields, legacy states report `None`.

<a id="task-39"></a>
- [x] 39. **`po.gram`** (E46): merge / subset / correlation / solve /
      lasso_path / coef_stats / vif / condition over `gram()` dicts, numpy
      only, each held against the model that computes the same thing online.

<a id="task-40"></a>
- [x] 40. **`label_delay`** (E47): a common parameter that holds each learned
      row until the clock reaches `t + delay`; plus `po.prep.embargo` for the
      doubled-stream recipe.

<a id="task-41"></a>
- [x] 41. **The Gram update, measured** (E48): the symmetric-half idea is
      rejected — it is neither bit-identical nor faster — and the deviation
      hoist that *is* both ships in its place.

<a id="task-42"></a>
- [x] 42. **Mergeable evaluation sums** (E49): `po.eval.sums` /
      `merge_sums` / `from_sums`.

<a id="task-43"></a>
- [x] 43. **A run with no per-row output** (E50): `po.run(output=None,
      save_state=)`, `online --no-output`; with it E53, `targets` optional
      in TOML for the unsupervised models.

<a id="task-44"></a>
- [x] 44. **Freeze a schema-4 fixture**: `state_schema4.rs`, as
      `state_v1.rs`, `state_schema2.rs` and `state_schema3.rs` freeze theirs,
      now that tasks 40–43 have stopped moving the layout.

Tasks 45–56 build `docs/ENHANCEMENTS.md` §10 (E54–E64) in the order that
section gives, plus the two tasks the examination found were missing: 47,
the ring-clearing signal three of the models need and no model currently
receives, and 56, the fixture that closes the batch. Each is written to be
built without re-deriving the design — the decisions, the maths, the state
and the oracles are in §11a under *Preparing E54–E64 for implementation* —
and every new model walks `docs/EXTENDING.md`'s eighteen steps. E63 is a
note, not a task.

<a id="task-45"></a>
- [x] 45. **Closed-group emission** (E54): `group_close = "monotone" |
      "session"` as a common parameter; `ModelBank.closed_groups(spec=None,
      *, drop=True)`; the sidecar table through `po.run(closed_groups=)`,
      `lf.online.fit_predict(closed_groups=)` and `online --closed-groups`;
      `po.gram.from_row`. `SCHEMA_VERSION` 4 → 5: a spec field moved every
      bank file's bytes (the `label_delay` precedent); schema-4 files load
      and continue to the bit. Acceptance: a closed row equals `gram()` read
      at the same point, field for field and bit for bit; `eig_*` equals
      `numpy.linalg.eigh` of the row's own `comoments`; the sidecar's bytes
      are invariant to chunking; a smaller later key, a null key, a
      non-orderable key dtype, `group_close` without `group`, `"session"`
      without `session` or with `session_gap`, and `group_close` with
      `label_delay` are each refused by name; the last group never closes
      and stays readable through `gram()`; a mid-stream save/load keeps the
      unclosed groups, the high-water mark and any undrained rows.

<a id="task-46"></a>
- [x] 46. **`deco`** (E55): Engle–Kelly equicorrelation as an `OnlineModel`,
      `O(m)` a row on `EwDiag`-standardised features; `dynamics = "ew" |
      "linear"`, `blocks`; outputs `u`, `rho`, `loglik`, `n_eff`, `coef =
      [rho]`. Acceptance: `u` equals the longhand pair sum; one pair under
      `"ew"` equals `ew_cov`'s `corr` on the same standardised columns to the
      bit; `loglik` equals a dense `numpy` Gaussian density; one block holding
      every feature reproduces the unblocked `u`; a zero-weight first row is
      guarded; the standard sweeps and the clock-rescaling test pass.

<a id="task-47"></a>
- [x] 47. **The ring-clearing signal**: `ClockAdvance::capped`,
      `RowPlan::capped`, `OnlineModel::clear_lags` (a default no-op), called
      by the stream on a session change or a capped gap (a reset already
      rebuilds the model). The rule every row-lagged state follows from here
      on, with its EXTENDING.md step. Enabling task for 48, 50 and 54.

<a id="task-48"></a>
- [x] 48. **Lagged co-moments on `ew_cov`** (E56): `lags=[...]`, `stats=[...,
      "lagcorr"]`, `C_ℓ' = a·C_ℓ + a·b·d_t d'_{t−ℓ}` with both deviations
      against the pre-row mean; `gram()` gains `lags` and `lag_comoments`;
      the E54 row carries both. Acceptance: `ℓ = 0` equals `comoments` to the
      bit; the longhand `numpy` recursion agrees; the ring clears on the
      three events of task 47 and on nothing else; `n_eff` and Kish are
      bit-identical to the spec without `lags`; chunk invariance.

<a id="task-49"></a>
- [x] 49. **`po.prep.refresh_time`** (E58): refresh-time sampling as a Rust
      operator in `online-polars` (`refresh.rs`, no model), exposed through
      `online-py` and wrapped as a lazy IO-plugin source in `po.prep`, the way
      `lf.online.fit_predict` is. Acceptance: a longhand Python loop on
      Poisson streams; an 8/9/10-tick three-series example with `N = 7` and
      retained fraction `21/27`; a synchronous input returned unchanged;
      identical output from 1 and 1000 batches.

<a id="task-50"></a>
- [x] 50. **`rcov`** (E57): the multivariate realised kernel, the
      pre-averaged (modulated) realised covariance and the plain realised
      covariance as a group-scoped, undecayed `OnlineModel` whose value is its
      E54 row. Acceptance: `kind = "plain"` equals `n ×` `ew_cov(lam = 1)`'s
      raw second moment at close, to the bit; the kernel and the pre-averaged
      estimator each equal a longhand `numpy` implementation on the same
      returns; Parzen PSD on adversarial streams; the `ψ`/`Φ` constants at
      their closed forms; a block with fewer than `2·jitter` rows gives nulls;
      zero-weight rows stay out of the ring; chunk invariance.

<a id="task-51"></a>
- [x] 51. **`po.corr`** (E62): the correlation-matrix helpers in `gram.py`'s
      style — Fisher z, Higham's nearest correlation matrix, constant-target
      shrinkage, equicorrelation, absorption, spectral and block forms,
      Marchenko–Pastur edges, signal share, forecast losses, the Epps
      inversion, Fisher standard errors. Acceptance: Higham's published
      examples; PSD and unit diagonal on random inputs; `qlike` zero at
      `fcst = real` and positive elsewhere; `equicorr_row` equals task 46's
      `u`; every function held to a longhand check.

<a id="task-52"></a>
- [x] 52. **`po.sim.regimes`** (E64): the seeded regime simulator, `numpy`
      only, with asynchronous observation, noise, AR(1), a volatility state,
      a cycle_profile factor and a volume process; returns `rows`, `truth_rows`,
      `truth_blocks`. Acceptance: block correlations recover the per-state
      matrices within the Fisher-z floor; the Epps curve of an asynchronous
      run rises with the sampling interval; a seed reproduces bytes.

<a id="task-53"></a>
- [x] 53. **`hmm`** (E60): the Hamilton filter over `K` Gaussian states with
      responsibility-weighted `EwCov`/`EwDiag` updates, an EW transition
      estimate from the filtered joint, `kmeans` seeding, `exog_tvtp`.
      Acceptance: the reduction to `ew_class` (Π uniform, `learn = False`,
      the states built from a fitted `ew_class`'s own `EwCov`s) to the bit; a
      longhand `numpy` filter at fixed parameters; `predict` is the step
      without the step; recovery of Π and the means on task 52's streams as a
      `docs/REGIMES.md` experiment; the sweeps.

<a id="task-54"></a>
- [x] 54. **`corrchange`** (E59): the Wied–Krämer–Dehling (2012)
      closed-sample constancy test run span by span (`horizon` rows, the
      paper's `D̂`, Kolmogorov critical values, Bonferroni over pairs) and
      the two-window `vech` statistic with a permutation critical value.
      Acceptance: the monitor's size and power reproduce WKD's Table 1
      (`.040/.035/.041` at `T = 500`; `.587` power on a `0.5 → 0.7` step)
      to Monte-Carlo error; `D̂` and `Q` against longhand `numpy` to
      `1e-12`; the window statistic to the bit; run length to false alarm
      and detection delay of the window kind on task 52's streams in
      `docs/REGIMES.md`. The Wied–Galeano (2013) sequential detector is a
      §10 follow-up, not part of this task.

<a id="task-55"></a>
- [x] 55. **`bocpd`** (E61): Adams–MacKay run-length recursion with
      normal-inverse-Wishart, per-feature normal-inverse-gamma and the robust
      diffusion-score-matching posterior; tail truncation. Acceptance: a
      longhand `numpy` Algorithm 1 on a univariate stream; a variance step
      detected; truncation changing nothing above `truncate`; the robust
      variant ignoring one 20-σ row where the Gaussian one restarts; the
      sweeps.

<a id="task-56"></a>
- [x] 56. **Freeze a schema-5 fixture and close the batch**:
      `state_schema5.rs` as task 44 froze schema 4, once 45–55 have stopped
      moving the layout, carrying a monotone bank with an undrained closed
      row, an `ew_cov` with `lags`, and one of each new model; ENHANCEMENTS
      §10's rows gain their task numbers; the README's model table, the
      CHANGELOG's `[Unreleased]` and `docs/RELEASE-READINESS.md` are brought
      up to the batch.

<a id="task-57"></a>
- [x] 57. **Fix what the review of 45–56 found** (`docs/REVIEW-E54-E64.md`):
      29 items -- four of them wrong answers rather than missing guards --
      each with the test that catches it, and one (B4) made and then
      reverted when Linux CI showed what it cost. The four: `ew_cov`'s lag ring read
      its depth from `VecDeque::capacity()` and never refilled after a save
      taken before it was full (L1); `rcov`'s `clear_lags` dropped the
      end-jitter ring instead of closing the stretch, losing `m` returns and
      re-emitting the next one `m + 1` times (R2); `hmm`'s `min_periods`
      gated the *update* as well as the report, so under decay a filter
      could never learn at all (H1); `bocpd` silently dropped a row whose
      hazard column was `<= 1` (B1). Acceptance: a regression test per item,
      the `predict == step` contract extended to the value a model reads out
      of the targets slot, and hard rule 8's recursion checked against
      zero-weight rows for every model.

<a id="task-58"></a>
- [x] 58. **Documentation pass, 2026-09-06.** Every document read against the
      code and brought current; the README gains a table of contents, a link
      from each model to its builder in the API reference and to its Rust
      source, and two regroupings (*Preparing a stream*, *Reading the fit*);
      every builder docstring states each keyword's default, checked against
      `stream.rs`; `docs/README.md` maps every document to what it is for;
      each document under `docs/` opens with a dated status line. Section
      numbers and file names are unchanged, because the code cites them
      (§11a records the rule). Acceptance: the README's blocks still run
      (`TestReadmeExamples`), every builder still has its README heading
      (`test_the_readme_documents_every_model`), `sphinx-build -W` is clean.

<a id="task-59"></a>
- [x] 59. **`llms.txt`, 2026-09-06.** The [llmstxt.org](https://llmstxt.org)
      map for coding agents at the repo root, copied into the docs build by
      `html_extra_path` so the same file answers at
      `hgilde.github.io/polars-online/llms.txt`. It carries the two streaming
      surfaces in full and the rules an assistant otherwise guesses wrong,
      chief among them that `po.spec.ewridge` is `type = "ew_ridge"` in a spec
      dict — the one name spelled two ways. Acceptance: `tests/test_llms_txt.py`
      holds every model name to the registry, every repository link to a file
      that exists, every README anchor to a heading and every reference link to
      a built page, and each of those five assertions was checked against a
      deliberately broken copy.

<a id="task-60"></a>
- [x] 60. **The README links into the API reference, 2026-09-06.** The model
      table's name links each model to its builder's entry in the Python API
      reference, and a `math` link beside it still reaches the section below;
      the first prose mention of each public object — `ModelBank`, `po.run`,
      the frame namespace, `po.eval`, `po.gram`, `po.corr`, `po.prep`,
      `po.sim`, the spec helpers, `InMemoryExpressionWarning` — links to its
      entry too. The Python reference is the destination everywhere: it is the
      surface a caller uses, and it is what the docstrings are built from. The
      `*Rust:*` links stay source links, because rustdoc is neither built in CI
      nor published (the crates are not on crates.io, so there is no docs.rs);
      if it is ever published, that line is where it goes. Acceptance:
      `tests/test_api_links.py` resolves every reference link to a page that is
      built and an object that exists, holds the table to the registry, and
      holds every model section to its `*API:*` and `*Rust:*` lines — all four
      checked against a deliberately broken README.

<a id="task-63"></a>
- [x] 63. **`window`: an EW accumulator with a hard cutoff.** A halflife `h`
      with `window = 3h` must guarantee that nothing older than `3h` of clock
      contributes at all — not "contributes 12.5%". Designed in §13 below;
      the colliding names are renamed first (task 63a), the mechanism and
      `ew_cov` land next (63b), the Gram models after (63c).
  - [x] 63a. **Free the word.** `rcov`'s `window` is a pre-averaging length in
        ticks and becomes `preavg_rows`; `corrchange`'s `window` and
        `horizon` are one concept — rows per comparison block — under two
        names, and become `span_rows`; `bocpd`'s `truncate` is a probability
        floor and becomes `prune_below`, since "truncate" now means the
        window. No compatibility shim and no dual spelling: the spec keys are
        renamed, the frozen state fixtures are regenerated, and a state saved
        by 0.2.0 does not load.
  - [x] 63a′. **The naming pass, 2026-09-07.** `preavg_ticks` was market
        jargon in a library whose clock is deliberately generic, and five more
        names did not say what they were: `rcov.n_max` → `block_rows`,
        `rcov.h_max` → `max_bandwidth`, `kmeans.sm_every` →
        `split_merge_every`, the ridge family's `coef0` → `coef_prior` (it is
        the prior mean, not the coefficient of feature 0), and
        `refresh_time`'s `n_obs_<s>` column, which had been `n_ticks_<s>`.
        Names that appear as symbols in a formula the docstring quotes —
        `theta`, `jitter`, `q`, `p0`, `c`, `beta_mu`, `share_p` — were left,
        because renaming them breaks the correspondence a reader checks
        against the cited paper.
  - [x] 63b. **The mechanism, and `ew_cov`.** `Snapshots<S>` in
        `online-core`, the truncated view, `window` and `window_every` on the
        `ew_cov` spec, refused by name everywhere else. Acceptance: the
        oracle in §13.4.
  - [x] 63c. **The Gram models.** `ewridge` done 2026-09-07: the Gram, the
        per-target cross-moments and the residual variance are truncated by
        the same identity, and the solve runs on the result, so the fit
        provably contains no row older than the window. `n_eff`, `sigma` and
        `resid_z` follow it. Refused with `ridge_decay` (the decaying prior's
        scale is a product over the whole stream) and with `session_shrink`
        (a second accumulator under a longer halflife). Acceptance: a direct
        weighted-least-squares solve over the rows inside the window, in Rust
        and again in Python against numpy.
        `lasso` done the same day: it solves from the same Gram, so the view
        threaded through `standardized()` alone. Its selection error is
        truncated with it — choosing `lambda` on the whole history while
        fitting on the window picks a path point for rows the coefficients no
        longer see — which `-D warnings` caught as an unused binding after the
        docstring already claimed the behaviour. A window can therefore change
        the *support*: a feature with no in-window evidence goes to exactly
        zero.
        `marginal` done the same day: it emits nothing per row, so the window
        applies at the readout -- every moment a pair is built from, the
        weight and both means and the three centred second moments, truncated
        one pair at a time so a readout stays O(1). The case for it is
        sharpest here: two regimes of opposite sign average to nothing over a
        long history, so an unwindowed screen reports corr 0.0006 where a
        40-unit window reports -0.99. *Two more of mine the tests caught*:
        `Marginal` was a unit-like spec variant (`Marginal {}`), so the fields
        had to be added to the enum and four `matches!` patterns updated; and
        every part of the window was written except the ring maintenance in
        `step`, which left it inert until the oracle failed.
        `ew_class` done the same day, and it is the one that costs: pure
        decay leaves a covariance unchanged, which is why `covariance="full"`
        caches its factor between rows, and a *truncated* covariance moves
        every row — so the cache dies every row and the shape pays one
        `O(k^3)` factorization per class per row. Stated in the keyword's own
        docstring and measured in `docs/PERFORMANCE.md` §16, which also shows
        the windowless path unchanged (±4%, both signs, against a build from
        before the feature). All five models of §13.2's first row now carry
        `window`.
        *Two things the tests caught*: the compact msgpack encoding writes a
        struct as an **array**, so a `skip_serializing_if` field must be
        **last** or it shifts every field after it — the state round-trip
        failed with "invalid type: boolean, expected f64" until both the
        model's `win` and the config's two keys moved to the end. And the
        first oracle disagreed because the *test* fed a clock that ran
        backwards: `lcg` there is `[-1, 1)`, not `[0, 1)`.

<a id="task-64"></a>
- [x] 64. **Document the output struct of every model** (`docs/ENHANCEMENTS.md`
      E65). Each model's README section and builder docstring gains a table of
      the fields it writes — name, dtype, when null, which switch adds it —
      and a test holds each documented set to `po.spec.output_fields(spec)` for
      a canonical spec, so an undocumented field cannot ship. Today only the
      *grammar* of the names is written down, and the per-model prose is
      inconsistent: `corrchange` lists its outputs, `kmeans` does not.
      Done 2026-09-07 as `docs/OUTPUTS.md`, one section per model, rather
      than a table in twenty-one docstrings: the *field lists* are generated
      from `output_fields` on a canonical spec and the *meanings* are written
      once per field stem in `scripts/outputs_doc.py`, so the two cannot
      disagree. Each model's README section links to its section beside the
      `*API:*` and `*Rust:*` links. `tests/test_outputs_doc.py` regenerates
      and compares, so a new field cannot ship undocumented — the generator
      writes `**undocumented**` for a stem it has no meaning for, and a test
      fails on that string.

<a id="task-65"></a>
- [x] 65. **`marginal(lags=)` and `n_serial`, 2026-09-07**
      (`docs/ENHANCEMENTS.md` E66, `docs/MARGINAL-LAGS-AND-BINS.md`).
      `marginal`'s `t` is built on `n_kish`, which is right for unequal
      weights and silent about serial dependence: on a smooth stream
      consecutive rows are nearly the same observation, so `t` reports
      evidence that is not there, and nothing in the contemporaneous moments
      can see it. `lags=` accumulates the pair's moments at those lags with
      `ew_cov`'s recursion — bit-identical to `ew_cov(lags=)`, which is the
      strongest check available, since both centre at the pre-row mean and
      mix with the same `a`/`b`. From them `serial_rule` forms Bartlett's
      factor and reports `n_serial`, `t_serial`, `phi_x`, `phi_y`, plus the
      four `lagcorr_*` list columns — of which `lagcorr_xy` against
      `lagcorr_yx` says whether a feature leads or follows its target, which
      is worth having on its own (2026-09-27: the lead/follow reading holds
      for two series that describe the same moment, not against a
      forward-looking target, E75). Two independent AR(1) series at
      `phi = 0.9` and `0.8` give `t = 2.39` and `t_serial = 1.03`. Three
      names changed from the sketch; see the enhancement's row for which and
      why. No schema bump: the bank file is a msgpack *map*, so a key with a
      `default` is backward-compatible.

<a id="task-66"></a>
- [x] 66. **`marginal(bins=)` and `split_gain`, 2026-09-07**
      (`docs/ENHANCEMENTS.md` E67, `docs/MARGINAL-LAGS-AND-BINS.md`).
      Everything else `marginal` reports is linear, and a feature can be
      strongly related to a target with `corr` at zero. A histogram of the
      target's moments inside the feature's bins gives the response curve
      (`bin_edges`, `bin_n`, `bin_mean_y`, `bin_var_y`) and the best single
      cut of it (`split_gain`, `split_at`, `split_gain_t`) — a regression
      stump's gain, `O(bins)` of state per pair and a binary search per pair
      per row. Decay stays `O(1)` per row by keeping the sums undecayed
      against one scale factor per group, renormalized at `s < 1e-150` on a
      row the clock alone decides. Learned edges hold their warm-up rows and
      replay them, so the histogram is what it would have been had the edges
      been known first — checked to the bit. Four parameter names and the
      quantile mechanism changed from the sketch; the design doc's API
      section says which and why. Ragged bins, since a binary feature has two
      and a constant one has none. `bins` with `window` is refused.

      Three bugs found by the tests written for it, all pre-existing or
      newly introduced here and none visible in shipped behaviour:
      (a) E66 had put a `skip_serializing_if` field in front of `win`, which
      breaks the *compact* msgpack encoding, where a struct is a bare array
      and a skipped field slides everything after it — `tests/state_encoding.rs`
      now sweeps every combination in both encodings; (b) `window_every`
      without `window` was validated by `ew_cov` and `ewridge` but not by
      `marginal`, `lasso` or `ew_class`, and in the compact encoding it
      decoded *silently* as `window = 1`; (c) `serde_json`'s default float
      parser is fast rather than correctly rounded and moved a spec float by
      one ulp — nothing for a halflife, everything for a bin edge, since
      edges read back from an earlier run are data values and a row sitting
      exactly on one is the common case. The crate's `float_roundtrip`
      feature fixes it; `crates/online-polars/tests/spec_floats.rs` holds it.

      And two costs that a test cannot see, found by measuring the plain
      path against a build from before either task (`docs/PERFORMANCE.md`
      §17). A stream with no lags and no bins had gone from 69.2M to 58.0M
      rows/s: task 65 had put the lag update inside `learn`'s per-target
      loop, and a call there -- even one never taken -- makes the compiler
      assume the callee could reallocate `mx`, `sxx` and `sxy`, so it
      reloads their base pointers every iteration and stops vectorizing;
      and the `O(p)` finiteness scan that guards the lag ring sat outside
      the `if let Some(lag)` that needed it, so every row of every
      `marginal` walked all `p` features to decide whether to push into a
      ring that did not exist. Hoisting the first into its own pass
      (`learn_lags`, bit-identity intact) and moving the second inside its
      guard puts the path back at 72.2M. Both look like ordinary guard
      clauses in a diff, which is the argument for keeping a benchmark
      cheap enough to run: `cargo run --release -p online-core --example
      marg_bench`.

<a id="task-67"></a>
- [x] 67. **Review of tasks 65 and 66, 2026-09-07.** Each changed model
      read in its current form for bugs, logical errors, performance and
      the edge cases its tests missed; every finding applied, since nothing
      had been released or pushed, so the layouts could change with no
      `SCHEMA_VERSION` bump. What changed, and the decision behind each:

      (a) **A capped gap is a `λ` of zero.** `max_dclock` caps a gap at
      `max_dclock`, and at `halflife=10`, `max_dclock=1e5` that is
      `2^-10000 = 0` exactly; the histogram's scale went to zero and every
      later row added `w/0`. `decay` now folds a factor that would take the
      scale under `1e-150` into the weights (multiplying by `s` and `λ`
      separately, since their product is what proved too small) — a zero
      wipes the histogram, which is what the pair moments do on that row.
      (b) **Per-bin Welford.** The sketch's `(Σw, Σwy, Σwy²)` loses the
      variance to cancellation at a modest offset — a target at `1e7` with
      noise of `1e-3` reported none — while every other accumulator in the
      crate is mean-form. Each bin now carries `(w, mean, M2)`; same state,
      one more multiply. The gain is formed as `(d_L/w_L)·(d_L/w_R)/var`,
      because the undecayed weights reach `1e150` before a fold and three of
      them multiplied together overflow.
      (c) **Weighted, point-mass quantile edges.** Plain quantiles of an
      indicator that is zero on 95% of rows all sit on zero and collapse to
      no edge, so the 5% that carry the signal share the zeros' bin — the
      commonest wide-input feature there is, unsplittable. Edges are now
      placed one at a time, each closing a bin of the remaining weight
      divided among the remaining bins, and an edge that would close an
      empty bin moves up to the next value: a point mass fills a bin of its
      own. Weighted by the row weights, not their decay (which says when a
      row arrived, not what the feature looks like). `bin_edges` is refused
      beside `bins`/`bin_rule`/`bin_warm_rows`, and budgeted as a learned
      histogram is.
      (d) **A missing target holds the lag moments.** `marglag.rs` decayed
      them on a row where the target was absent, on top of the ageing `W_t`
      already carries: a hundred null rows at halflife 20 took a `lagcorr`
      of 0.75 to `0.75·2^-5` while `corr` stood still, `n_serial` doubled
      and `t_serial` inflated — invisible at the `halflife=inf` the tests
      ran at. Now it holds, as the pair moments hold (`a = 1, b = 0`).
      (e) **`lagcorr_*` are `ew_cov`'s numbers.** The lists were clamped to
      `[−1, 1]` and the auto terms divided by `var` rather than `sd·sd`;
      neither is what `ew_cov(lags=, stats=["lagcorr"])` reports for the
      same columns. Now `C/(sd_a·sd_b)` in every orientation, unclamped — a
      lagged correlation is not bounded by one in finite samples — and
      `tests/test_marginal_lags.py` holds the two surfaces bit for bit
      through weights, a session change and a capped gap.
      (f) **An empty histogram takes no decay.** `label_delay`'s doubled
      stream begins with a zero-weight prefix; the pair moments carry no
      trace of it, but the histogram picked up a scale, which then moved
      every later weight and read by a rounding — the one last-bit
      difference in the doubled-stream test. An `empty` flag (true at
      construction and after a fold takes every weight to zero) makes
      `decay` a no-op, so `scale == 1` whenever it is set.
      (g) **The closed-group row carries both blocks.** `ClosedRow`'s pair
      block had none of the new columns. It now has `pair_lagcorr_*`,
      `pair_n_serial`, `pair_t_serial`, `pair_phi_*`, `pair_bin_*` and
      `pair_split_*` — `List(List(Float64))` where `marginal()` has a list
      per pair — gated on whether any closing spec asked, under
      `marginal()`'s null rule (NaN only; `±inf` stays, so `pair_t`,
      `pair_t_serial` and `pair_split_gain_t` are `±inf` where the frame
      is), which the block had not been following. `Bank::marginal` builds
      its schema from the spec, so the columns are there before any row and
      for a group never seen. The CSV sidecar refuses every nested column,
      so a closed marginal row stays parquet/ipc/ndjson-only as before.
      (h) Minor: a zero-weight row across a total gap (`λ = 0`, `w = 0`)
      returned from `learn` before touching `wt`/`qt`, so a target's `n_eff`
      outlived a gap the model's did not — it now ages them; `check_edges`
      validates without allocating; docstrings and the design doc rewritten
      to what shipped.

      Tests: `margbins.rs` 6 → 15, `marginal.rs` 17 → 21,
      `state_encoding.rs` covers every optional part; `test_marginal_bins.py`
      13 → 26 functions, `test_marginal_lags.py` 8 → 12,
      `test_closed_groups.py` +3. Every finding above has the test that
      fails without it.

<a id="task-69"></a>
- [x] 69. **The bank's read surface, 2026-09-07** (`docs/ENHANCEMENTS.md`
      E68). A state file was already self-describing and none of it was
      guarded, documented in the README, or reachable from anything but
      Python.

      *Protected.* `ModelBank.specs` was set in `__init__`, so it was an
      instance attribute; `tests/api_surface.txt` is generated by walking
      `dir(po.ModelBank)` and therefore never recorded it. A public name the
      snapshot cannot see is one the README's "a change is a reviewable
      diff" claim does not cover. It is now a read-only property returning a
      copy -- it was writable, and an in-place
      `bank.specs[0]["features"] = [...]` left `coef()` asserting about a
      coefficient count from a spec the bank was not running. A test asserts
      `dir(instance) - dir(class)` is empty, so the next one cannot escape
      the same way.

      *A JSON export*, `to_json()` / `save_json()`. An export, not a second
      state format: `load` reads msgpack and only msgpack, and no second
      format means no second obligation under rule 5.

      **The trap, and why the enforcement is what it is.** `serde_json`
      writes `NaN` and `±inf` as `null` and says nothing about it, where
      msgpack round-trips all three. That is not a corner: `halflife = inf`
      means no decay, is documented, and puts an infinity in every stream's
      `decay`, so the obvious export was silently wrong on ordinary banks.
      Three things follow:

      - A custom `serde_json` `Formatter` **cannot** fix it. `serialize_f64`
        sends a non-finite value to `write_null` before any formatter is
        consulted, so `write_f64` is unreachable for exactly the values it
        would be written for. Tried, and reverted.
      - The fix is per field: `online_core::humanfloat` tags the three as
        `"nan"` / `"inf"` / `"-inf"` -- the spelling `spec::Num` already
        used, so a spec's `halflife` and a state's `decay` read alike --
        keyed on `is_human_readable()`. **msgpack reports `false`, so the
        state file's bytes did not move**: `state_encoding.rs` pins an
        annotated field to the same bytes as a bare `f64` in both encodings,
        because if that ever became untrue every state file would stop
        loading, silently, as a string where a float belongs.
      - Missing a field is caught, not shipped: `save_json_string` re-reads
        its own output into a `BankFile`, re-encodes, and requires the
        msgpack to match byte for byte. That is `O(state)` twice, paid once
        at the end of a run, and it is what found all 18 field annotations,
        across 6 structs -- `Decay`'s two variants, `kalman`'s `halflife` and
        `revert_halflife`, `sgd`'s `clip_gradient`, `holt`'s two halflives,
        `kmeans`'s `far_cut`, and `LastRow`'s ten diagnostic vectors, where
        `NaN` is an ordinary value because an unsupervised model has no
        `pred`. The error quotes the offending lines, since the export runs
        to thousands and a line number alone names nothing.

      `serde_json` moved from a dev-dependency to a workspace dependency;
      nothing new is linked, since `online-py` already depended on it.

      Tests: `tests/test_state_json.py` (25 cases, one per model family with
      a non-finite wherever the spec allows one -- which doubles as the audit
      for the next model), four new cases in `tests/test_bank_ergonomics.py`,
      and `a_human_readable_float_does_not_move_the_msgpack` in
      `crates/online-core/tests/state_encoding.rs`. README: "A state file
      describes itself" and "Reading a state without this library".

<a id="task-70"></a>
- [x] 70. **The leak test measured the ramp, not the plateau, 2026-09-08.**
      `ubuntu-latest` failed the 0.3.0 release candidate on
      `test_output_outliving_its_input`: marks of
      [291868, 291868, 291868, 295840, 301264] KB -- identical to the byte
      three times, then 4.0 MB and 5.3 MB -- a median gap of 33.1 KB/iter
      against a limit of 4.

      Not a leak. Over twelve blocks locally the growth decays to exactly
      zero and stays there (`[1.87, 0.27, 0.13, 0.4, 0.0, 0.27, 0.0 ...]`),
      and `cargo test` passed on the same tree. What failed is the
      measurement: `warmup` was 40 iterations, and the allocator's ramp is
      not over until about 600 -- RSS is 7,344 KB up by iteration 40 and does
      not hold flat to within 8 KB per 120 iterations until 600. Every block
      the test measured was inside the ramp. macOS steps finely enough to
      stay under the threshold; glibc steps in whole arenas, and two
      consecutive steps outvote a median of three gaps, which is the one
      thing task 61's fix was built to survive.

      The repo already knew: `test_plugin_over_groups` passed `warmup=600`
      with a docstring measuring the same ~600-iteration ramp. It was a
      workaround that was right there and silently wrong for the other nine
      callers.

      `assert_plateaus` now **finds** the plateau instead of assuming one --
      it runs blocks until two consecutive ones grow by less than the
      threshold, then measures. That is self-calibrating, so it needs no
      constant tuned per platform, and it cannot be fooled by a leak: a leak
      never plateaus, exhausts `warm_blocks`, and is measured anyway. The
      per-test override is gone.

      Checked both ways rather than just green: three consecutive runs of the
      file pass (6.7s to ~9s), and an injected 8 KB/iter leak is still caught
      (`8.6 KB/iter, gaps [8.7, 8.5, 8.7, 8.5]`).

<a id="task-73"></a>
- [x] 73. **The clock does not have to be a time, and the headline docs did
      not say so, 2026-09-08.** The user's point, and it was right: the clock
      is documented as "a monotone numeric column — seconds, cumulative
      volume, anything", which states the mechanism and leaves the reader to
      derive the consequence. The consequence is worth a headline. Sort a
      frame by one of its features, clock on that feature, and `halflife` is
      a bandwidth in the feature's units — each row fit on the rows before it
      under weight `0.5 ** (Δx / halflife)`, which is local linear regression
      with a one-sided exponential kernel, in one pass, out of `O(k²)` of
      state.

      *Measured before it was written.* An independent oracle — the
      kernel-weighted least squares that definition describes, recomputed
      from scratch at every row, with the mean-form ridge and unpenalised
      intercept the model solves with — agrees with the model to `1.6e-13`
      at three (halflife, ridge) settings, and the null patterns match, so
      `min_periods` gates the same rows. On `sin(x)` at a bandwidth of 0.25
      the fit sits `0.077` from the truth against the best straight line's
      `0.394`; at 1.0 it is `0.290`, most of the way back to the line. Both
      are in `tests/test_ewridge.py`
      (`test_a_feature_as_the_clock_is_a_local_linear_regression`,
      `test_a_bandwidth_in_the_clock_column_follows_a_curve_a_line_cannot`).
      `ew_cov` clocked the same way tracks a correlation that moves with the
      clock column at `corr = 0.91` to the truth — reported here, not
      claimed with a number in the README.

      *What the docs say, and what they admit.* The README's opening list
      gains a paragraph and "How a bank sees a stream" gains "A clock that is
      not time" with a runnable block (so the block test runs it), the
      varying-coefficient reading, one local fit per `group`, and the caveat
      that matters: the kernel is one-sided *because* a row is scored before
      it is learned from, so a curve is followed with a lag, and the
      bandwidth trades that lag against noise. `polars_online.spec`'s
      docstring and `llms.txt` gain a line each.

<a id="task-74"></a>
- [x] 74. **`sgd`'s scaler standardises a row against moments that exclude
      it, and is wrong wherever there are few rows per feature: the start
      of every stream, and every row of a wide fit — found 2026-09-08 by
      task 72's short-history table, and again by its wide table the same
      evening. Fixed 2026-09-08; the decision and the numbers are at the
      end of this entry.** `scale_features` (ENHANCEMENTS E24)
      standardises `x_t` against the running mean and variance from *before*
      row `t`, "so the scaling cannot see the row it is scaling". That is
      stricter than the leakage rule requires and it is unstable: the rule
      forbids the *target*, and the features of the row being predicted are
      known at prediction time; while the moments are two or five rows old,
      a variance estimate can be tiny by chance, the standardised value
      huge, and one LMS step with `eta · |z|² > 2` throws a coefficient far
      enough that the next hundred rows do not bring it back.

      Measured: 500 groups × 200 rows, `k = 20`, each group its own
      coefficients, `learning_rate=0.01`, R² over rows 25–50 of a group is
      **−6.9** (`−99` at 0.03), where `SGDRegressor` with the same step and
      sklearn's scaler order scores 0.45 and `ewridge` 0.97. A pure-numpy
      LMS switched between the two orders reproduces both: moments from
      before the row `−64,309`, moments including the row `0.4328` — and the
      latter is within 0.02 of sklearn's 0.4501 at the same `eta` (the
      rest is where the *prediction* is standardised; see the closing
      paragraph). Over 100,000 rows scored from row 1,000 the two
      orders give the same number (0.9899 both), which is why every earlier
      test and table missed it.

      *The wide table, re-run at a stable learning rate (`0.2 / k`), shows
      the same defect from the other side.* Against `SGDRegressor` row by
      row at the same step, `scale_features=False` predictions correlate
      0.999999 at `k = 20`, 0.999922 at `k = 1,000` and 0.9954 at
      `k = 10,000`; with `scale_features=True` the same three read 0.9999,
      0.978 and **0.521**, and the R² at `k = 10,000` is 0.0209 against
      sklearn's 0.0322. Nothing diverges there — the rate is small enough —
      the standardisation is simply against moments that are never mature,
      because 2,000 rows over 10,000 features is 0.2 rows per feature on
      every row. "Short history" was the special case; **few rows per
      feature** is the condition, and the fix is the same two lines. The
      test list gains the wide case: `scale_features=True` at `k = 1,000`
      within 0.001 R² of `scale_features=False` on the wide stream (today
      0.8221 against 0.8516).

      The fix is sklearn's order — `scaler.partial_fit(x)` then
      `transform(x)` — which bounds a standardised value by about `√n`:
      in `SgdModel::step`, call `sc.update(&raw_z, lam, weight)` *before*
      standardising, not after. `n_eff` is untouched (the scaler does not
      feed it), chunk invariance holds (the order is per row), and the state
      at the end of any row is identical to today's (both orders leave the
      scaler updated with the row), so no `SCHEMA_VERSION` bump — but every
      `scale_features=True` prediction changes, so this is a **behaviour
      change**: golden fixtures regenerate, E24's row and the `sgd`
      docstring say the new order and why, the CHANGELOG entry names it,
      and the README's "Against scikit-learn" and PERFORMANCE §19 drop
      their "until it lands" sentences with the re-measured row. A
      zero-weight row still learns nothing: `update(raw, lam, 0)` applies
      the decay only. Tests: the short-history table as a test (`sgd` at
      `learning_rate=0.01` above 0.4 over rows 25–50 of 200-row groups, and
      the numpy replica agreeing with the model to 1e-12 when the row is
      included); the E24 demonstration (`[0.002, 900]` recovered) kept.

      **Done 2026-09-08, on the user's go — with one change to the recipe
      above.** "Call `sc.update` before standardising" would have broken a
      contract the model already has: `predict(x, d)` is `&self`, carries
      no weight, and must return exactly the `pred` the following `step`
      reports (E31, `model_contract.rs::sgd_predict_is_the_step`, run with
      the scaler on and off over weights 0, 0.5, 1 and 2). Updating the
      scaler with the row's weight before standardising makes the
      standardised row — and so the prediction — depend on the weight,
      which `predict` does not have. The order landed instead is: the row is
      standardised against the moments **with the row admitted at unit
      weight** after the decay — `EwDiag::including(lam)`, a view that reads
      `update`'s recursion slot by slot without touching the accumulator —
      the same `z` serves the prediction and the gradient, and the real
      `update(raw, lam, weight)` stays *after* the gradient step, at the
      row's actual weight. Properties, each with a test: at `w = 1` the
      moments read are to the bit what `update` leaves behind (the
      recursion is written operation for operation, `ewdiag.rs::
      including_reads_what_a_unit_row_would_leave`), so on unit-weight
      streams this *is* sklearn's `partial_fit` then `transform`;
      `predict`/`step` parity is exact by construction (E31 passes
      unchanged); a standardised value is bounded by `sqrt(lam · n_eff)`
      (`z = a·d / sqrt(a·c + a·b·d²) ≤ sqrt(a/b)`, with equality when the
      history has no spread), so the first row of a stream is `z = 0` on
      every feature and only the intercept learns from it
      (`sgd.rs::scaling_admits_the_row_it_scales`, which also checks that
      a row a million deviations out predicts within
      `|b0| + |b1|·sqrt(W)`, where the old order predicted about 1.7e6);
      always well defined, since the total weight is at least the row's own
      1 even on an empty accumulator; and the weight governs *what the row
      teaches*, not where it sits among the rows seen. `kalman` also
      standardises against the moments from before the row and is left
      alone: its gain `P z / (z'P z + σ²)` self-normalises in `|z|` the way
      `pa`'s step does, so a huge `z` moves the state by a bounded amount —
      the defect is specific to a fixed learning rate.

      What did not change: `n_eff`, chunk invariance, the state at the end
      of a row (so no `SCHEMA_VERSION` bump), the cost (`sgd_bench` at
      `k = 10,000`: 20.7 µs a row before, 21.3 after, within the run-to-run
      spread), and a zero-weight row still learns nothing. What did: every
      `scale_features=True` prediction, so `GOLDEN_SGD` regenerated
      (`sgd_squared_golden` and the pipeline goldens run unscaled and stood).

      Measured, `scripts/sklearn_comparison.py`, the day it landed. Short
      history (500 groups × 200 rows, `k = 20`): `sgd` at
      `learning_rate=0.01` reads **0.4328 / 0.7006 / 0.9067** over rows
      25–50 / 50–100 / 100–200 (was −6.9 / −2.4 / 0.1095), against
      `SGDRegressor` with the same constant rate at 0.4501 / 0.7119 /
      0.9111 and `ewridge` at 0.9693 / 0.9860 / 0.9882; at 0.03, **0.7182 /
      0.9213 / 0.9788** against sklearn's best 0.7277 / 0.9242 / 0.9789.
      The numpy replica's 0.4328 is the model's number to four places, and
      switching the replica to predict against the moments from *before*
      the row while learning against the ones including it — sklearn's
      loop, two standardisations a row — gives sklearn's 0.4501 / 0.7119 /
      0.9111 to four places, so that is the whole of the remaining gap.
      Kept as one `z`: a row is then standardised the way every row the
      coefficients were learned from was, the prediction is bounded like
      the step, the scaler is one pass, and the gap is an artefact of an
      unconverged fit (included moments shrink a row's `z` by about `1/n`,
      and a fit still growing into its coefficients scores higher inflated
      by that much) that closes with the history: 0.0044 by rows 100–200
      at 0.01, 0.0001 at 0.03. Wide:
      `scale_features=True` at `k = 1,000` is R² 0.8512 against
      `scale_features=False`'s 0.8516 (within the 0.001 asked for; was
      0.8221), and its predictions correlate **0.99999995** with
      `SGDRegressor`'s row by row (was 0.978); at `k = 10,000` R² 0.0321
      against sklearn's 0.0322 and a correlation of **0.999997** (was
      0.521). Tests added in `tests/test_sgd.py::TestFeatureScaling`: the
      short table (R² > 0.4 over rows 25–50, > 0.85 over 100–200), the
      numpy replica to 1e-12 on every prediction of three groups, and the
      wide case at `k = 200` (scaled within 0.01 R² of unscaled, predictions
      correlating > 0.999); the E24 demonstration (`[0.002, 900]`
      recovered) kept.

<a id="task-72"></a>
- [x] 72. **Against `sklearn.linear_model.SGDRegressor` — analysis done
      2026-09-08, measured and written up the same day.** Written because
      "how does this compare to sklearn" is the first question a reader has
      and the README answered it nowhere.

      **The comparison is not the one the name suggests.** `SGDRegressor`'s
      counterpart here is `po.spec.sgd`, which this repo calls the cheap
      baseline. The primary regression is a different algorithm class:
      `SGDRegressor` is a first-order stochastic optimiser whose answer
      depends on the learning rate, the schedule, feature scaling and row
      order, while `ewridge`/`rls`/`lasso`/`huber`/`quantile` accumulate
      sufficient statistics and solve — with decay off and `ridge = 0`,
      `ewridge` is OLS to `2e-13` of `numpy.linalg.lstsq` in any row order,
      and there is no learning rate because nothing is being descended.

      Where the designs diverge, in the order that matters to a reader:
      forgetting (`partial_fit` weights all history equally and its
      learning-rate decay is about convergence, not recency; every spec here
      takes a `halflife` on a real clock); leakage (sklearn leaves the
      predict-then-fit ordering to the caller, here it is a hard rule with
      tests); grids (`ridge=[...]`, `halflife=[...]` share one accumulator,
      so only the solve repeats, against N estimators over N passes);
      multiple targets sharing one Gram; `group=` against a dict of
      estimators; streaming standardisation against a `StandardScaler` that
      leaks when fitted on a batch; chunk invariance; and a versioned
      cross-OS state file against pickle.

      Where sklearn wins, and the README should say so: `SGDRegressor` is
      `O(k)` in memory where `ewridge` carries a `k×k` Gram — 800 MB at
      k = 10,000, which is the whole reason `marginal` and E51 exist — plus
      the ecosystem (pipelines, `GridSearchCV`, calibration) and far more
      use. `sgd` here is the `O(k)` answer.

      Loss coverage, for the record: `sgd` adds `quantile`, `poisson` and
      `logistic` over `SGDRegressor`'s four, and lacks
      `squared_epsilon_insensitive`; `l1`/elastic net lives in `lasso`,
      solved exactly by coordinate descent rather than by a subgradient.

      **Measured, on one generated stream** (`scripts/sklearn_comparison.py`,
      scikit-learn 1.9.0, which is deliberately *not* a dependency of this
      project — the script says so and exits if it is missing).
      `docs/PERFORMANCE.md` §19 has the four tables; the README gains an
      "Against scikit-learn" section with one of them. Every contender is
      swept over a small grid of its own settings and reported at its best,
      so the comparison is between designs and not between defaults, and
      sklearn is given the setup its own documentation prescribes for
      out-of-core work — an online `StandardScaler`, then `partial_fit`.

      What the numbers said, including where they went against this library:

      - *Accuracy is not the difference.* On a stationary stream at
        `k = 20` everything lands on the same noise ceiling (0.9826 to
        0.9831 out-of-sample R²). The gap opens only under drift, and it is
        about forgetting: `sgd` 0.9906 and `ewridge` at `halflife=500`
        0.9878, against `SGDRegressor`'s 0.9820 batched and 0.9840 row by
        row. A halflife on a clock beats a learning rate tuned in rows —
        which is the claim this repo has been making, now with a number.
      - *The throughput gap is a difference in semantics, and saying so
        matters more than the ratio.* sklearn's fast form is one update per
        1,000-row batch, whose predictions are up to 999 rows stale;
        2.2M rows/s. Ours is one update per row: 5.8M for `sgd`, 570k for
        `ewridge` refitting every row, 5.0M at a 100-row solve cadence. Ask
        sklearn for our guarantee — `partial_fit` per row — and it is 3,300
        rows/s, and that gap is Python's per-row overhead, not the
        algorithm. The README says that in those words rather than quoting
        1,700×.
      - *A grid is nearly free on one side only*: six penalties cost sklearn
        3.66× the work and this 1.40×.
      - *Where sklearn wins, measured rather than conceded in prose.* At
        `k = 10,000`, `ewridge` carries 860 MB of state (`save_bytes`) and
        runs at 51 rows/s against 0.31 MB (a pickle of the fitted estimator
        and its scaler) and 17,221 rows/s. `gram_block_rows=256` buys 5.2×
        of the throughput and none of the memory. `sgd` is the `O(k)`
        answer here — 1.06 MB, 6,206 rows/s — and is *still* slower than
        sklearn at that width, for the same per-row reason. The README says
        that too. (Slower than *batched* sklearn: row by row it is 3×
        faster, and the 0.01 learning rate this row was timed at had both
        libraries diverged — see the evening's addendum below.)

      Two measurement mistakes worth recording, both caught by the numbers
      looking wrong rather than by review. Peak RSS is useless as a memory
      proxy here — the frame and the numpy array dominate it — so the state
      comparison is `save_bytes()` against `pickle.dumps`, which is what a
      deployment actually keeps; and the first wide run had `ewridge`
      solving every row at `k = 1,000` (`solve_every=0.0` with
      `halflife=inf`), which measured a `O(k³)` solve per row at 206 rows/s
      and would have been a strawman of this library, not of sklearn.

      *Corrected the same afternoon, after the user asked whether the
      comparison could improve `ewridge` or whether the feature set buys
      anything.* Re-measured with wider grids, the noise ceiling printed, and
      two new streams; three of the bullets above were wrong in their
      reading and one number was a grid edge:

      - `ewridge`'s halflife sweep stopped at 500 rows, and its best point
        was that edge; at `halflife=100` it scores 0.9907 on the drifting
        stream and ties `sgd` (0.9906), 0.002 from the ceiling of 0.9923.
        **A sweep whose best point is at its edge is a grid, not a result**;
        the script now prints every sweep, best to worst, and the ceiling.
      - "A halflife on a clock beats a learning rate tuned in rows" was the
        wrong reading of the drift table. A constant learning rate *is*
        forgetting (an LMS step `eta` remembers about `1/eta` rows), and on
        evenly spaced rows that is a halflife. `SGDRegressor` row by row at
        `constant, eta0=0.003` scores 0.9899 and `sgd` at
        `learning_rate=0.003` scores 0.9899 — the same recursion, the same
        number. The 0.9840 quoted for sklearn row by row was its
        `invscaling` default, whose decay is about convergence and not
        recency. What separates the drift column is the batched 0.9820,
        and that is staleness. The clock's advantage exists on unevenly
        spaced rows, which this stream cannot show and the write-up
        claimed anyway.
      - Correlated features do not separate the contenders either: a
        one-factor stream at pairwise correlation 0.5 and 0.9 (condition
        number 22 and 187) leaves every row-by-row contender within 0.0007
        of its ceiling over 100,000 rows. First-order convergence slows
        with conditioning, but 100,000 rows is long enough not to notice.
        Heavy-tailed features (`t(3)`, `max |x| = 176`) cost `sgd` its
        best learning rate and nothing at all to `ewridge`; the best
        settings still land within 0.0005 of each other.
      - **Where the Gram measurably wins is a short history.** 500 groups
        of 200 rows, each with its own coefficients: at rows 25–50 of a
        group `ewridge` scores 0.9693 against a ceiling of 0.9896, the best
        `SGDRegressor` setting 0.7277, its default 0.2731; by rows 50–100
        it is 0.9860 against 0.9242. The exact solve is right about `k`
        rows in; a first-order method needs `1/eta` rows per direction. The
        rows/sec column is the group feature: 5.1M for one bank call
        against 3,350 for a Python loop over 500 estimators.
      - **And the same table found `sgd` losing to sklearn's, badly**: R²
        of −6.9 at rows 25–50 at `learning_rate=0.01`, −99 at 0.03. That
        is task 74, a real defect the 100,000-row tables hid by scoring
        from row 1,000.

      What the comparison gives back, then: for `ewridge`, nothing to
      borrow — sklearn's recipe is a scaler and a schedule, `ewridge` has
      the scaler folded into its solve (`standardize`) and no schedule
      because nothing is descended, and its limit is the Gram itself (stored
      full in `EwCov::c`; a triangle would halve 860 MB to 430 and not
      change the wide-row verdict, so it is not done). For `sgd`, the order
      of two lines (task 74), and `average=True` — sklearn's best stationary
      setting, 0.9830 against `sgd`'s 0.9826, and its worst under drift at
      0.817 because the average never forgets — would fit this library as a
      halflife-weighted average of the iterates, if that 0.0005 ever
      matters. `docs/PERFORMANCE.md` §19 is rewritten from the second run;
      the README table carries the corrected numbers and the short-history
      table; the CHANGELOG records the known defect.

      *Two more questions the same evening, both measured at `k = 10,000`.*
      "Is it so much slower because the test emits coefficients every
      row?" No: `coef_every` defaults to 0, so the wide table emitted none,
      and turning it on (10,000 coefficients per row, the output frame
      153 → 302 MB) costs 2.5%; one solve instead of two costs 0.2%;
      `po.spec.sgd` through the same bank with the same 10,005-column
      output runs at 6,217 rows/s against `ewridge`'s 53. The cost is the
      Gram: a rank-1 update moves all 800 MB of it per row, 1.6 GB in
      18.8 ms = 85 GB/s, within 1.5× of a plain in-place numpy add over the
      same 800 MB (123 GB/s on the M4 Pro). Bandwidth, not arithmetic and
      not output. `gram_block_rows` is the lever that exists: 256 → 269
      rows/s, 512 → 306, 1024 → **379** (7.2×, an 82 MB buffer), 2048 →
      376. "Does our `sgd` produce the same thing as `SGDRegressor` at
      6,000 rows/s?" Yes, and 6,000 is the faster side: `SGDRegressor` at
      this library's semantics (row by row) ran at 2,267 rows/s at
      `k = 10,000` and 3,043 at `k = 1,000`, against `sgd`'s 6,944 and
      185,641 (33,271 and 407,925 since task 75), with the correlations
      above; sklearn's 18,980 is its batch of
      1,000, predictions up to 999 rows stale, the narrow table's trade at
      2.7× instead of 1,700×. Checking that exposed that the wide table's
      `learning_rate=0.01` had both libraries diverged (R² −6.8e8 and
      −5.8e26; the timings are unaffected, a diverged LMS costs what a
      converged one does): an LMS step is stable only while
      `eta · |z|² < 2` and a standardised row has `|z|² ≈ k`, so the script
      now runs both at `0.2 / k` and prints R² in the wide table. It also
      put a number on `sgd`'s inner loop — a flat 13–14 ns per feature per
      row from `k = 1,000` to `k = 10,000`, against about 5 ns for sklearn's
      batched Cython — and misread it: "the columnar gather is not it",
      because `to_numpy` of the whole 10,000-column frame ran at 8.4 µs/row
      against `sgd`'s 142. That compared one transpose of the frame with
      a bank that gathered every row *again* from `k` separate columns,
      at a stride; the walk was most of the 142, and one transpose per
      chunk is what the bank does now. Task 75 took the number apart the
      same day: 2.2 ns per feature, 45,000 rows/second at `k = 10,000`,
      2.4× sklearn's batch at the row-by-row semantics. And at
      `k = 1,000` the R² column repeats the short-history lesson: 20 rows
      per feature, `ewridge` at 0.9274 where every first-order contender is
      at 0.85, predicting from coefficients up to 1,000 rows old.

<a id="task-75"></a>
- [x] 75. **`sgd`'s per-feature cost, 2026-09-08: 13–14 ns per feature per
      row to 2.2 at `k = 10,000`, 45,000 rows/second in a bank against
      `SGDRegressor`'s batched 20,007, and every prediction from the state
      as it stands.** Task 72 left the number as "a real target, and an
      independent one"; this is it taken apart, and none of it was the
      arithmetic (`docs/PERFORMANCE.md` §20 has every measurement).

      *The columns.* The bank kept one `Vec<f64>` per feature and a row
      was a walk across all of them at a stride of the chunk's height:
      a cache line and a TLB entry per feature per row, 10,000 pages at
      `k = 10,000`, 78–130 µs per row depending on the stride — the
      "erratic with chunk size" the earlier tables showed. Each chunk is
      now gathered once into a row-major buffer
      (`crates/online-polars/src/rows.rs`, `FeatureRows`): a tiled
      transpose, 128 rows by 64 columns so source and destination lines
      both stay in L1, the layout permutation folded into the gather, in
      parallel above 65,536 cells (`PAR_MIN_CELLS`, alongside the row
      threshold). 3 µs per row at that width. The summary, the label-delay
      buffer and the models all read `features.row(i)`; `Scratch` lost its
      `xs` copy.

      *The step.* Indexed `beta[j][i]`, `zbuf[i]`, `g2[j][i]` through two
      levels of `Vec` per feature, a bounds check on each, the schedule
      chosen by a `match` inside the loop, `(1 + n)^power` raised per
      feature, and three vectors allocated per row on the scaler path.
      Now zipped slices, the rate raised once per row, the intercept's
      constant 1 folded into the dot product as its coefficient (so the
      unscaled path copies nothing), persistent buffers. 2.25 → 0.93 ns per
      feature; `scale_features=True` 14.8 → 2.6. Bit-identical, and proven
      so rather than argued: `crates/online-core/examples/sgd_signature.rs`
      hashes every `step` and `predict` over 48 configurations (intercept ×
      scaler × three schedules × two losses × two penalties, 400 rows with
      zero weights, clock gaps and a missing target) and read the same on
      the old code and the new. `crates/online-core/examples/sgd_bench.rs`
      is the core step's clock.

      *The dot product — the one decision.* A single running sum is a
      chain of dependent additions, three or four cycles each whatever the
      core could do alongside, and 10,000 of them were 7 of the 9.3 µs the
      step still took. It is now `DOT_LANES = 8` interleaved partial sums
      folded pairwise, then the intercept: 0.93 → 0.49 ns per feature.
      The order is fixed by the code, the same on every platform, and not
      0.3.1's, so `sgd` predictions differ from the last release's at
      rounding level: 1e-16 relative on squared loss, 1.5e-11 at worst in
      the signature — Huber at a constant rate, whose clipped gradient
      neither damps a perturbation nor amplifies it, over 400 rows of zero
      weights and gaps. The pipeline goldens (1e-12) pass unchanged, and
      their own docstring says what the tolerance is for: reordered
      arithmetic. `predict` standardises into a thread-local buffer under a
      scaler, since it has `&self` and a fresh vector per row was the cost
      being removed. Taken because the gain is a third of the step and
      neither order is more right than the other; it is its own commit, so
      it can be reverted alone.

      *The accept walk.* `Iterator::all` over the row short-circuits, which
      keeps it scalar; `all_usable` folds over the compare and vectorises.
      1–2 µs per row at `k = 10,000`.

      *Measured, before → after*, one bank, one spec, `learning_rate =
      0.2 / k`, `scale_features=False`: `k = 20` 0.11 → 0.09 µs per row,
      `k = 100` 0.45 → 0.26, `k = 1,000` 4.60 → 2.06, `k = 10,000` 76.5 →
      22.2 (20,000 rows) and 137.9 → 30.4 (2,000 rows); the same stream fed
      as eight chunks at `k = 10,000`, 127.1 → 26.4. `predict` 49.5 → 9.4.

      *What is left at `k = 10,000`, 20 µs per row.* The step 4.9, the
      transpose 3, the accept walk 0.4, and the data summary (task 35)
      about 7 — a Welford update of a 48-byte record per feature per row,
      measured by switching it off. It is now the largest item; making it
      cheaper means laying the records out column-wise so the update
      vectorises, which is a state-layout change (a `SCHEMA_VERSION` bump,
      or a serde mirror of the wire form) for perhaps 4 of the 20 µs. Not
      done. The cheap version — comparing before storing `min` and `max` —
      measured no change and was dropped. The frame hand-off's per-call
      constant is about 8 ms at 10,000 columns, 0.8 µs per column in
      pyo3-polars' `PyDataFrame` extraction (`get_columns`, then one
      `_export` per Series), none of it this library's: it is what
      separates 2,000 rows per call (4 µs per row of it) from 20,000
      (0.4), and the "eight chunks" figure at 2,000 rows from the rest.
      Feed wide frames in tall chunks.

      *Corrected.* PERFORMANCE §19, the task-72 addendum below and the
      README's "Against scikit-learn" all said `sgd`'s inner loop was 13–14
      ns and sklearn's batch faster on a wide row; §19's wide table is
      re-run and the three now say what is measured.

<a id="task-76"></a>
- [x] 76. **`hit_rate` reads 1.0 for every binary fit, and classification has
      no metric of its own — found 2026-09-08, asking whether an SGD
      classifier was worth adding. Fixed the same day; the decision and the
      numbers are at the end of this entry.** A hit is scored as
      `pred.signum() == y.signum()` with `y == 0` rows dropped
      (`SlotMetrics::update` in `crates/online-core/src/stats.rs`, and the
      same expression in `python/polars_online/eval.py`). A logistic fit
      predicts a probability in (0, 1) against a 0/1 target, so every zero
      row is dropped and every surviving row is a hit: the field is **1.0
      whatever the fit does**. Measured, 20,000 rows, `k = 2`, `y` drawn from
      `sigmoid(1.2·x0 − 0.8·x1)`:

    | fit | `hit_rate` | `r2` | `ic` | accuracy at 0.5 |
    |---|---:|---:|---:|---:|
    | `sgd`, `loss="logistic"` | 1.0000 | 0.2719 | 0.5216 | 0.7268 |
    | `ftrl`, `loss="logistic"` | 1.0000 | 0.2784 | 0.5278 | 0.7304 |
    | `sgd` logistic on pure-noise features | **1.0000** | −0.0159 | 0.0189 | — |

      A model that knows nothing reports a perfect hit rate, in the streaming
      metric and in `po.eval.metrics` alike. The other two survive the
      translation and nothing says so: on a 0/1 target `r2` is the Brier
      skill score against the running base rate and `ic` is a point-biserial
      correlation. There is no log loss anywhere, and `ew_class` refuses
      `emit_metrics` outright ("it has no predictions, so no residuals"), so
      the multiclass model has no metric at all.

      The fix, in three parts:

      1. *`hit_rate` on a binary fit is accuracy at a 0.5 threshold.* Gate it
         on the **declared** loss — `sgd`'s and `ftrl`'s `logistic` — not on
         sniffing the target's values, which would make a metric's meaning
         depend on the data a chunk happened to carry. A regression fit keeps
         the sign agreement it has now. Alternative if a threshold is judged
         arbitrary: refuse the field for a logistic loss the way `ew_class`
         refuses the lot, which is honest but leaves classification with two
         metrics rather than three.
      2. *Log loss, and the constraint that shapes it.* `−[y ln p + (1−y)
         ln(1−p)]`, exponentially weighted like the rest. The streaming
         version would put a `ln` result **into the state**, which §11a's B4
         rule forbids: a libm result may be compared or reported, but
         persisting one costs cross-platform reproducibility (that is what
         broke a frozen fixture on `ubuntu-latest` alone). So either it lives
         only in `po.eval`, over the collected frame, or the accumulator is
         excluded from the frozen-state fixtures with the reason written
         down. Decide before building.
      3. *Document what the metrics mean on a 0/1 target* — `r2` as Brier
         skill, `ic` as point-biserial — in `emit_metrics`' docstring, the
         README's output table and `po.eval.metrics`.

      **Not the classifier E28 declined.** No new loss, no new model, no
      margin: this is the plumbing that makes the two logistic fits already
      shipped (`sgd(loss="logistic")`, `ftrl`) reportable. E28 (perceptron
      and PA-classifier hinges) and E39's discriminative multiclass stay
      declined until a caller wants a margin instead of a probability.

      Ships after 0.4.0, which is tagged with the defect in it. Whether
      `hit_rate` changing counts as "a change to the numbers a model
      returns" under 0.4.0's own widened rule is the user's call; the reading
      here is that it is a diagnostic rather than a prediction, and one that
      is currently a constant, so a patch — but it should be the first line
      of the release note either way.

      Tests: the pure-noise fit above as a regression test (a model that
      knows nothing must not report a perfect hit rate); accuracy against a
      numpy replica on a fit that does know something; the regression path's
      `hit_rate` unchanged to the bit; and `po.eval.metrics` agreeing with
      the streaming metric on the same stream, which is the invariant E22
      already claims.

      **Done 2026-09-08, on the user's go.** Built as planned, with the two
      questions the entry above left open resolved:

      - *A threshold, not a refusal.* `SlotMetrics::update` (and
        `_metric_exprs` in `eval.py`) take a `binary` flag: sign agreement
        with `y == 0` excluded when `false`, accuracy at 0.5 with every row
        scoring when `true`. It is set from the model's **declared** loss —
        `sgd`'s or `ftrl`'s `logistic` — computed once per instance in
        `run_instance` (`crates/online-polars/src/stream.rs`), not sniffed
        from a row's value; `po.eval`'s functions take `binary` as an
        explicit keyword, the same fact the caller already knows from
        choosing the loss. No new spec field: a `logistic` fit's `hit_rate`
        simply means something different now, the way task 74 changed what
        `scale_features` computes without adding a knob for it.
      - *Log loss lives only in `po.eval`.* `metrics(..., binary=True)` and
        `rolling_metrics` add a `log_loss` column
        (`-(y·ln p + (1−y)·ln(1−p))`, clipped, mean over the window);
        `sums`/`from_sums` (E49) do not carry it — a straightforward
        addition if a chunked reduction ever needs it, not built because
        nothing has asked. Confirms the B4 call in the entry above: nothing
        here puts a `ln` result into a model's persisted state, so no
        cross-platform risk is taken on.

      `merge_sums` needed no change — it sums whatever `hits`/`signed` hold
      without caring how they were computed, so mixing `binary` settings
      across chunks is a caller error the docstring now names, not a new
      failure mode to guard against. No `SCHEMA_VERSION` bump: `SlotMetrics`'s
      fields are unchanged in shape, only in what a caller's own choice of
      loss makes them count — the same kind of change task 74 made to `sgd`.

      Measured, the pure-noise fixture from the top of this entry (20,000
      rows, `k = 2`, `sgd(loss="logistic")`): `hit_rate` (binary) 0.5042,
      where it read 1.0 before. On the informative fixture: `hit_rate`
      (binary) 0.72677, matching a numpy replica of the same threshold test
      to 1e-12, and `po.eval.metrics(binary=True)` agreeing with the
      streaming `hit_rate_y0` field to 0.72675 (one row's difference in
      20,000 — `emit_metrics` reads before the last row, `po.eval` scores it
      too). `sums`/`from_sums` with `binary=True` reproduce `metrics`'s
      `hit_rate`, `r2` and `ic` (0.72677, 0.27193, 0.52167) exactly.

      Tests: `crates/online-core/src/stats.rs` —
      `a_probability_that_knows_nothing_does_not_score_a_perfect_hit_rate`,
      `binary_hit_rate_is_accuracy_at_one_half` (against a hand-rolled
      replica), `a_zero_label_scores_under_binary_where_it_is_excluded_under_sign`;
      every existing `SlotMetrics` test kept its `binary = false` call
      unchanged, so the regression path is untouched to the bit.
      `tests/test_eval.py::TestBinary` covers the same four cases in Python,
      plus `log_loss` against a numpy replica and the `sums`/`from_sums`
      round trip. Every test file that already used `emit_metrics` or
      `hit_rate` (`test_diagnostics.py`'s E22 test among them) is regression
      and passes unchanged, since none of them named `binary=True`.

<a id="task-85"></a>
- [x] 85. **The expression plugin removed, 2026-09-17.** It was O(data) by
      polars' own rules and warned on every use since 2026-09-03 (task 19);
      keeping a surface whose whole behaviour contradicts the library's point
      cost more than it bought. Deleted: `crates/online-py/src/expr.rs`,
      `python/polars_online/_expr.py`, `tests/test_expr.py`, the `online`
      namespace, `po.online` and `InMemoryExpressionWarning`. **The build
      consequence is the real prize and was measured, not assumed**:
      `pyo3-polars/derive` turned on `polars-plan/python`, which only
      `polars-lazy/python` propagates to `polars-mem-engine`, so `derive` was
      the reason `lazy` was in the extension's features at all. Both are gone
      and `cargo check -p online-py` is clean. ~30 expression tests went too:
      both paths call the same `Bank::fit_predict`, so they tested packing, not
      arithmetic (survey in the session log: every model file keeps 3-60 bank
      tests, five keep numpy oracles). Four tests were *restored* after a bulk
      deletion over-reached -- two `test_every_surface_runs_it`, the
      two-specs-stay-distinct hardening case, and two typed-dict runtime cases
      -- because their subject survives the plugin.

<a id="task-84"></a>
- [x] 84. **The bank takes a plan, and learns without keeping the output,
      2026-09-17.** `fit_predict_batches` accepts a `LazyFrame` and builds the
      chunk iterator itself (`chunk_rows`, defaulting to the shared
      `default_chunk_rows`), so the caller no longer writes
      `lf.collect_batches()`; a `DataFrame` is one chunk and an iterator is
      passed through. `fit` is the learn-only form -- the same generator with
      its frames dropped -- which gives Python the runner's `--no-output` shape
      now that the runner is the command line's alone (task 83). Measured
      claim deliberately *not* made: this is not faster. The runner's own
      discard path drops each frame after assembly (`runner.rs:618`), and so
      does this; what it saves is the write and the assembled result.
      `_check_frame`'s advice changed with it, since telling a caller to reach
      for `collect_batches` is wrong once the method takes the plan.

<a id="task-83"></a>
- [x] 83. **The runner leaves Python; the `online` command line keeps it,
      2026-09-17.** `polars_online.run` built its chunk iterator with py-polars
      and handed the frames to `run_config_on`, so it duplicated what
      `ModelBank` and `lf.online.fit_predict` already do while dragging the
      lazy engine, the parquet/CSV/IPC readers and the sinks into the
      extension module, which no Python path reached (`Input::Lazy` has one
      constructor, `runner.rs:430`, and only the CLI reaches it). Removed:
      `_runner.py`, `run_config_frames`, `PyFrames`, `PyFailure`, `formats()`.
      Tests moved rather than dropped wherever the path survives -- the
      hardening trio and the no-output and predict clusters to the command
      line, the sidecar cases to the query path, `test_sink_equals_run` to a
      sink-versus-collect comparison -- and deleted only where the bank half
      sat in the same function. `progress` has no command-line equivalent and
      is removed, not moved.

<a id="task-82"></a>
- [x] 82. **N9: `quantile` did not settle on the quantile regression -- found
      2026-09-15 writing the code review's T-S4, fixed the same day on the
      user's go-ahead.** IRLS on each row's prior residual, with its weights
      frozen as the rows arrived, kept the fit off `statsmodels`' `QuantReg`
      at every length measured: 0.164 in the intercept at the median of a
      skewed noise after 20 000 rows and 0.477 at the 0.9 quantile, where
      `QuantReg`'s own standard errors are 0.007 and 0.021. It takes one
      Newton step on the kernel-smoothed check loss now (`robust.rs`,
      `row_update`), which lands inside those standard errors, and
      `quantile_eps` is the band's half-width, default 0.2. The measurements
      and the derivation are in `docs/REVIEW-2026-09-12-PROGRESS.md`, N9.

<a id="task-81"></a>
- [x] 81. **`target_gaps`: which rows a target's fit is read from -- the code
      review's N3, decided 2026-09-14.** With a target null on some rows,
      `ewridge` and `lasso` read the Gram over every row and the target's
      cross-moments over its own rows, so a slope moved with the target's
      level: `(m_j − m)·ȳ_j / Var(x)` on top of the fit, with `m_j` a
      feature's mean over the target's rows and `m` its mean over all of
      them. The user asked for every option behind one parameter, then, of
      that behaviour: "what value does it add if we implement the other
      options?" None: it costs what `"pairwise"` costs and adds the level
      term, so it goes. Two values, on `ewridge` and `lasso` -- `robust` and
      `kalman` already fit each target on its own rows, and `rls` learns a
      row only when every target has one:

      - `"own_rows"`, the default: each target is fitted on exactly the rows
        it is present on, the fit of the frame with its nulls dropped.
        Targets present on the same rows share one Gram. A target missing
        from a row that another target of its Gram is present on takes a
        copy of that Gram as it stood before the row, and keeps its own from
        then on. A single target never copies (its Gram skips its null
        rows), and a bank of targets costs one `k×k` Gram per pattern of
        missing rows.
      - `"pairwise"`: the Gram over every row, each target's cross-moments
        centred at its own rows' means -- pandas' pairwise-complete
        covariance. One Gram whatever the gaps. Exact when the gaps have
        nothing to do with the features; where they do (a target present
        only on trade rows between market-data rows), each slope is scaled
        by the ratio of the feature's variance over the target's rows to its
        variance over all rows.

      Both solve `(C + ridge·I)·β = c_j` with `β_0 = ȳ_j − m_j·β`: one
      formula, read from a different Gram. `n_eff` counts every row in both,
      as rule 8 has it, and `min_periods` gates on it as before (S2 stays
      open). `lasso` takes `ewridge`'s centred cross-moments, which it needs
      for `m_j`: that is N2. `bank.gram()` returns one entry per Gram, each
      naming its targets, and adds `means_by_target` -- the column means
      over each target's rows -- which `po.gram.solve` needs to reproduce
      `"pairwise"`.

      State: `SCHEMA_VERSION` and `MIN_SCHEMA_VERSION` are both 8. The user,
      2026-09-14: "Do not worry about state saved before the next version
      release, we are pre 1.0 and we can change things now". So the
      schema-6 loaders the review round wrote go, with their frozen
      fixtures. Changes the numbers of every `ewridge` and `lasso` stream
      with a null target: a minor, with a CHANGELOG line.

      **Done 2026-09-14.** `crates/online-core/src/gaps.rs` holds what both
      models share: `TargetGaps`, the centred cross-moments (`Cross`, with
      the weight over every row that is `n_eff`), and `Grams`, the Grams
      and which one each target reads. Under `own_rows` a Gram learns the
      rows one of its targets has, ages over the rest (`EwCov::skip`, which
      holds a blocked Gram's skipped row as a zero-weight one rather than
      flushing), and splits where its targets part; a window truncates a
      Gram against the one its targets read at the boundary, and a blend
      mixes a Gram with the twin's of the same targets. The intercept reads
      the Gram's own mean under `own_rows` -- so a row the target misses
      leaves its fit where it was to the bit -- and the Gram's mean plus the
      target's offset under `pairwise`. Without gaps `ewridge` is the same
      to the bit as before, checked against the previous build's goldens.

      Before the fix, the new library tests failed as N3 predicts: the fit
      of a target present where `x0 > -0.5` was off by 1.17 at level 0 and
      by 49.2 at level 50, and a second gappy target of three by 6.18.
      Library tests, `tests/test_second_opinion.py::TestATargetWithGaps`:
      `own_rows` against `numpy.linalg.lstsq` and statsmodels' `WLS` on the
      target's rows (with decay, at a level, plain and standardized, three
      targets with three patterns of gaps), statsmodels' ridge
      (`fit_regularized`, `L1_wt=0`) with a penalty, and statsmodels'
      elastic net against `lasso` at a non-zero penalty, a noise feature
      zeroed by both; `pairwise` against pandas' `DataFrame.cov` and
      `numpy.cov` with weights; a `window` with gaps against both. In
      `test_gram_module.py`, `coef_stats` on a gappy target's Gram is
      statsmodels' OLS on its rows (coefficients, standard errors, R²).
      Rust: each target of a three-target bank is, to the bit, the Gram of
      a model of that target alone, blocked or not, and through a save; a
      blend is each target's own blend; RLS agrees with `ridge_decay` on a
      stream with gaps under `own_rows` and not under `pairwise`; a level
      moves no slope under either reading; `lasso` at `1e8` (N2).

      On the way, N5: `po.gram.solve(standardize=True)` and
      `po.gram.lasso_path` without an intercept scaled the centred
      co-moments against the raw cross-moments, the hybrid the models lost
      in the review's C8; both read raw moments now, with a test each.
      Re-frozen: the core goldens for `ew_ridge`, `ew_ridge_std` and
      `lasso` (a null target at row 31), and in the golden pipeline the
      `ridge`, `lasso` and `seqtest_compare` keys (a null every 29th row);
      no `n_eff` moved. A closed row carries one Gram, and its `n_eff` is
      that Gram's weight, as its `n_kish` was.

      A second read, at the user's request, found two things the tests had
      not. A zero-weight row with its targets parted split their Gram,
      although the row learns nothing and the copy was the Gram itself, kept
      twice from then on; it splits nothing now while there is weight to
      age, since taking the row at weight 0 and skipping it age the Gram by
      the same steps (`a_zero_weight_row_splits_no_gram`). And a target not
      seen yet, alone in a Gram with no weight, cost one jittered solve --
      counted in `solve_failures` -- on every solve under `ridge = 0`; that
      solve is skipped, the coefficients the zeros it gave
      (`a_target_not_seen_yet_costs_no_solve_failure`), and with a penalty
      the solve runs, so a target not seen yet still sits at its
      `coef_prior`. Tests added for paths nothing covered: a closed group's
      row per Gram, each solving to its own `coef`, and merged shards'
      Grams against the fit of their union by `numpy` and pandas.

<a id="task-80"></a>
- [x] 80. **The code review of 2026-09-12, worked through, begun
      2026-09-13.** `docs/REVIEW-2026-09-12.md` (and its pass-10 supplement)
      is the reviewer's; `docs/REVIEW-2026-09-12-PROGRESS.md` is the status of
      every finding in it, and the place to look. The rule for the round, the
      user's: fixes whose test is an independent library (`numpy`, `scipy`,
      `river`, and from 2026-09-13 `statsmodels`, `filterpy` and
      `bayesian-changepoint-detection`; `tests/test_second_opinion.py`)
      first, everything else kept for later with its reason. Commits carry
      the finding IDs. Tasks 78 and 79, parked on `design/task-78` while it
      ran, were merged back on 2026-09-13; 79 was this round's C5.

      **Continued 2026-09-14, after task 81: the findings no library can
      check.** The user: "We want to be careful fixing the items that
      cannot be verified by a third party library. As always a test first
      methodology is standard but we want to be sure that we double check
      each finding with a fresh eye since there is no other check, if we
      write the test and it passes where the reviewer assumes it would not,
      document the issue in the progress doc and raise it to me." Then: fix
      everything except D1 (hard rule 5, on not breaking file
      compatibility). So each remaining finding is re-derived from the code
      before its test is written; the test runs on the build before the
      fix and must fail where the finding says; one that passes is
      recorded and raised, not fixed. The findings that need a decision go
      to the user first.

      **The user's decisions, 2026-09-15**, each premise first confirmed on
      the build: S29/S30 -- `holt` becomes a weighted mean, level and trend
      `(λW·p + w·y)/(λW + w)`, so a weight means what it does elsewhere and
      `halflife = inf` is the cumulative fit (today the level at weight 0.5
      equals weight 1 to the last digit, and `inf` freezes the level at the
      first row). S2 -- each target's `min_periods` is checked against its
      own weight; the emitted `n_eff` stays the shared weight. S1 -- a
      windowed spec's `sigma` and `resid_z` are windowed. S3 --
      `max_dclock` caps the total delta a row sees after skipped rows (ten
      rows 100 apart handed the next one about 660). S23 -- `emit_averaged`
      compares the slots' errors as ratios, `exp(−eta·(σ²/σ²_best − 1))`.
      S27 -- `inf` is accepted where it means something and refused in both
      layers elsewhere. S31 -- a `strict_binary` target other than 0 or 1
      refuses the chunk, naming the row. C24 -- river's `FTRLProximal` has no
      forgetting (its sums only grow), so the halflife is ours: it stays,
      repaired with a decayed proximal sum, and at `halflife = inf` the
      model is river's to the bit, as T-R1 holds it.

      **Batch 1, 2026-09-15**, the core findings: C6, C7, S4, S6, S11, S13,
      S16, S32 and D2-D6 fixed, with N6, found on the way (`σ²`'s weight
      did not age on a row with a target and no prediction); S10 is not a
      defect. The progress doc has each, and what was raised: S10's
      premise, S13's test passing on the old build, S6 and D6 worse than
      written, C6's test unable to show it.

      **Batch 2, 2026-09-15**, the spec layer: S5, S7, S8, S21, S22, S24,
      S25, S26, C18, C20, D7, D8 and V23. `Spec::check` -- fill, validate,
      build -- is the one door every entry point uses (S25). Every new test
      failed on the old build. One call taken: `kalman`'s `coef_halflife`
      beside `q` is documented, not refused, since the docs already say `q`
      overrides it and specs use the pair.

      **P4 and P5, the user's decisions, 2026-09-15.** P4: the window's
      snapshot ring gets both bounds the user asked for, chosen by one new
      parameter that names the action and the budget, `window_budget =
      {"thin": MiB}` or `{"refuse": MiB}`. Past the budget the ring either
      refuses at runtime, naming the size and `window_every`, or thins: every
      other snapshot dropped and the spacing doubled, so the boundary grows
      coarser and never keeps an older row. P5: `fit_predict_batches` takes a
      `closed_groups` path and drains after each batch, and the docs say a
      `ModelBank`'s queue is bounded only by draining it; the runner writes
      its closed-groups file chunk by chunk.

      **Batch 3a, 2026-09-15**: C4, C19, P1, P2, P3, S12, V7 and V25 fixed;
      V5, V11 and V24 closed; V25 is not a defect for `kmeans`, which the
      review presumed. Batch 3b is P4 and P5, as decided above.

      **Batch 3b, 2026-09-15**: P4 and P5 fixed, as decided. Two calls of
      mine, raised: a window with no `window_budget` refuses past 256 MiB per
      ring; and a bank refused for its budget refuses every later
      `fit_predict`, `predict` and `save`, since the budget is found as the
      rows go in and the bank has by then learned part of the chunk (every
      other refusal is checked before a stream is touched).
      `fit_predict_batches(closed_groups=)` writes what it drained however
      the chunks stop, where the plan's source writes only at the end.

      **Batch 4a, 2026-09-15**: S29, S30, S31 and C24 fixed, as decided,
      with D9's and D10's `holt` and `ftrl` items; schema 9. Raised: every
      `holt` stream's numbers move (the weighted means' gains start at 1),
      `trend_halflife = inf` is the whole history's drift with no spelling
      left for a plain level, and C24's repair stops short of the review's
      numbers, since the penalties stay constants on the sums' scale --
      which is what keeps the undecayed model river's.

      **Batch 4b, 2026-09-15**: S2, S3 and S23 fixed, as decided, with
      D9's last items. S2 adds `OnlineModel::target_n_eff_into`, which the
      models that keep a weight per target override (docs/EXTENDING.md);
      the emitted `n_eff` is unchanged. The suite's oracles now cap the
      folded delta and gate a target's output on its own weight. Batch 4c
      is S1, S27, N4 and the rest of D10.

      **Batch 4c, 2026-09-15**: S27 fixed, as decided, and D10's rest. Seven
      parameters take `inf` where it names a limit, nine say `finite`, and
      `ridge` and `q` leave the Python table. Raised: the review's
      `ridge`/`q` half was the table's alone, since the builders validate
      through Rust. Two new observations fixed with it: N7 (`sgd`'s NaN
      `huber_delta` reached `f64::clamp`) and N8 (`marginal` refused
      `min_periods = inf`). S1 is batch 4d and N4 batch 4e.

      **Batch 4d, 2026-09-15**: S1 fixed, as decided: under a window the
      stream cuts its per-slot spread with a ring of its own, so `sigma`,
      `resid_z`, drift's scale, the conformal band and the slot ranking
      describe the rows the fit does. The ring rides on schema 9 and counts
      in `window_budget`. N4 is batch 4e.

      **Batch 4e, 2026-09-15**: N4 fixed: `bank.gram()`, a closed row and
      `merge` carry the centred cross-moments the model holds, and
      `po.gram.solve`, `lasso_path` and `coef_stats` read them. With it
      every finding of the review is fixed but D1, which the user excluded;
      what was raised is in the progress file.

      **Batch 5, 2026-09-15**: the review's second opinions that nothing had
      written. T-S18: `pa` is river's `PARegressor` to `1e-15`, once river
      is given the loss it documents (river 0.26.1's scores a target of 2
      as 3). T-S9: `ew_cov`, the target moments and `marginal`'s pair are
      pandas' `ewm`, absolute at `1e8`. T-S4's quantile half, which
      `quantile` does not meet: N9, task 82; `sgd`'s quantile loss does.

      **Batch 7, 2026-09-15**: the second review's findings
      (`docs/REVIEW-2026-09-15.md`): the quantile fit's warm-up and the
      per-target gate read the rows present, the band has a floor in the
      effective sample and takes least-squares rows while it holds under a
      row per coefficient (what rebuilds a fit a row at the input bound
      left behind), and a failed run publishes the closed groups it
      drained. Schema 10.

<a id="task-79"></a>
- [x] 79. **`label_delay` ignores a clock event on a skipped row — found
      2026-09-11, checking task 78's parity; in released 0.5.1.** A reset
      that lands on a row the spec skips (a null feature, an unusable
      weight) restarts the model but leaves the `label_delay` buffer full,
      so rows from before the reset are later learned into the model that
      was meant to start clean. Measured: `ewridge`, no decay,
      `label_delay=3`, `session_gap="reset"`, the first row of the new
      session carrying a null feature. `n_eff` per row:

    | row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | reset row skipped | 0 | 0 | 0 | 1 | 2 | — | **1** | **2** | **3** | 4 | 5 | **6** |
    | reset row accepted (null target) | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 1 | 2 | 3 | 4 |

      The fresh model has learned a row by row 6 and ends two rows ahead:
      two rows of the old session leaked across the reset.

      *Cause.* `run_instance` acts on a row's `reset`, `session_changed`
      and `capped` before its `accept` test — deliberately, the comment
      says, "because a skipped row's gap breaks adjacency just as much".
      `apply_label_delay` does the opposite: its `!plan.accept` branch
      comes first and skips the event handling, so the buffer never sees a
      skipped row's reset (should drop), session change or capped gap
      (should release). The reset is measured; the other two go through the
      same branch and are expected to fail the same way — releasing late,
      across the break, into lag rings that were just cleared for it, the
      exact case `docs/REVIEW-E54-E64.md` L2 fixed for accepted rows.

      *Fix.* In `apply_label_delay`, handle the event for every row — a
      reset clears the buffer, a session change or capped gap releases it —
      and push only accepted rows. Tests: the table above as a regression
      test (`n_eff` equal in both rows from row 5 on, bar row 5's own), and
      the same shape for a session change and a capped gap on a skipped row,
      with a lag feature to show the ring is not refilled across the break.
      Changes numbers only for a stream with `label_delay` and an event on a
      skipped row; a patch, with a CHANGELOG line. Before 78b (task 106,
      discarded by the user on 2026-09-25), which would have extended the
      same release rule to a backward clock.
      **Done 2026-09-13** as the code review's C5, fixed with C21 in
      `cb6c57c` (task 80): a reset on a skipped row now clears the buffer,
      and a session change or capped gap on one releases it.

<a id="task-78"></a>
- [x] 78. **Windowed EWMAs in both directions, as columns -- built
      2026-09-30 (`34af36f`).** Its API is replaced by task 143's window
      expressions with task 144's names, and its targets by task 104; the
      core below carries over.

      #### What it computes

      For row *t* at policy time `τ_t`, over the rows *j* in its window:
      `y_t = Σ w_j λ^|τ_j − τ_a| v_j / Σ w_j λ^|τ_j − τ_a|`, `λ = 2^(−1 /
      halflife)`. `po.window.ewm` takes the rows less than `horizon` older,
      `(τ_t − h, τ_t]`, or every earlier row with no horizon;
      `po.window.lookahead_rewm` the rows after *t* less than `horizon`
      later, `(τ_t, τ_t + h)`. `po.stream.with_windows(frame, windows, ...)`,
      also `lf.online.with_windows` and `df.online.with_windows`, adds their
      columns in one pass; the input passes through.

      #### How

      - **The core** (`crates/online-polars/src/windows.rs`): segment sums
        anchored at the near end, combined associatively, in a two-stack
        queue -- O(1) a row on average, exact, no subtraction. Only rows with
        a usable value and a usable, non-zero weight enter a queue; a weight
        below zero is refused.
      - **The clock is the model's**: each group steps its own `ClockState`,
        so the horizon is measured on the policy clock. A gap past
        `max_dclock` or a session change ends the windows open across it as
        partial; a reset discards them. A group's policy time restarts at
        each such event. The rows also step one shared clock in input order
        across groups, under the same policy: a step back the policy
        refuses is refused naming the row, a reset there discards every
        group's windows, and a gap there ends them; with a group column a
        session is each group's (review R2, W3), without one the stream is
        the one group. A group silent for more than `max_dclock` of the stream's time
        has its windows cut then, which bounds how long it holds the output;
        a row-count clock has no such bound.
      - **Options**: `weight`; `split=(column, [values])` with `total` and
        `unlisted`; `partial` (`"keep"` backward, `"null"` forward, or
        `"drop"`); `complete`, a Boolean column per description and horizon;
        `same_clock` (stream order at one clock value); name templates over
        `{column}`, `{halflife}`, `{horizon}`, `{split}`, collisions refused.
        `like=spec` takes a spec's clock policy and nulls a look-ahead on
        rows the spec skips; clock keywords beside it are refused.
      - **Rows out** in input order, each once every window over it has
        closed; with only backward windows, as they arrive. The rows a
        look-ahead holds are the input's own Arrow chunks, sliced, never
        copied. A row unresolved at the end of the input is null unless
        `save_state` keeps it. Under `head(n)` the input is read up to the
        row that resolved the *n*-th output row. A projection reaches the
        input only when no state is loaded or saved. The input plan gets the
        bank's row-order and spent-stream checks.
      - **The state** is versioned msgpack: the call, the core, and the held
        rows as Arrow IPC. It resumes only the call that saved it, on the
        same kind of clock, over the same columns. The clock policy is
        stored in a tagged form, since `ClockCfg`'s own derive writes
        `session_gap = "reset"` as a unit that msgpack reads back as no gap.
        The clock-policy checks are shared with `Spec` (`clock_cfg_of`).
      - **As a target today**: the column fitted with `label_delay` at least
        the horizon, on the same clock, and a horizon no longer than
        `max_dclock` (the README); each prediction is then emitted one
        horizon late. Task 104 makes it native.

      #### Measured (docs/PERFORMANCE.md §32)

      9.5 to 9.8 million rows a second at the memory of reading and writing
      the file, flat in the window's length; Polars' `rolling` recipe for
      the same column 11 to 64 times slower, at 4.9 to 18.3 GB, agreeing to
      1.7e-13 where both give a value. As a target under `label_delay` it
      adds at most 0.2 s to the model's own time.

      #### Tests

      The core against a brute-force loop over every clock event, both
      clock kinds and a row-count clock (480 streams), nine bugs planted one
      at a time each caught; each direction against the definition and
      against Polars' `rolling`; interleaved trades and quotes with three
      VWAPs equal to one split window, to the bit; every event; chunk
      invariance at 1, 7, 64 and 1000 rows; a save and load at every row; a
      slice, a filter and a projection; names and collisions; queues flat
      across quote density; held rows sharing the input's buffers; windows
      feeding a model in one query equal to two steps
      (`tests/test_windows.py`, `windows.rs`, `windows_frame.rs`).

<a id="task-77"></a>
- [x] 77. **A plan with nothing to write runs once where a query uses it
      twice, 2026-09-10.** The IO source declares `is_pure=True` to
      `register_io_source` exactly when the run has no writes — no
      `save_state`, no `closed_groups` sidecar. Polars then shares one
      execution where a single query uses the same plan twice (a self-join,
      `pl.concat([plan, plan])`) instead of running the source twice,
      concurrently, as R2 of `docs/STATE-WORKFLOW.md` measured before this.
      That covers every `predict` and every fit that keeps its state in
      memory.

      **The condition is the point of the entry.** The first attempt
      declared purity unconditionally, on the strength of a measurement:
      with `save_state` set and `pl.concat([plan, plan])`, the query yields
      8,000 rows and the state file it leaves is **byte-identical to the
      file a single ordinary run leaves**. The whole suite passed on
      py-polars 1.44.1 and 2.0.0rc1 alike, 2,443 tests. So it worked — and
      it was still the wrong declaration, because it answered "does dedup
      break anything today" when the question was "is this source pure".

      *What the flag means, read rather than inferred.* In polars' source,
      `is_pure` is the only thing that can make two `PythonOptions` compare
      **equal** (`crates/polars-plan/src/plans/ir/equality.rs`: the
      comparison opens `(*l_is_pure && *r_is_pure) && ...`), and IR node
      equality is what CSE and plan dedup are keyed on. So the flag is not a
      request to share a run; it is an assertion that two occurrences are
      the same node, and dropping one of them drops its **effects** as well
      as its work. A source that writes a file has effects. The rows here
      are pure by construction — `make_bank()` is called inside `source`, so
      every execution starts from the same bytes (R3) and feeds them the
      same rows in the same order, which is exactly R2 — but identical bytes
      are not no bytes, and R2's idempotence is a weaker property than
      purity, not the same one.

      *We cannot dedupe the impure case ourselves, and it does not need it.*
      Polars hands the source callable
      `(with_columns, predicate, n_rows, batch_size)` and nothing that
      identifies an execution, so a second concurrent run within one query
      is indistinguishable from a later `collect()`, which must re-run (R2).
      Overlap in time is the only signal left, and sharing rows between two
      consumers pulling at their own rates means buffering the whole stream
      — the memory bound this library exists to hold. Nor is there anything
      to fix: the two runs write identical bytes, atomically, and
      `atomic.rs`' counter already makes the pair safe. The cost is
      duplicated work in one rare plan shape, which is a price, not a bug.

      *An objection that turned out to be about something else.* The first
      reservation raised against the flag was that purity licenses polars to
      skip the source entirely, so a query asking for no rows would silently
      not write. Measured: `head(0)`, `limit(0)` and `clear()` run the
      source **0 times with `is_pure=False` and 0 times with
      `is_pure=True`**. That is polars pruning a source it has no use for.
      It predates this line and is unrelated to purity.

      No new test: the property is R2's, already covered by C1 and C3, and
      what a test could pin here is polars' scheduling choice rather than
      our behaviour.

<a id="task-71"></a>
- [x] 71. **E51, the blocked rank-B Gram update — design settled 2026-09-08,
      `EwCov` half built and reviewed the same day, `ewridge` wiring built
      the same day.**
      Three things were measured before any of it was built, because the
      ledger row's own numbers turned out to need qualifying — and then the
      review of the built half found two more, both of which had passed
      every test.

      *The speedup is real, and the row's 5–10× is honest — single-threaded.*
      The design probe first reported **9.5× at k = 1,000 and 11.2× at
      k = 2,000**, and those numbers were wrong in a way the review caught:
      the probe multiplied with `faer`'s `*` operator, which parallelises
      over the global rayon pool by default, against a rank-1 baseline that
      runs on one core. The bank runs each group on its own pool
      (`crates/online-polars/src/pool.rs`), so the honest comparison is one
      core against one core: a full GEMM measures 3.6×, a triangular product
      (`syrk`-style, lower half then mirrored) 5.4×, and the flush as built —
      the triangular product with no `S` allocation and the corrections
      applied on the triangle — **6.6× at k = 1,000 and 6.65× at k = 2,000**
      with a 256-row block, 4.6× at k = 256; a 64-row block gives
      4.9×/4.5×/2.9×. The rank-1 path is memory-bound (it re-touches the
      whole `k×k` at ~110 GB/s every row); the product is compute-bound at
      ~50 GFLOP/s on one core. So the product is `Par::Seq`, always.

      *The review's first finding: the flush as first written measured
      1.0×.* The merge was a hand-rolled triple loop, which is the rank-1
      path's flop count in a different order. The tests could not see it,
      because they check the answer, not the time. It is now
      `faer::linalg::matmul::triangular::matmul` on rows pre-scaled to
      `√uᵢ·(xᵢ − m_B)`, so that `DᵀD` *is* `S` and the product writes into
      the lower triangle of `C` in place. Rule for the row: **a plan row's
      speedup is measured on the implementation, single-threaded, not on
      the probe** — the memory note from E48 ("measure before believing a
      plan row") applied twice in one task.

      *The review's second finding: 49% of the variance, lost to
      cancellation, on a stream every test passed.* The merge was centred on
      the running mean `m_A`, which the design chose over zero (E67's
      cancellation at an offset of `1e7`). But when the history is *nearly*
      gone — an uncapped gap leaving `W_A = 1e-30`, not zero — `m_A` is
      still where the old data was, and the block's scatter is formed as a
      difference of two numbers of order `U·(m_B − m_A)²`: 49% of the block's
      variance at an offset of `1e7`, 24% centred on the block's first
      weighted row instead (that row can be the one *before* the gap). The
      rank-1 recursion is exact here, because its mean jumps to the row. So
      the flush is **two passes: the block's own weighted mean `m_B` first,
      then the product on rows centred on it**, at a cost of one extra `B·k`
      pass. Measured `8.4e-10` of the variance on that stream (the data's own
      resolution at `1e7`) and `3e-14` at `1e3`, and the test that pins it
      (`a_history_all_but_gone_leaves_nothing_to_cancel`) fails at 17% with
      the centring switched back.

      *The merge, as built.* With `Λᵢ` the suffix decay to the block end,
      `uᵢ = wᵢ·Λᵢ`, `U = Σuᵢ`, `W_A = (Πλᵢ)·W_open`, and `W'` the weight
      after the block from the scalar recursion (below): `m_B = Σuᵢxᵢ/U`, `dᵢ = xᵢ − m_B`, `S = Σuᵢdᵢdᵢᵀ`
      (the product), `Δ = m_B − m_A`,
      `C' = (W_A·C_A + S)/W' + (W_A·U/W'²)·ΔΔᵀ`, `m' = m_A + (U/W')·Δ`.
      (The code keeps a residue term `δ = Σuᵢdᵢ/U` that is zero in exact
      arithmetic and `1e-16` in floats, so the general-origin form is what is
      written: `C' = (W_A·C_A + S − U·δδᵀ)/W' + (W_A·U/W'²)·ΔΔᵀ` with
      `Δ = (m_B − m_A) + δ`.)

      *The scalars can stay bit-identical, which the row did not anticipate.*
      `w_sum`, `q_sum`, `prior_scale` and `precision_scale` are an `O(B)`
      recursion, not an `O(B·k²)` one, so the block runs the shipped scalar
      path exactly, per row, as each row is held. Measured against the
      sequential recursion over weights, zero-weight rows and decay: all four
      match to `0.0e0`, and the mean to `1e-16`. Only the `k×k` matrix moves,
      at `5e-13` of the largest variance (offset `1e3`) to `8e-10` (`1e7`).
      **`n_eff` is emitted every row, so this is what keeps blocking from
      changing any per-row output.**

      A closed form for `precision_scale` is *wrong* and was caught here:
      `Π aᵢ = Π λᵢ · W₀/W_B` looks like it telescopes, but the shipped update
      **resets** the product to 1 whenever the history has no weight
      (`a ≤ 0`), which the product cannot see. Measured 0.84–0.99 relative
      error before the scalar pass replaced it.

      *The constraint the row does not mention.* The Gram must be flushed
      before it is read, so the effective block is
      `min(gram_block_rows, solve_every)`. That is not fatal — at k ≥ 1,000 a
      solve is ~3·10⁸ flops, so anyone in the regime E51 targets already
      solves rarely — but it means the parameter cannot be documented without
      it. **And it is incompatible with `window`**: `Moments::of` reads the
      matrix on every row to build the truncated snapshot
      (`crates/online-core/src/window.rs`), which would flush every row and
      buy nothing. The wiring refuses `gram_block_rows` together with
      `window` rather than silently degrading.

      *The read surface narrows the blast radius, once looked at.* `n_eff`,
      `n_kish` and `q_sum` read scalars only; `mean`, `cov`, `raw`, `var`,
      `comoments`, `means` and `precision` read the matrix. The scalars are
      advanced **eagerly, per row, by the shipped path** even while rows are
      held, so nothing that is emitted per row can ever lag — the flush needs
      only `W` at the moment the block opened, which is one snapshot. And
      `comoments`/`means` hand out `&[f64]`, so a `&self` method cannot
      flush; instead every matrix read carries
      `debug_assert!(!self.has_pending())`, which turns a forgotten flush
      into a failure on the first test that reads mid-block rather than a
      wrong number in production. `set_moments` and `decay` flush first: the
      held rows were measured against the state those two replace or age
      (`blend_toward_long_run` is the caller that reaches `set_moments`).

      *What the flush may be triggered by.* Only functions of the row
      sequence: the block filling, a read on the learned-row schedule
      (`solve_every`), `set_moments`, `decay`. **Never a chunk ending** — a
      flush every row is the rank-1 association and a flush every sixteen a
      merge, and the last digits differ, which is the chunk-dependence rule 3
      forbids. The test with teeth is the bank's chunk-invariance test with
      `gram_block_rows` on (wiring, below).

      *The state.* The held block travels in `EwCov`'s state as a last field
      `pending` (`block_rows`, `x` row-major `B×k`, `lam`, `w`, `w_open`),
      skipped when blocking is off so every existing state file is
      byte-identical; a bank saved mid-block resumes on the same block
      boundary. `SCHEMA_VERSION` stays 6. Two consequences found on the way:
      `q_sum` no longer has `skip_serializing_if` (the crate's compact
      encoding is positional, so at most one field may skip and it must be
      last — `tests/state_encoding.rs`; `None` now writes a `nil`, and named
      bytes for `Some` are unchanged), and `ModelState::EwCov` is boxed like
      every other variant, because the extra 88 bytes tripped
      `clippy::large_enum_variant`. Noted, not changed: `EwCovModel`'s `lag` and `win` both
      skip at its tail and `EwRidge`'s `tm` skips mid-struct, which the same
      rule would reject; benign because the bank writes the named form.
      The JSON export (task 69) shows a held block as named finite fields.

      *Reproducibility.* The rank-1 path is bit-reproducible across CPUs;
      the product is `faer`'s and its kernel dispatch follows the CPU's
      vector width, so a blocked Gram's last bits can differ between
      machines. **Frozen fixtures must not enable blocking**, and the
      cross-platform state hand-off stays on the rank-1 path.

      *Only `ewridge` turns it on.* Blocking pays exactly where the Gram is
      consumed on a schedule rather than per row, which is `ewridge` (and
      later `lasso`). `ew_cov` emitting per-row statistics would flush every
      row and gain nothing. So `EwCov` gains the machinery but the five other
      models on it — `ew_cov`, `ew_class`, `corrchange`, `hmm`, `deco` —
      never set `block_rows` and run the untouched path, to the bit
      (`block_rows_zero_is_the_untouched_path`).

      *Tests, in `crates/online-core/src/ewcov.rs::block_tests` (19).* The
      shipped recursion is the oracle throughout. Co-moments are compared
      against the matrix's own scale at the data's own resolution
      (`max(1e-9, 64·|m|·ε/σ)`), not entry by entry — a per-entry relative
      tolerance fails on an off-diagonal that happens to be near zero while
      saying nothing about the merge. Sweeps over `k ∈ {1, 3, 8, 37}`,
      `λ ∈ {1, 0.995}`, three weight patterns, blocks `{2, 7, 64}`, offsets
      `{0, 1e3, 1e7}`; a `k = 131` case for the product's tile edges; the
      near-total-decay stream above; the capped gap (`λ = 0` exactly, rule 9)
      at four positions in a block; a block of nothing but zero-weight rows;
      a zero-weight first row (opens no block, exact); the block boundary
      itself; a read mid-block (`should_panic`); the scalars readable
      mid-block; save/resume at five cuts in both msgpack encodings; the
      JSON shape; block sizes 1 and longer than the stream; resizing
      mid-stream; `decay` and `set_moments` mid-block, both verified to fail
      without their flush. `cargo mutants` cannot see any of the Python
      suite, so this module is where the coverage has to be.

      **The wiring, as built (2026-09-08).** `gram_block_rows` on `ewridge`
      only, `0`/`None` off. In `EwRidgeCfg` the field is `#[serde(default)]`
      and sits *before* `window`, not at the tail: the crate's compact
      encoding is positional and `window` is the one skipping field that
      may be last, so a trailing `usize` would let a saved `window` decode
      into the new slot. The bank writes the named form, so a state saved
      before the field loads with `0` and re-saves with the key.

      *Flush discipline.* `EwRidge::new` blocks its `cov` and the slow
      twin's; `solve()` flushes before it reads; `blend_toward_long_run`
      flushes both twins before it mixes them and carries `block_rows` into
      the accumulator it rebuilds; `restore` leaves the pending block where
      the save left it (a `set_block_rows` there would flush and move the
      boundary). The bank's `gram_of` reads through a new
      `EwCov::flushed() -> Cow`, so `gram()` — the one read of the matrix
      from outside the model — reports the held rows without merging them:
      a read is not a function of the row sequence and must not move the
      block. What may flush is
      unchanged from the design: the block filling, a solve on its
      clock/row schedule, a session blend. Never a chunk ending, and the
      bank's chunk-invariance test with blocking on holds every column,
      `coef_*` included, at 1, 7, 13, 100 and 333 rows a chunk.

      *Three refusals, in `EwRidgeCfg::validate`, surfacing as
      `ValueError` from the builder and the bank.* With `window` (it
      snapshots the Gram on every row, so nothing would be held). With
      `solve_every <= 0` or `max_rows_between_solves <= 1` — a solve every
      row flushes every row, so a block could never hold more than one;
      the message names the default (`halflife / 50`, which is `0` for
      `lam` and an infinite halflife, so those need an explicit
      `solve_every`). And a budget: `B × (k + intercept) × (1 + twin)`
      floats over 256 MiB (`GRAM_BLOCK_BUDGET`, the `bins` pattern), the
      message giving the GiB it would have held.

      *Tests.* Rust, `ewridge::tests` (5): blocked against per-row at
      blocks 1, 3, 16 and 256 (`pred` to `1e-9` of scale, `n_eff` to the
      bit); a solve merges the block first; a blend merges both held blocks
      and keeps the block size; a state saved mid-block resumes bit for bit;
      each refusal. Python, `tests/test_gram_block.py` (21): the same fit
      through the bank at blocks 3, 16 and 256; `0` bit-identical to the
      untouched path; the chunk sweep above; save mid-block and resume bit
      for bit; the held rows legible in the JSON export (`n_held ×
      (k + 1)` values, intercept slot included); `gram()` mid-block
      reporting the held rows and moving nothing; a session change blending
      the held block first; every refusal, including the two default
      cadences; the budget message; and a state file written before the
      field existed — made by stripping the key from a current file with a
      sixty-line msgpack codec in the test, since the loader cannot tell the
      two apart — loading with blocking off, continuing to the bit and
      re-saving with the key. `tests/test_semantics_all_models.py` gained a
      `VARIANTS` list with a blocked `ewridge` (block 5 against a 4-row
      cadence, so every solve merges a partial block), swept through null
      policy, warm-up, weights, resets and the rest of the contract, and
      through `test_predict`'s row oracle; it is a second list because
      `test_model_registry` holds `MODELS` to one entry per model, and the
      first gate run said so.

      *Reviewed (2026-09-08)*, both commits, for bugs and for tests missing:
      no bug. Every matrix read was traced — inside the model only `solve`,
      `solve_standardized` (called from within `solve`) and the blend read
      the moments, and each merges first; outside it only `gram_of`, which
      reads `flushed()`, and both `gram()` and the closed row go through it.
      The gap was in the tests, not the code: the `debug_assert!` on every
      matrix read is compiled out of the release build the Python suite
      runs, so a bank reader that forgot `flushed()` would have passed the
      whole suite and returned stale moments. Added: Rust,
      `crates/online-polars/tests/bank.rs::every_bank_reader_survives_a_held_block`,
      a debug-build pass over every reader — closed groups, `gram` (both
      forms), `coef`, `last_row`, `summary`, `describe`, `predict`, a
      save/load round trip that carries the held rows — with rows held in
      every open group, then the stream continued against an untouched bank
      to the bit; replacing `flushed()` with a borrow makes it panic from
      inside `fit_predict`, which is the check that it has teeth. Python,
      `tests/test_gram_block.py` is now 31: the blocked fit against the
      plain one on every path the solve and the plumbing take —
      standardize (with and without intercept), a ridge × feature-set grid,
      a coefficient prior with `ridge_decay`, two halflives (the slow
      twin), `label_delay`, a drift reset and groups; scoring with
      `predict()` mid-block, which gives the plain model's numbers and
      moves nothing in the state; and a group closed mid-block, whose
      closed row carries the merged moments (checked against an unblocked
      bank's `gram()` at the same row).

      Two things worth knowing that are not bugs. A state saved by this
      build with an `ewridge` spec, blocked or not, is refused by a 0.3.0
      build: `ModelKind` is `deny_unknown_fields` and always writes
      `gram_block_rows` (nil for none). Loud, and the precedent — `window`
      did the same to 0.2.0 — forward compatibility is not promised. And
      "the block never exceeds the solve cadence" is a statement about the
      main accumulator: the slow twin's block is merged when it fills or at
      a blend, never at a solve, because a solve does not read the twin.

      *Measured*, `crates/online-core/examples/gram_block_bench.rs`, the
      `EwRidge::step` — prediction, cross-moments and Gram — single-threaded
      (`Par::Seq`), one core, this machine; rows per second and the ratio
      to the per-row update:

    | k | solve | per row | block 64 | block 256 |
    |---|---|---|---|---|
    | 256 | never | 91,214 | 372,772 (4.1×) | 464,135 (5.1×) |
    | 256 | every 512 rows | 85,578 | 288,352 (3.4×) | 347,461 (4.1×) |
    | 1,000 | never | 5,650 | 28,593 (5.1×) | 37,167 (6.6×) |
    | 1,000 | every 512 rows | 5,711 | 20,863 (3.7×) | 23,719 (4.2×) |
    | 2,000 | never | 1,415 | 6,544 (4.6×) | 8,380 (5.9×) |
    | 2,000 | every 512 rows | 1,309 | 4,435 (3.4×) | 5,644 (4.3×) |

      The `never` rows are the update alone, which is what the block
      changes; a real cadence puts the same `O(k³)` solve on both sides and
      dilutes the ratio — at `k = 2,000` a solve costs ~30 ms whichever way
      the Gram was built, and every 512 rows that is a third of the blocked
      path's time. The ledger's 5–10× is honest for the update, single-
      threaded; the number a bank sees is that, times how rarely it solves.

<a id="task-68"></a>
- [x] 68. **README clarity pass, begun 2026-09-07, done 2026-09-09.** Going through the
      reader-facing prose and fixing phrasing that only parses if the reader
      already shares the frame the sentence was written in. Three rules, in
      the order they bind:

      1. **One idea per sentence.** A sentence carrying a rule, its reason
         and its exception makes the reader hold all three to get any one.
      2. **A sweep belongs in a table.** A list of eight things behind
         semicolons is a table that has not been drawn yet. `bocpd` already
         had a table for its output fields while `micro`, two sections away,
         had the same content in prose.
      3. **Name the mechanism, do not allude to it** (the user's rule,
         2026-09-07, and the one that generalises). "What it costs is state,
         not data" is clear only to a reader already thinking about memory;
         "memory use is proportional to the model's state, not the amount of
         data that has passed through it" needs nothing brought to it. The
         test is whether a sentence can be understood by someone who has not
         yet been told what the sentence is about.

      Prose only: no heading moves, no code block changes, no claim altered,
      so the anchors and `tests/test_examples.py` hold throughout. Progress
      is measured, not eyeballed -- sentences of 45+ words in the README went
      from 37 to 25 in the first batch of twelve edits. Rendered for review
      through GitHub's own `POST /markdown` so the judgment is made on what
      the repo page will show, not on an approximation.

      **Done 2026-09-09, as a whole-README rewrite rather than a sentence
      pass.** The user reported phrasing problems one at a time into
      `docs/PHRASING.md` (each kept verbatim, with the reading and the
      resolution separate), the rules those reports implied were written
      into `docs/WRITING.md`, and the README was redrafted to them over four
      review rounds on a rendered page before anything was committed. The
      three rules above survived and gained a stronger one ahead of them: a
      stated reader — statistics, a little Polars, time-ordered data, and
      nothing of the internals of Polars or of this project — against whom
      every sentence is read. What that changed: the introduction defines
      *spec*, *model bank*, *stream*/*chunk* and *state* before using them;
      "the bank" is announced as always meaning a model bank; Polars-internal
      words (plan, sink, collect, struct column) are replaced by what they
      do; the stream section is one subsection per concept; and prose that
      described a parameter, a structure's fields or an API call is a code
      block with comments — 59 of them in the README, every one run by
      `tests/test_production_hardening.py`, which is how three keyword names
      the old prose had wrong were found. `po.run` and the CLI moved to
      `docs/RUNNER.md`, whose blocks the same test now runs. Four things a
      reader would have needed to know in advance were found by that test
      and not by reading — the point of the rule that every block runs.

<a id="task-61"></a>
- [x] 61. **The leak test's statistic, 2026-09-06.** `assert_plateaus` compared
      the first and last of its post-warm-up marks, which cannot distinguish a
      late allocator step from a slope — the distinction its own docstring
      claims. It failed `main` on a tree whose previous run was green. It now
      takes the median block-to-block gap over five blocks; the CI trace that
      failed reads 0.0 KB/iter, a sustained 14 KB/iter leak still reads 14.
      Acceptance: the file's 16 tests, and the three traces (the real one, a
      leak, a step) checked against the statistic directly.

<a id="task-62"></a>
- [x] 62. **The release gate, 2026-09-06.** The `Pypi` environment now has the
      owner as a required reviewer (self-review allowed, or one reviewer
      deadlocks it), so `publish to PyPI` waits for a click that is only asked
      for once every wheel, the sdist and the state hand-off are green. The
      gate is in front of the one step nothing can undo and in front of
      nothing else: rehearsals, re-runs, the GitHub release and the Pages
      deploy all proceed unattended. A wrong tag is left unapproved rather
      than moved — `refs/tags/v*` has `deletion` and `update` rules with no
      bypass — and the fix ships as the next version, which is PyPI's rule
      too. Recorded in the comment above the `publish` job and in
      `docs/RELEASE-READINESS.md`, "The release gate".

<a id="task-87"></a>
- [x] 87. **Not using a model before it is ready, 2026-09-21
      (`docs/WARMUP-AND-CONVERGENCE.md`).** Two gates stated as intent, on
      every entry point: `min_settled_frac`, a fraction of steady state the
      decay window must have filled (`settled_frac = 1 − 2^(−T/h)`, `T` the
      decay time seen -- rate-independent by construction), off by default;
      and `max_error_inflation`, the ratio by which estimation error may
      inflate a prediction's error over the noise floor,
      `sqrt(1 + edf / n_kish)`, default `sqrt(2)`, which is `ew_ridge`'s
      readiness gate now (its `min_periods` defaults to 0; the `k + 1` rows
      its first solve needs are the model's own floor). Per row: `settled_frac`
      and `withheld_reason` (a categorical, 1.4 B/row) on every model that
      writes a row; `support_coef` beside `coef` for `ew_ridge`, each
      coefficient's data share `1 − λ(Σ̂⁻¹)_jj`; opt-in
      `error_inflation_<slot>`, the row's own leverage against the fit's
      factor. `summary()` carries the same per group; a `ReadinessWarning`
      names, once per (spec, group), a coefficient more ridge than data or a
      noise gate the stream has settled below. Built for `ew_ridge` only:
      `rls`, `lasso`, `kalman` and the scalar models keep `min_periods`
      (the doc's §5.8 says what each would read). Held to identities rather
      than tuned (`crates/online-core/tests/readiness.rs`): the gate within
      2% of the observed error down to `n_kish ≈ 2.3 k_eff`; the per-row
      form conservative by 10–54% at small `n`, which is what an exact `G₂`
      would buy (doc §2.1). Schema 13. Released in 0.9.0.

<a id="task-88"></a>
- [x] 88. **Durations for a time clock: `timedelta` half-lives on a
      `Datetime` clock, requested 2026-09-23.** A `Datetime` or `Date` clock
      column is accepted when the spec's time parameters are given as
      durations -- a Python `timedelta`, as asked -- so a halflife states its
      own unit ("ten minutes") instead of being a number in the column's
      storage unit. Today such a column is refused (the clock dtype decision
      of 2026-08-30 below), because casting it to f64 exposes its internal
      representation: the same 60 seconds reads as 60e3, 60e6 or 60e9 for
      `Datetime(ms/us/ns)`, so `halflife = 600` on a microsecond column
      silently meant 600 µs. That decision declined auto-converting to
      seconds because a halflife's meaning would then depend on the input
      dtype. **Durations dissolve that objection**: with both the clock and
      the parameter carrying units, the meaning depends on neither, and the
      library converts both to one internal scale. So this is the principled
      form of the refusal, not a relaxation of it. Precedent, and the oracle:
      pandas' `ewm(halflife=pd.Timedelta(...), times=...)`, which T-S9
      already uses as the second opinion on the clock model.

      *The rule, as two refusals.* A temporal clock with plain-number time
      parameters is still refused, with today's error still pointing at
      `dt.epoch`. A numeric clock with duration parameters is refused too.
      Either mixture brings the ambiguity back. Numeric clocks with
      plain-number parameters are unchanged.

      *The surface is every parameter measured in clock units*, spec-level and
      model-level -- for example `halflife` and its list form, `max_dclock`,
      `session_gap`, `min_backwards_jump`, `label_delay`, `solve_every`,
      `window`, and model halflives such as `kalman`'s `revert_halflife` and
      `holt`'s `level_halflife`. That list is not the source of truth: a
      completeness test walks every clock-unit field and fails on any that
      lacks the duration form, so a parameter added later cannot be missed.

      *Other surfaces.* The CLI's TOML has no duration type, so it needs a
      string spelling; polars' own duration strings (`"10m"`, `"1h30m"`, as
      in `rolling(period=)`) are the natural one. The Arrow import path (task
      86's unbuilt half) receives a timestamp's unit in the Arrow type, so it
      must read the clock's unit from the schema rather than cast blind.

      *Tests.* The T-E10 trap as a pass: the same wall-clock data as
      `Datetime(ms)`, `Datetime(us)`, `Datetime(ns)` and a `dt.epoch("s")`
      float column, with `halflife = timedelta(minutes=10)` against
      `halflife = 600`, gives identical output on every float field. Both
      mixtures refused, each naming the column, the parameter and the fix. A
      `Date` clock with day durations. The pandas oracle on an irregular
      clock. A timezone-aware `Datetime` across a DST change: deltas are taken
      on the UTC epoch, so the change must not stretch the clock. The
      completeness test. A spec with durations saves and loads -- the spec's
      serialised form changes, so `SCHEMA_VERSION` moves (pre-1.0: raise
      `MIN_SCHEMA_VERSION`, no loader).

      *Open, for the build to decide.* Whether Python also accepts the string
      spelling beside `timedelta`, which would make a spec read the same in
      Python and TOML. Whether a `Duration` column may be a clock; a `Time`
      column stays refused, since time of day wraps at midnight and is not a
      clock. Whether the spec keeps a parameter as the duration the user wrote,
      so `summary()` can show "10m" rather than a converted number.

      **Built 2026-09-23**, on the user's ask to express a temporal clock's
      halflife, `max_dclock` and the other clock-scaled parameters "in
      pl.duration", and to reject a duration that is incompatible with the
      data. The open questions, as the build settled them:

    | question | settled |
    |---|---|
    | the forms Python takes | three: a `pl.duration(...)` that reads no column (evaluated once, at the builder), a `timedelta`, and polars' duration text (`"10m"`, `"1h30m"`), so a spec reads the same in Python and TOML |
    | what the spec keeps | the text: a `timedelta` or an expression is written by the Rust formatter (`timedelta(minutes=90)` is `"1h30m"`), and text is kept as written (`"90m"` stays `"90m"`), so `bank.specs` equals the dict that built it |
    | units | whole numbers of `ns`, `us`, `ms`, `s`, `m`, `h`, `d`, `w`; `mo`, `q` and `y` are refused (no fixed length), and so is `i` (it counts rows) |
    | a `Duration` column | a clock, like a `Datetime`: elapsed time is time |
    | a `Time` column | refused: a time of day starts again at midnight |
    | `0` and `inf` | unit-free, so they may stay numbers beside durations |
    | a number in the clock's own units | `lam` (a decay per unit) and `kalman`'s `q` (the noise a row one unit after the last adds) have no duration form, so a temporal clock refuses them and names the half-life to give instead |
    | the internal scale | the column's own integer nanoseconds (`ArrowCol::Nanos`), and the gap between two rows taken in integers before it becomes seconds (`ClockValue`), so one instant in ms, µs or ns is the same value and a nanosecond gap is exact whatever the stream's age; the previous row's instant is the clock state; a value past what nanoseconds in an `i64` hold (the years 1677 to 2262) is refused by row; a zone-aware column is read as its UTC instants |
    | outputs in clock units | seconds: `holt`'s trend is per second; `summary()`, `groups()` and `closed_groups()` give clock values as seconds since 1970 (`ClockValue::seconds`); `output_index`'s `halflife` is seconds; a grid's field names keep the text (`@h10m`) |

      *Incompatible, and refused.* A temporal clock with a clock parameter
      as a plain number (naming the column, the parameter, the duration fix
      and `dt.epoch`); a numeric clock with a duration (naming the column,
      the parameter and `pl.from_epoch`); one spec that gives both; a
      duration with no clock column; a rate on a temporal clock; a `Time`
      clock; a temporal clock column that another spec also reads as a
      feature, target or weight, which in seconds would be a number nobody
      asked for; and a duration the parameter's own rule refuses, such as a
      zero halflife, which is checked on its seconds.

      *The state.* The model layout, `SCHEMA_VERSION`, is unchanged, so the
      plan's guess above was wrong: nothing to raise. The envelope,
      `BANK_FORMAT_VERSION`, is 3, and a bank writes 3 only when a spec
      carries a duration. Every other file is still version 2, byte for
      byte, so an older build reads it, and refuses a file with a duration
      by its version rather than by a type error inside a spec.

      *Completeness, three ways.* `CLOCK_FIELDS` in `spec.rs` lists the
      clock parameters, and a Rust test walks every field serde knows and
      holds the table to the types; dropping `marginal`'s `window` from it
      fails the test. `Spec::clock_spans` walks the fields directly, since
      the chunk builder calls it on every chunk, and a second test ties it
      to the table. In Python, `tests/test_temporal_clock.py` holds the
      table to the builders' annotations and to their docstrings (12 of the
      15 model parameters say "clock unit" in their own entry; `solve_every`
      in `lasso`, `huber` and `quantile` defers to `ewridge`'s), and fits
      every entry both ways on a temporal clock.

      *Measured.* The same wall-clock data as `Datetime(ms)`, `(us)` and
      `(ns)`, with the parameters as `pl.duration`, `timedelta` or text,
      give output identical to the bit to a float clock in seconds with the
      same parameters as numbers; a `Date` clock with `"5d"` matches a
      clock in days with `5`; a zone-aware clock across the March change
      matches its UTC instants and differs from the wall clock. Ticks 1 to
      5000 ns apart under a 1 µs halflife decay by their exact gaps to
      1e-12, where the same instants as a float clock in epoch nanoseconds
      (a double at 1.7e18 resolves 256 ns) drift by more than 1e-3. Started
      a day, a year and a decade into the stream, the same ticks are read
      to the double's resolution at that age, and the drift is at most that
      resolution over the halflife: 3.9e-6, 9.6e-4 and 1.1e-2 against
      bounds of 1.5e-5, 3.7e-3 and 6e-2, which a test pins.

      *The origin (review of 2026-09-24).* The first build read the clock
      as seconds since 1970, and the pandas oracle on a nanosecond clock
      agreed only to 1.2e-9: a double at 1.7e9 seconds resolves 2**-22 s,
      0.24 µs, so nanosecond ticks were being rounded. The user saw that a
      nanosecond timestamp and `pl.duration` are both exact and asked why
      the result was not. The second build kept each temporal clock
      column's first instant in the state (`ClockOrigins`) and read the
      column as seconds from it, which moved the rounding from 0.24 µs to
      the double's resolution at the stream's age (the table above); the
      README said nothing was lost, and the user asked whether that was so.
      Against the exact nanosecond recursion that build agreed to 2.2e-16
      at the origin; pandas' own `ewm(times=)` is 1.8e-9 from it, so the
      pandas oracle's tolerance is pandas', not ours.

      *Exact gaps (2026-09-24, later the same day).* The user asked whether
      perfect precision was possible, and it was cheaper than the origin:
      the models never see a clock position, only `d_clock`
      (`OnlineModel::step`), and the one subtraction of two positions is in
      `ClockState::advance`. So the clock crosses the chunk as the column's
      own nanoseconds (`ArrowCol::Nanos`; a `Datetime("ns")` column is
      taken as it is, with no pass over it), `ClockState` keeps its
      previous row's value as a `ClockValue` -- `F64` for a numeric clock,
      `Ns` for a temporal one -- and takes the gap between two `Ns` values
      in integers before it becomes seconds. A numeric clock's arithmetic
      is untouched, so every numeric-clock test pins bit-identity. The
      origin, its state field, the summary's offset and the frame's
      `predict` special case are gone; `summary()`, `groups()` and
      `closed_groups()` report a temporal clock as `ClockValue::seconds`,
      seconds since 1970. Schema 14 (`MIN_SCHEMA_VERSION` 14, the pre-1.0
      rule). Memory: a clock column is 8 bytes a row in either form, and
      an `Option<ClockValue>` is the 16 bytes an `Option<f64>` was. Time,
      2M rows of `ewridge` with four features, best of three on a quiet
      machine: a `Datetime("ns")` clock 0.365 s before and 0.367 s after;
      the float clock, whose code did not change, 0.322 s and 0.336 s in
      the same runs, so the change is inside the run-to-run noise. Tests:
      `TestEveryUnitAgainstEveryColumn` holds every duration unit (`ns`,
      `us`/`µs`, `ms`, `s`, `m`, `h`, `d`, `w`), in each form that can
      write it (text; `pl.duration` with the unit's keyword; `timedelta`,
      which has no nanoseconds), against every temporal column kind and
      unit (`Datetime` ms/us/ns and a zone-aware one, `Date`, `Duration`
      ms/us/ns): 192 cases, each in exactly one of two tests and none
      skipped. Where the column can express the unit, the exact recursion
      on gaps of that unit; where a cap or a disorder threshold is finer
      than the column's step, a refusal naming the parameter, the value and
      the step; and a halflife finer than the step read at its own scale.

      *Other surfaces.* The command line takes the same text in TOML,
      tested end to end on a parquet file with a `Datetime(ns)` clock. The
      Arrow import path (task 86's unbuilt half) still has to read the
      clock's unit from the Arrow type when it is built.

<a id="task-89"></a>
- [x] 89. **README rewrite: ten sections in a hierarchy, requested 2026-09-23.**
      The user's brief: plan every section first; put similar concepts
      together, with a moderate number of large sections holding
      subsections, in the order that reads best; then take each section for
      clarity and brevity, preferring code with comments and keeping prose
      for what a comment cannot hold (a reason, a warning, a trade-off,
      math); tables over long bullet lists; every rule in `docs/WRITING.md`.

      *Measured before* (prose outside code and tables): 2,735 lines,
      13,941 words, 82 of 447 sentences at 45+ words, 17 cost words
      (`costs|pays|buys|for free|the price|the point`), 59 python blocks.

      *The map.* Seventeen top-level sections become ten, each holding the
      subsections listed; `←` says where moved content comes from.

    | section | subsections |
    |---|---|
    | Introduction | The idea · Four words · Install (← Install, and the Polars-version note from the top) · A first fit (the two examples) · What you can rely on (← Two guarantees, Mistakes are named, And the rest, as a table) |
    | How a bank sees a stream | What a spec names · Time and decay · A local fit along any feature · Convergence without a decay · Groups · Weights · Warm-up · Labels that arrive late (← Preparing a stream: a spec parameter every model shares) · Nulls, and three ways to hold a row back · Row order and the two guarantees |
    | Running a bank | As a query · In a loop · Output as Arrow (← a paragraph inside In a loop) · Outside a live Python process · Series that tick at their own times (← Preparing a stream) |
    | Saving, loading and serving | Save and load · Serving without learning · What a state file holds · Reading a state without this library |
    | Reading the fit | Coefficients · Output field names · The running sums behind a fit · One row per finished group · Reading a correlation matrix |
    | Diagnostics, selection and evaluation | Per-row diagnostics · Conformal intervals · Evaluating an output frame · Evaluating a stream too large to hold · Data whose truth is known |
    | Models | the model table, grouped by family · Linear models (`ewridge`, `rls`, `lasso`, `kalman`, `huber`/`quantile`, `sgd`, `pa`, `ftrl`, `holt`) · Moments and correlation (`ew_cov`, `marginal`, `deco`, `rcov`) · Clustering and classification (`kmeans`, `micro`, `ew_class`) · Sequential tests and regimes (`seqtest`, `corrchange`, `hmm`, `bocpd`); each model one level below its family |
    | Performance | Throughput · Memory: which calls stream · Tuning memory with Polars' own settings · Chunk size (← a paragraph inside Tuning memory) · Parallelism · Against scikit-learn (← its own top-level section) |
    | Scope and integrations | What this is not (its bullets → a table) · Pathway · Databases: DuckDB and ADBC |
    | Versions, testing and development | Versioning and the Polars pin (its four parts one level down) · Testing · Development · License |

      The Contents line becomes a table: one row per section, its
      subsections linked beside it.

      *Constraints held.* Every anchor `llms.txt` and `docs/RUNNER.md` link
      to keeps its heading text, since an anchor is the text and not the
      level. The model table keeps its header and one link per builder.
      Each model keeps its *API:* and *Rust:* lines. Every python block still
      runs in the namespace `tests/test_production_hardening.py` gives it.
      The models move one heading level down, under their family, so
      `test_api_links` and `test_model_registry` read them there, and
      `docs/EXTENDING.md` step 15 says so.

      *Rules applied to every section.* A parameter, a structure's fields or
      a call sequence becomes a code block with comments; prose stays only
      for a reason, a warning, a trade-off or math. A sweep of six or more
      becomes a table, and so does a bullet list that compares things. One
      idea per sentence. A term is defined before it is used. Nothing is
      said about "the bank" that is true of only some models. Detail cut
      from the README moves to the deep document that owns it, and never
      simply disappears.

      **Done 2026-09-23, to the map above.** Measured on paragraph
      boundaries, the fair count: a count that collapses whitespace merges
      text across headings, tables and code blocks, and reported 82 → 50
      long sentences, most of them artefacts.

    | measure | before | after |
    |---|---:|---:|
    | prose words | 13,496 | 11,590 |
    | sentences of 45+ words | 44 | 2, both two sentences the count merges |
    | sentences of 35+ words | 100 | 33 |
    | mean sentence, in words | 23.4 | 19.9 |
    | cost words | 16 | 0 |
    | tables, as rendered | 15 | 37 |
    | top-level sections | 17 | 10 |
    | python blocks, all running | 59 | 59 |

      Checked mechanically rather than by reading: every in-page link lands
      on a heading (107 written, resolved against the rendered ids); every
      anchor `llms.txt` and `docs/RUNNER.md` use survives; every table row
      has its header's cell count; and the only numbers that left the README
      are the command line's memory figures, which `docs/RUNNER.md` holds.
      Found and fixed on the way: the withheld reasons are now listed in the
      order `WITHHELD_REASONS` declares, which is their precedence; "row
      order matters only when a model forgets" was false of the models that
      step, filter or test; the introduction's link into the API reference
      for `lf.online.fit_predict` had been lost, and now sits under *As a
      query*.

      **Follow-up, 2026-09-23: the approach is now the rule.** The user's
      verdict on the result: "This is an improvement rewrite the writing and
      phrasing doc so that future writing is more like this". So
      `docs/WRITING.md` is rewritten in the style it describes, keeping
      §0–§6 where other files cite them: §2 plans the map before any
      sentence; §4 checks a fact against the code and lets a rewrite drop
      words but never add facts; §5 aims near 20 words a sentence; §6 is the
      pass as eight checked steps, with the counting traps this pass hit;
      §7 gives the table shapes that replaced bullet lists; §8 shows this
      rewrite's before and after. `docs/PHRASING.md` logs both messages
      verbatim, its first entry that confirms an approach rather than names
      a fault.

<a id="task-90"></a>
- [x] 90. **Every user document rewritten to `docs/WRITING.md`, requested
      2026-09-23.** The user's ask: "rewrite every user document that is not
      a work list, plan or history using the latest writing guide".

      *Which documents.* `docs/README.md` divides `docs/` into guides and
      records, so the guides are rewritten and the records are not, with the
      top-level documents a user reads beside them:

    | rewritten | not rewritten, and why |
    |---|---|
    | `docs/PERFORMANCE.md`, `docs/RELEASE-READINESS.md`, `docs/TESTING.md`, `docs/EXTENDING.md`, `docs/STATE-WORKFLOW.md`, `docs/RUNNER.md`, `docs/REGIMES.md`, `docs/OUTPUTS.md` (through `scripts/outputs_doc.py`, which writes it), `docs/README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `llms.txt` | the records: `PLAN.md`, `ENHANCEMENTS.md`, `CLUSTERING.md`, `ANSWERS-E54-E64.md`, the reviews, `IMPROVEMENTS.md`, `SIMPLIFICATION.md`, `BEYOND-O-STATE.md`, `BOOSTED-TREES.md`, `MARGINAL-LAGS-AND-BINS.md`, `WARMUP-AND-CONVERGENCE.md` and `ARROW-SOURCES.md` (the last three a design, a design and a plan); `CHANGELOG.md`, history; `PHRASING.md`, a running log kept verbatim; `VALIDATION.md`, generated and never edited by hand; `CODE_OF_CONDUCT.md`, the Contributor Covenant as adopted; `README.md` and `WRITING.md`, rewritten to the guide in task 89 |

      *How.* One agent per document or pair, each first returning a map and
      the claims it found false or stale, with evidence from the code; the
      maps are recorded here before any prose changes (WRITING.md §6 step
      2). Each rewrite keeps every number, name, link, ID and code block, or
      says where it went; keeps every section number, step number, ID and
      heading text another file cites; corrects a stale claim only with the
      code's evidence; and is measured before and after.

      *The maps.*

    | document | reader | new structure ← what moves |
    |---|---|---|
    | `PERFORMANCE.md` | someone who hit a slower run or a bigger process, or a contributor measuring a hot-path change | the 21 numbered sections stay H2 with their numbers; a new *Reading this document* opens it (a contents table with a "read it when" column ← the "where to look" paragraph; *How the numbers are made* ← the machine and regenerate paragraph; *The headline* ← the status block, as a table; *Words this document uses*); §3's P1–P11 become a status table and one H3 each, their `<details>` plans kept; §5 gains an index of the rejections recorded in later sections; long sentences in §10–§13 and §19 become tables of their figures |
    | `RELEASE-READINESS.md` | whoever cuts a release, or asks which Polars versions are promised | five sections replace two parts of 22 H2s: *Cutting a release* (← the release gate and the rehearsal), *Which Polars versions are promised* (← the pin, 2.0.0rc1 measured, raising the ceiling, whose numbered steps stay), *Keeping the API stable* (← Part 2's API sections), *CI cost while the repo was private* (← the quota and cost sections), *Going public, as recorded* (← R1–R6 and the open-source preparation); cited headings keep their text |
    | `STATE-WORKFLOW.md` | a user about to rely on a state file across runs; second, whoever re-checks a guarantee on a new Polars | a guide section first, *The workflow in four steps* (← the four bold paragraphs, as a table, and a table of R1–R7 with the F and C evidence for each), then the research record of 2026-09-03 with §0–§9 one level down, numbers and headings unchanged |
    | `RUNNER.md` | someone running a bank with no live Python | four sections replace six: *Running it* (a first run; saving, resuming and scoring; a run whose product is its state; closed groups), *The configuration file* (a flag and key table; files; specs in TOML; clocks that are times), *Memory, threads and chunk size*, *From Rust, and the Polars it needs*; every executed `sh` line unchanged, since the tests run them |
    | `REGIMES.md` | someone about to trust `hmm`, `corrchange` or `bocpd` | §0–§7 keep their numbers; §0's six findings become bold-led paragraphs; each section states its conditions before its table; §7 becomes one commented code block and an experiment-to-section table |
    | `OUTPUTS.md` | someone who has found a model's record column and wants each field's meaning | a contents table by family; *Reading this reference* (the field-name parts; the four fields most models write, stated once; what is not listed; how the page is made); the 21 model sections and their anchors unchanged |
    | `docs/README.md` | someone looking for the one document that answers a question, or following a citation | *Guides* and *Records* as tables with family rows; the ten documents neither table listed get rows; the top-level documents get a table |
    | `TESTING.md` | someone deciding whether to trust the library, or choosing what to test next; or following a `T-` ID | five sections replace five unrelated ones: *What the suite proves* (← the scorecard and the backlog's hardening rows, as tables), *What it has found* (an index of every defect, linked to its entry), *Where it is thin, and what is left*, *How the suite looks for defects* (← the oracle rule, the mutation run, the FFI audit), *The entries, by ID* (the lettered tables A–E, each ID once, the backlog rows merged in); every T-ID and section letter kept |
    | `EXTENDING.md` | a contributor adding a model kind, following the steps in order | a checklist table up front (step, file, what you add, the check that fails if you skip it); the 16 steps as `### Step N` headings under their layer, numbers and subjects unchanged; long sentences in steps 1, 6 and 8 become tables of what each kind of model needs |
    | `CONTRIBUTING.md`, `SECURITY.md` | a developer who wants a change merged; someone reporting a vulnerability | same sections; the gate's steps become a numbered list, long sentences split, cited sentences verbatim |
    | `llms.txt` | a coding agent reading it in one pass | the llmstxt.org order: summary, details (the rules as bold-led bullets, a surface table), then file lists, one link per item |

      *Decisions taken on the maps.* Stale measurements are replaced by a
      fresh run of the repository's own script, the old figures kept as a
      dated note: `REGIMES.md` §2–§4, whose `corrchange` numbers the S3 fix
      of 2026-09-19 moved (a t5 size of .075 is now .045, and the power
      finding flips), and §1, which says what its streams really contain
      (one regime, so no switch was ever tested). A record keeps its date
      and gains a dated note rather than a silent edit. Figures credited to
      a surface that did not produce them go: `RUNNER.md`'s 0.95 and 1.41
      GB were the removed `po.run`'s. `scripts/regime_experiments.py`'s
      `epps` step, which stops `all` since a rename on 2026-09-07, is
      fixed, since `REGIMES.md` §7 tells the reader to run it.

      One mapping found a test gone rather than a sentence wrong: `kalman`
      held to river's `BayesianLinearRegression` (`TESTING.md` T-R2, and
      E19) was deleted in `509c6cf` with nothing in its message, and nothing
      has checked the match since. It passes unchanged today, so it is
      restored to `tests/test_river.py` as `23671a8` wrote it, and the
      claim is true again rather than rewritten.

      *Follow-up, not in this task.* `REGIMES.md` §1's `hmm` recovery ran on
      streams that never switch regime; rerun it on streams that do.

      **Done 2026-09-24, to the maps above.** Measured the same way as task
      89, with the splitter that also ends a sentence inside bold; every file
      checked independently for lost numbers, names, links and IDs, table
      cell counts, in-page links and wrapped code spans:

    | document | 45+-word sentences | 35+-word sentences | mean sentence |
    |---|---:|---:|---:|
    | `PERFORMANCE.md` | 107 → 0 | 193 → 0 | 27.3 → 16.6 |
    | `RELEASE-READINESS.md` | 17 → 0 | 45 → 2, both merges | 19.9 → 15.4 |
    | `TESTING.md` | 9 → 0 | 16 → 0 | 21.2 → 13.5 |
    | `EXTENDING.md` | 20 → 1, task 88's paragraph kept whole | 39 → 1 | 28.1 → 14.8 |
    | `STATE-WORKFLOW.md` | 11 → 0 | 23 → 0 | 22.3 → 16.1 |
    | `RUNNER.md` | 0 → 0 | 5 → 0 | 22.3 → 16.7 |
    | `REGIMES.md` | 3 → 0 | 15 → 0 | 21.3 → 17.1 |
    | `OUTPUTS.md` | 0 → 0 | 1 → 0 | 21.1 → 15.9 |
    | `docs/README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `llms.txt` | 2 → 0 | 4 → 0 | 13–22 → 11–13 |

      Every in-page link in the twelve resolves, no code span is wrapped,
      and every table row has its header's cell count. What the loss check
      reports gone is either a correction (`covariance="diag"`, `__hl`,
      `[save_state]`, `online_run`, `finish_label`, `tests/test_lasso.py`,
      the two `po.run` figures, `REGIMES.md` as the home of three defaults,
      `deco` and `rcov` as measured there) or a fragment of an old wrapped
      span; every command and name it lists besides moved into a code block
      or a link. The means sit below the guide's aim of about 20. Bold
      labels, counted as sentences, pull them down, but even without them
      the prose runs near 15 to 16 words, a little choppier than the
      README's 19.8: the long sentences were split and not merged back.

      Found on the way and fixed outside the twelve: the `hmm` docstring and
      `tests/test_corrchange.py`'s notes, which carried the old REGIMES
      numbers; the experiment script's printed conclusions; the README's
      save-and-load paragraph (a bank object has `save`/`load`), its `coef`
      cadence, its "four hundred" chunks (the test runs 7 and 100) and its
      chunk-size figure (`PERFORMANCE.md` §20's); the API reference's front
      page, which still offered the removed expression form; the CLI's
      `--no-output` help; the release and canary workflows' comments, still
      on `<2`; `MIN_SCHEMA_VERSION`'s missing entry for 13; four code
      comments citing the wrong P-ID or section; and C1 and C2 swapped in
      `tests/test_frame.py`. A new test holds `RUNNER.md`'s example config
      to the file its shell blocks run against.

<a id="task-91"></a>
- [x] 91. **Every significant Python, to the newest, requested 2026-09-24.**
      The user's words: "We want to support all significant python versions
      up to the latest." Significant is read as SPEC 0 reads it, the
      versions released in the last three years: today 3.12, 3.13 and 3.14,
      which is the floor this package already had. One `abi3-py312` wheel
      per platform installs on all three. But CI never asked for a version:
      each runner's `uv sync` took whatever interpreter it had, so the macOS
      leg ran 3.14 by accident while the classifiers stopped at 3.13, and a
      test that parsed docstrings by their 3.12 indentation failed there
      alone (`c2462f6`). Now `pyproject.toml` declares 3.12 to 3.14, and CI
      picks each leg's interpreter with `UV_PYTHON`. It runs every declared
      version on Linux, and the floor and the newest on macOS and Windows,
      since a Python version rarely behaves differently by OS, or an OS by
      Python version. The API reference is built on one leg, since its
      Pages artifact can be uploaded once a run.
      `tests/test_ci_cost_policy.py::TestPythonVersions` holds the matrix to
      the classifiers, so a declared version cannot go untested; six faults
      injected into the workflow and the metadata each fail it. The whole
      suite passed locally on CPython 3.13.15 (3,055 tests) and 3.14.6
      (3,059, the new policy tests among them). Not done,
      each a decision for later: a floor below 3.12 (3.11 needs
      `abi3-py311`, a new wheel tag and a syntax pass under ruff's py311
      target; 3.10 also `typing_extensions` for `Unpack`); an advisory leg
      on the next pre-release (3.15, due October 2026); and free-threaded
      builds, which an `abi3` wheel cannot serve.

<a id="task-92"></a>
- [x] 92. **Mutation testing in CI, a release comparison, property tests
      and oracles, requested 2026-09-24.** The user asked whether more
      mutation testing would catch subtle bugs. The analysis found that this
      session's subtle bugs came from oracles, review, experiment and a CI
      leg's Python, while mutation testing found weak tests (48 survivors in
      `stats.rs`) and no bugs. The user asked for all three recommendations,
      and for the README's row-order section to say how a bank detects rows
      out of order.
      - *Mutation testing* (`.github/workflows/mutants.yml`). The changed
        lines of every push and pull request, failing on a survivor that
        `scripts/mutants_equivalent.toml` does not name; all of
        `online-core` weekly in sixteen shards of up to two hours, reported
        through `scripts/mutants_report.py`, on its schedule only while the
        repo is public. The scope, `online-core` and `span.rs`, was
        measured: `span.rs`'s own Rust tests caught 55 of 60 viable mutants,
        and its 4 real survivors now have tests; a one-in-ten sample of
        `spec.rs` left 16 of 39 viable mutants, pytest's ground, out of
        scope. The equivalents list matches by file, function, mutation and
        code on the line, so an entry follows its line and lapses when the
        line changes.
      - *The release comparison* (`scripts/compare_release.py`,
        `scripts/release_probe.py`): every output of 30 specs over 400 rows
        against a PyPI release, bit for bit. The first step of a release
        (docs/RELEASE-READINESS.md, which now lists the steps), and a report
        on every CI push. Measured: identical to 0.10.0 and 0.9.1; 112
        fields differ from 0.8.0, all from 0.9.0's declared gates.
      - *Property tests* (`tests/test_properties_temporal.py`, 11 properties)
        on duration text and temporal clocks. They found four bugs, all
        fixed with the tests pinning them: a space after the sign
        (`"+ 5m"`) was accepted and named a grid's field; a `pl.duration`
        past 292 years wrapped (585 years stored as `384ns`), because
        `dt.total_nanoseconds()` wraps; a `timedelta` that long was an
        unnamed `OverflowError`; and `label_delay` broke hard rule 3, since
        the held rows' clock was a running sum within a chunk and a fresh
        sum at each chunk's start, which round differently, so
        `settled_frac` depended on the chunking. The stream now keeps that
        clock per instance, in the state: schema 15, and a schema-14 file
        loads with it rebuilt as 14 did. Single-chunk outputs did not move
        (identical to 0.10.0, bit for bit).
      - *Oracles* for 40 model paths no independent reference held
        (`tests/reference_paths.py` and five `test_oracles_*` files, 59
        tests): the lasso's targets, gaps, windows, selection and
        intercept; `ewridge`'s grids, windows, sessions and schedule;
        `rls` and `kalman` with several targets; `ftrl`, `pa`, `sgd`,
        `holt`; the plain sigma. Each is written from the objective or the
        docstring's equations, with seeded bugs failing it. They found four
        disagreements, left open as tasks 94 to 97, and `sgd`'s docstring
        giving the intercept an `l2` it does not get, now corrected.

<a id="task-93"></a>
- [x] 93. **The PyPI page's links to other files, reported 2026-09-24.** PyPI
      shows the README, where a relative link resolves against pypi.org and
      is a 404; in-page links worked. `scripts/pypi_readme.py` rewrites the
      76 relative links to GitHub at the release's tag, run by
      `release.yml` before maturin in the `sdist` and every `build` job;
      the README in the repository keeps its relative links. A published
      description never changes, so 0.10.0's page stays as it was.

<a id="task-94"></a>
- [x] 94. **A windowed Gram's zero variance comes back as rounding noise.**
      Found by the task 92 oracles. The drop rule is "variance exactly
      zero" (`variance_is_usable`, since T-E9), but a window's Gram is
      rebuilt by subtraction, so a feature constant over the window reads
      about 1e-14 and is kept. A windowed lasso at λ = 0 whose window holds
      one row of a sparse target then divides noise by noise: predictions
      of 8.6e33 and 2.8e55 where the fit is −1.0. It takes a lowered gate
      (`min_periods=0.5`); at the default the rows are withheld, λ = 0.1 is
      sane, and `ewridge` is withheld by its noise gate. The fix is a
      noise-aware zero for a subtracted Gram, since T-E9 removed the
      relative threshold for good reasons.
      **Done 2026-09-24, without a threshold.** Measured first: the
      remainder a constant feature leaves is of the order of machine epsilon
      times its level times the spread it had, and grows with the rows
      since the boundary. At a level of 1e8 and a spread of 1e-3 it was
      1.2e-5 of the terms that cancelled for one row, and 0.46 at 100,000
      rows, so no threshold told it from a real spread; a spread of 1e-5 at
      a level of 1e6 came back within a factor of 20. Instead every `EwCov`
      keeps, per feature, the value it has held since it last changed and
      the weight of those rows (schema 16). Where that weight is all the
      window's, to `EMPTY_FRACTION` of the live weight, `truncated` sets the
      feature's variance and covariances to exactly zero and its mean to
      the value. `gaps` then zeroes that slot's cross-moment with each
      target (Cauchy-Schwarz), so an unstandardized ridge gives a held
      feature a slope of exactly 0, where it gave 0.0057 at a level of 1e6.
      A row of weight 0 ends no run, a blend with the slow twin forgets the
      runs, and a state from before schema 16 starts them at its next row.
      The tests: `window::tests` over every level, spread, halflife and
      window length measured, a window of a thousand halflives among them;
      `the_runs_follow_the_rows_blocked_or_not`;
      `a_feature_held_over_the_window_gets_no_slope`. The oracle
      references drop a feature that holds one value on every row of its
      Gram, and hold the windowed problems they had left out.
      `TestWindow::test_a_window_down_to_one_row_of_a_target_fits_its_intercept_alone`
      fails on 0.10.0. Under cargo-mutants, the changed lines of tasks
      94-97 give 99 mutants: 4 unviable and the rest caught. The last
      survivors needed tests with a second target and a third feature.
      **Review 2026-09-25.** The run's weight against the window's, two
      numbers equal in exact arithmetic, drifted apart under a long window
      and a long halflife -- the window's `w_now − f·old.w` takes one `exp2`
      over the window's clock where the run's weight is a product of per-row
      factors -- by 1.6e-12 of the live weight at a halflife of 1e6 with
      fifty thousand rows of history and a window of as many, past the
      1e-12 the rule allowed, and the held feature read as moving again
      (`a_long_window_under_a_long_halflife_still_reads_a_held_feature`
      failed on the old rule). Runs now keep the learned-row index that
      started each run, `EwCov` and `marginal` count learned rows, the
      window's snapshot records the count, and a slot is held over the
      window when its run started at or before the first learned row inside
      it: row indices do not drift. A row whose weight is nothing next to
      the accumulator's (`EMPTY_FRACTION` of it, the window's own notion of
      nothing) is neither counted nor tracked, which keeps the boundary
      cases of no account as they were. The runs' ageing went with the
      weights. Also found: the zero reaches `ew_cov`'s and `ewclass`'s own
      windows (a held feature reads variance 0 and a null correlation, where
      it read noise; `ew_covs_window_reads_a_held_feature_as_no_spread`),
      and an unstandardized solve with no ridge meets an exactly singular
      Gram on a held feature, which the solver's jitter takes with the
      feature's slope exactly 0
      (`a_held_feature_under_a_window_leaves_an_unregularized_solve_finite`).

<a id="task-95"></a>
- [x] 95. **The lasso's `lam_selected` under a `window` departs from its
      documentation** ("the selection error is truncated with the sums"):
      14 to 19 rows of ~277 at the default `select_halflife`, 88 to 90 at
      `select_halflife=10`. Three departures in `lasso.rs`, each moving rows
      on its own: the snapshot's selection part is taken after the row's own
      error and its weight is aged twice; `window_sel_err` reads the ring as
      it stood at the previous row; and the truncation uses the model's
      halflife, not `select_halflife`. An emulation with the three corrected
      lands on the oracle on every row.
      **Done 2026-09-24,** and a Rust oracle found a fourth departure: on a
      row that does not score the target the choice stood from the last
      scored row, while the window moved on. Now the window moves to the
      row first, its snapshot takes the selection before the row's own
      error, aged once by the selection's decay, the truncation ages by the
      same decay, and the choice is made again on every row. Where nothing
      has aged out of the errors it reads the live ones, to the bit.
      `lam_selected_under_a_window_is_the_argmin_inside_it` recomputes the
      choice from each row's reported predictions. The oracle file holds
      `lam_selected` under a window, with and without a halflife of its own,
      and that fails on 0.10.0 at eight rows.
      **Review 2026-09-25.** A window with no scored row of a target fell
      back on the whole-history errors and chose from them every row, which
      the docstring said it would not; the choice now stands until the
      window has a scored row again (`an_empty_selection_window_keeps_the_choice`).
      A state saved by 0.10.0 mid-window carries selection snapshots in the
      old meaning (after the fold, aged twice) and is off for one window
      after resuming; pre-1.0 compatibility is waived.

<a id="task-96"></a>
- [x] 96. **`lam_selected` under a `min_periods` list.** Each target's
      selection error folds its predictions on rows that its own threshold
      still withholds, which is review S2's "a gate on the output, not on
      the model", but E7 says a target not yet ready is withheld "before it
      can reach … selection". The code and the two documents need one
      reading; the user's call.
      **Done 2026-09-24: the code is kept,** the user's call. A target's
      selection folds the model's own predictions for it from the model's
      first prediction, whatever its own threshold withholds, as learning
      does (S2). E7 and the lasso builder's `select_halflife` entry now say
      so, and the oracle file holds `lam_selected` with a `min_periods` list.

<a id="task-97"></a>
- [x] 97. **Two conventions to settle.** `holt` reports `coef` as `[0, 0]`
      before a target's first observation, where `docs/OUTPUTS.md` says
      `coef` is null "before the model has anything to report"; and the
      test reference `kalman_ref` builds its standardized `coef` from the
      scales before the row and the means after it (off by up to 0.38), so
      no test compares it, while the library's `coef` predicts the next row
      to 1e-16.
      **Done 2026-09-24,** as recommended and the user asked. `holt`'s
      `coef` is null for a target before its first observation (NaN from
      `coefficients`, which the stream writes as null), and its oracle holds
      `coef` on every row. `kalman_ref` reads its standardized `coef` with
      the scales and means of one moment, after the row, and
      `tests/test_oracles_rls_kalman_paths.py` compares kalman's `coef`.

<a id="task-98"></a>
- [x] 98. **The Rust tests link no Python, 2026-09-24.** Task 91's matrix
      failed on its first push. The Linux legs for 3.13 and 3.14 could not
      start the CLI's tests: "libpython3.13.so.1.0: cannot open shared
      object file", exit code 127. `cargo test --workspace` built online-py
      into the same graph, where pyo3-polars turns on polars-error's
      `python` feature, so every test binary that links polars linked
      libpython too. The runner's own 3.12 keeps its libpython on the
      loader's path, and uv's 3.13 and 3.14 do not. macOS passed because its
      libpython has an absolute install name. `cargo test` now leaves
      online-py out in CI, the canary, the gate and the docs that give the
      command. It has no Rust tests, pytest covers it through the extension,
      and the CLI's tests now run the CLI as it ships, with no Python in it.
      `tests/test_ci_cost_policy.py::TestTheRustTestsLinkNoPython` holds
      every workflow and the gate to it, and asks cargo whether pyo3-ffi is
      in that graph.

<a id="task-99"></a>
- [x] 99. **pyarrow, tested as a reader and as a source, 2026-09-24.** The
      user: "packages that we do not want to depend on are fine if needed
      to test with other libraries." The README, CLAUDE.md, a docstring, the
      type stub and a Rust doc comment said pyarrow reads a bank's Arrow
      output. `docs/ARROW-SOURCES.md` said that half stayed unverified,
      since pyarrow "must not become [a dependency] to test a sentence".
      pyarrow 25.0.1 is now in the dev group, and
      `tests/test_pyarrow_interop.py` measures it. `pa.array`,
      `pa.chunked_array`, `pa.record_batch` and `pa.table` each take an
      `ArrowStruct` as it is, with `fit_predict`'s values and dtypes, and a
      struct pyarrow has read is spent. A `RecordBatchReader` streams into
      both streaming paths with the whole frame's numbers, and a reused
      reader plan warns. The package keeps its single dependency:
      `tests/conftest.py` makes pyarrow unimportable in the rest of the
      suite, which so runs as a user without it does, and the interop tests
      run pyarrow in child interpreters. The same check corrected three
      places that said duckdb reads the struct directly. duckdb 1.5.5
      refuses it and takes it through `pl.Series`, as the README said.

<a id="task-100"></a>
- [x] 100. **Libraries the package does not depend on, allowed in tests,
      2026-09-24.** The user: "Be sure that the future test plan allows non
      dependent libs for testing." §9's test plan now says so, and
      `docs/TESTING.md`, "Libraries the package does not depend on", gives
      four rules. Declare the library in the dev group by name, and import
      it plainly. Keep it out of the package's reach when its presence
      changes other libraries, as pyarrow's does. Never add it to the
      package's dependencies. `CLAUDE.md` points there.
      `tests/test_dependency_policy.py` checks the first, second and fourth
      rules from the tests' own imports. Two faults injected each fail it:
      pandas taken out of the dev group, and an `importorskip` put back. It
      found pandas and scipy reaching the tests only through statsmodels,
      and both are declared now. The 34 `importorskip` calls became plain
      imports: they never skipped where the suite runs, and could only have
      hidden a broken environment. Not taken: scikit-learn in
      `tests/test_sgd.py`, and Pathway, whose licence keeps it out of
      `pyproject.toml` until the user decides. Reading the first Windows
      run's skips found one more that was not a platform guard:
      `test_the_cli_writes_the_sidecar` looked for `target/<profile>/online`,
      which has an `.exe` there, and its TOML carried the path's
      backslashes as escapes (T-W3b's trap). It now takes the `online_cli`
      fixture and writes POSIX paths, so it runs on Windows too.

<a id="task-101"></a>
- [x] 101. **A feature that stops moving leaves a rounding artifact outside a
      window, and the standardizing models read it.** Measured 2026-09-24:
      one feature of three holds one value after 300 rows, halflife 20 rows.
      Its running mean approaches the held value until the step rounds to
      nothing, and stops `1/(2b)` rounding steps short. The variance and the
      cross-moment with the target then settle on that gap instead of
      decaying together. The lasso's slope on the feature keeps the 0.47 it
      learned until the stall and degenerates after it: -79 at 50 halflives
      at a level of 1000, -6.4e9 at 100, and -4.7e3 at 40 at 1e8. `ewridge`
      and `huber` with `standardize` do the same, and `kalman` and `sgd`
      predict differently at each level, by up to 0.12. A held target does
      it too: slopes stuck at 2e-8 at 1e8, a target variance on the gap's
      square, `marginal`'s correlation at -0.06 where it goes to zero, and
      its bins' split gain at 0.45.
      **Built: every running mean is a pair, 2026-09-24**
      (`crates/online-core/src/comp.rs`). A mean is `hi + lo`, `hi` the
      double and `lo` what it leaves out. A row's deviation is
      `(x − hi) − lo`, and the step `b·d` goes into the pair by Kahan's
      compensated sum, so no part of it is dropped. Where a plain step would
      round to nothing, `lo` keeps it; the deviation keeps its precision as
      it shrinks, and the mean follows exact arithmetic to the value. Nothing
      needs to know that a value is held. The pairs are in `EwCov` and its
      blocked flush, `EwDiag` and its `including` view (which now hands back
      the deviation), `TargetMoments`, the gaps accumulator's feature and
      target means, `robust`'s target means, `marginal`'s pairs, lags and
      bins, `EwAutoCorr`, `deco`'s level and `bocpd`'s runs. `EwLagCov`,
      `ewclass` and `hmm` take their deviations from the pairs. A row of
      weight 0 takes no step: Kahan's sum does not keep `lo` under half a
      rounding step of `hi`, so adding zero could round the pair afresh, and
      `label_delay`'s doubled-stream oracle caught it.
      **Two designs were built first and replaced.** Snapping the mean to
      the held value at the stall, the fix this entry first proposed, jumped
      where exact arithmetic still carries the gap. With no decay that gap
      closes only as `1/n`, and the snap moved `sgd` 8.2e-3 at 1e8 and
      6.9e-2 at 1e12. Stepping the gap itself toward a value detected as
      held, `g' = (1 − b)·g`, followed exact arithmetic but needed every mean
      to know which values were held, and a rule wherever a mean moves by
      other means. A row of weight 0 found a slot held while carrying another
      value, and moved the mean to that value less the gap: the ridge then
      predicted -0.02 where it predicts 2.41. Blends, `robust`'s score steps
      and the `including` view each needed a rule of their own, and a blocked
      Gram's slope on a held feature still wandered from -0.45 to 1.71 at a
      halflife of 5 rows. The user asked whether the idea was as solid as it
      seemed. The pair was prototyped against it and measured better on
      every count below.
      **Measured**, as the worst relative prediction difference between the
      fit at a level and the same fit at 0.5, pair against the gap design:
      `sgd` 1.7e-9 at 1e8 and 3.2e-6 at 1e12 from the stop (2.3e-5 and
      1.7e-3). The regressions stay within one rounding step of the level.
      With no decay at 1e12, `sgd` falls to 4.5e-10 by row 300,000, where
      the gap design grew to 1.7e-3, and the ridge's slope stays 0.50–0.52
      where a plain mean took it to 0.04. A blocked Gram's slope stays inside
      the row-by-row range at every block size, 0.33–0.65 at a halflife of
      5 rows in blocks of 16. A held target's slopes decay to 1.35e-45 at
      every level, 2^-150 of 2.
      **Cost.** Outputs change in their last bits: 19 of the release
      comparison's 30 specs, by a median of 3e-16 of the value and at most
      1.4e-14 (0.11.0 declares it). Seven Python oracles that replayed the
      plain recursion to the bit replay the pair. Interleaved with HEAD,
      `EwCov::update` is unchanged at 64 slots; `EwDiag::update` takes 6.0,
      10.7 and 24 ns a row at 4, 16 and 64 slots, against 5.2, 6.3 and 15.5;
      and `sgd`'s step 57 ns at 16 features, against 49. `ewridge`'s step is
      unchanged, its solve dominating. *2026-09-27: not so at scale.* The benchmark,
      0.11.1 against 0.10.0, read `ew_cov(lags=[1..5])` at 0.53 and a
      10-target `ewridge` at 0.64; a bisect put the first and 15 % of the
      second on this task (the pair's deviation inside the lagged
      co-moments' `k * k` loop) and the rest on task 132's own means, whose
      low parts' length was checked once a feature. Both now take each
      deviation once, bit for bit the same, and are back at 1.07 and 0.94 of
      0.10.0 (PERFORMANCE §26).
      **Found by the sweep of every running mean, and fixed here:** a
      windowed `marginal` pair did not get task 94's zero spread for a slot
      held over the window, and `beta` divided a remainder by itself.
      `marginal` now keeps runs for its window, in the `Runs` that `EwCov`'s
      task-94 runs moved into. **Found and left:** Page-Hinkley's mean feeds
      only a sum that `delta` swamps; `holt` centres nothing; the uncentred
      EW means of squared residuals have no gap to settle on; `ColumnStats`
      in the data summary is report-only. The clusters are task 102, and
      `corrchange`'s span variance task 103.
      **Tests.** `tests/held_values.rs` holds every centring model for 150
      halflives after a feature or target stops, at levels from 0 to 1e12
      and with no decay. Twelve of its fourteen tests fail with the means
      plain; the other two are contracts that hold of any design: a row of
      weight 0 changes nothing, and a state saved mid-hold resumes to the
      bit (which fails with the low parts left out of the state).
      `comp::tests` pins the pair, and each accumulator has a test beside it.
      `cargo mutants` over the change: 402 mutants, 393 caught, 6 unviable,
      3 equivalent (`scripts/mutants_equivalent.toml`: a term that is 0
      either way on a row of weight 0, a product against 0 that a quotient
      matches, and a guard at exactly 0 that takes 0 away), none missed;
      the 20 that timed out beside a running gate were re-tested alone and
      caught. Each survivor's test was applied to the mutant by hand first,
      and three did not kill it until rewritten: a residue test whose unit
      weights made the block mean correctly rounded, and a variance read
      through `var`, whose `max(0)` takes a NaN for 0.
      **Review 2026-09-25.** The pair is the right fix and every site steps
      and reads it; three deviations still read `hi` alone (`deco`'s and
      `corrchange`'s standardized residuals, `ew_cov`'s PCA score) and now
      read the pair; the window's subtraction `m_u − m` reads `hi` alone,
      bounded at `ratio·g·ulp(level)/σ` of the windowed variance (2e-5 at
      1e12 with unit spread over three halflives), and the held slot is the
      runs' job, so it is left. Schema 16 changed twice before release
      (`runs`, the `*_lo` fields) without a bump: pre-1.0 compatibility is
      waived, a bank file from the intermediate build loads with its runs
      restarted. Pinned since: `EwDiag`'s own skip on the fixture, a
      windowed hold (the slope stands while the window has spread, then the
      feature is dropped as one without spread), a hold that ends, a blocked
      ridge at 1e12 with rows of no weight carrying another value. Left:
      the classifiers and `no_weight` for `bocpd`, `deco`, `hmm`.

<a id="task-102"></a>
- [x] 102. **A stopped feature in the clusters' metric.** Found by task
      101's sweep. `kmeans` and `micro` standardize distances by `1/var`
      from `FeatureMoments` (`cluster/summary.rs`), refreshed every row,
      with `standardize` on by default. A feature that moves and then holds
      stalls there too: its variance settles near the mean's gap squared,
      `g_m²`, and a row's term against its own centre, which has its own gap
      `g_c`, becomes `(g_c/g_m)²`. The entry first put that anywhere from 1
      to 10³; it is `(n_c/N)²`, at most 1 and `1/k²` for equal clusters,
      since each gap is a rounding step over that mean's own step size `b`.
      **The problem is the metric's before it is rounding's.** `1/var` is
      unbounded as the recent variance goes to zero, and under decay it does
      so within ten halflives of any feature going quiet -- a flag that
      stops firing, a sensor at rest, a market closed -- not only of one
      that stops for good. A centre that receives rows follows the held
      value at the rate the variance decays, so its own term fades; a centre
      that receives none is infinitely far for good; and on the row where
      the feature moves again every centre is infinitely far: the argmin
      cancels what every centre shares, but the radius does not, and a
      structural move or an admission test that reads it goes wrong. The
      `else 1` at exactly zero variance is a discontinuity: raw units on one
      row, `1/ε` on the next. **Measured 2026-09-24** on the prototype
      (`scripts/clustering_experiments.py`'s streams, 20 000 rows, k = 5, p
      = 4), with the metric's variance floored at a fraction of the
      feature's long-run (undecayed) variance, `v_i = max(var_i, scale_floor
      · var_i^∞)`: at 0.01 and 0.1 the floor binds on no row of the static,
      drifting, feature-scaled-by-100, noise-grows-tenfold or feature-stops
      streams, so every number there is unchanged; on a feature quiet for
      twenty halflives (rows 8 000-14 000 at halflife 300) it binds on 3
      600-4 600 rows, and after the feature moves again `kmeans` split-merge
      recovers to ARI 0.731 in place of 0.220 (purity 0.804 for 0.487; 0.583
      at a floor of 0.01), `micro` unchanged at 0.786, plain `kmeans` 1.000
      either way. At 1 the floor is a different metric, binding on half the
      rows of every stream: 0.626 for 0.665 on the scaled stream's first
      quarter and the same elsewhere. **A feature that stops for good costs
      `kmeans` 0.24 ARI for a halflife after the stop at every floor**
      (0.756 against 0.999 with the feature dropped): the stale spread keeps
      penalizing every centre the held value is far from, until the centres
      and the variance have followed it, which is the model's memory at
      work, the same lag as a level shift, and not something a metric floor
      reaches. **Design, from the measurement:** `scale_floor`, default 0.1,
      0 for the metric as it was, on `kmeans` and `micro`; `FeatureMoments`
      keeps a reference beside the EW moments, and its means become pairs
      under task 101's rule. The centres stay plain: with the floor, a plain
      centre's gap costs `g_c²/(scale_floor · var_long)` in a distance, 1e-9
      at a level of 1e8 and 0.04 at 1e12 with unit spread. **The reference
      is not the undecayed variance.** That was built first, and
      `model_contract`'s recovery caught it: a row at the input bound put
      1e200 into a reference that never forgets, and the metric was lost for
      good, where the EW variance takes a thousand halflives to forget such
      a row and the contract gives it fifteen hundred. No reference that
      decays slowly enough to hold through a quiet spell forgets an extreme
      in time, so the reference must never take one: it is the same Welford
      variance at eight times the halflife, each row's weight clipped at the
      reference's own (a row takes at most half of it) and its deviation at
      ten standard deviations, started from the medians of the first five
      learned rows, which up to two rows at the bound among them do not
      move. A row at the bound then moves it by a factor of twenty-six at
      most, which a few of its halflives undo. The cost of the slow
      reference is that a hundredfold drop in a feature's variance (tenfold
      in its spread) is under-weighted by up to tenfold for some
      twenty-seven halflives, a hundredfold drop in spread by up to two
      hundredfold for eighty; the cost of the clip is that a feature without
      spread has no scale to clip against and takes its first move as it
      comes. **Built 2026-09-25.** `scale_floor` on `kmeans` and `micro`
      (default 0.1; 0 is the metric as it was), validated finite and ≥ 0 in
      the cfgs and the spec; `FeatureMoments` keeps `w_long`, `mean_long`,
      `var_long` and the start rows beside the EW moments, both means as
      pairs, `decay` takes the reference's factor (`decay.factor(d_clock /
      LONG_HALFLIVES)`), and `metric` takes the floor; a state without them
      loads and starts the reference at the next rows. The Python reference
      mirrors it bit for bit at every floor, the prototype carries it, and
      CLUSTERING.md §6.1 and §10 record the measurement, re-run on the final
      reference. **Tests**:
      `summary::tests::the_metric_is_floored_at_a_fraction_of_the_long_run_variance`,
      `the_reference_starts_from_the_medians_of_the_first_five_rows`,
      `a_row_at_the_bound_moves_the_reference_by_a_bounded_factor` and
      `a_state_without_the_reference_starts_it`; `kmeans` and `micro` each
      give a stream whose third feature moves and then holds at a level of
      1e8 the labels and distances (to 1e-6) they give it at a level of 0
      (`a_stopped_feature_at_a_level_leaves_the_assignments_alone`,
      `..._summaries_alone`); `model_contract`'s recovery from the bound,
      which the first design failed; `tests/test_kmeans.py::TestScaleFloor`
      holds the bank to the oracle at floors 0, 0.1 and 1 on a feature quiet
      for thirty halflives and shows the floor binding only in the spell; a
      negative floor is refused by name in both builders. **Review
      2026-09-25.** Four findings, all confirmed and fixed. The cfg's
      `scale_floor` had no serde default, so every earlier `kmeans` or
      `micro` state file was refused on load (hard rule 5); it has one, 0,
      the metric the state had, and a whole-state test strips the key and
      restores. The docs claimed a bound the mechanism does not give: the
      reference decays through a quiet spell, so the weight grows as
      `2^(Q/8)/scale_floor` over `Q` halflives of quiet, 58 at twenty, where
      `1/var` alone gave `2^Q`; every doc now says so. The reference's start
      weight was a row count, which broke the weight-scale invariance every
      other moment has (the same stream at weights 1 and 1e-3 read metrics
      2× apart through a quiet spell); it is five times the median of the
      start rows' weights, and `the_reference_scales_with_the_weights` holds
      the two streams to 1e-9. A feature without spread took its first move
      as it came, and a first move a million away put 1e11 into its
      reference for two hundred halflives; the reference is now per feature
      and a first move from no spread starts it over, that row the first of
      the next five
      (`a_feature_without_spread_starts_its_reference_over_on_its_first_move`).
      Measured, the cost the review predicted for a level shift in a
      feature: at halflife 100 with ten thousand rows after the shift, the
      floor binds on no row for a shift of 10 or 30 standard deviations (the
      clipped mean step follows a shift at five a row) and on a third of the
      rows for 100, where the shift itself takes `kmeans` to ARI 0.21 with
      the floor or without. Nits taken: the bound row's factor is `1 +
      CLIP/4` = 26, not fifty; `total_cmp` for the medians; a reference
      whose weight has decayed to nothing takes the next row whole and
      starts over on the one after
      (`a_gap_past_the_references_weight_starts_it_over`); a state saved
      mid-start resumes to the bit; a state missing any one field of the
      reference starts it over; the core cfgs refuse a bad floor by name;
      the reference's decay factor at the call sites is pinned against a
      `FeatureMoments` driven by hand
      (`the_reference_decays_at_eight_halflives`).

<a id="task-103"></a>
- [x] 103. **`corrchange` takes a span's variance as `E[x²] − E[x]²`.**
      Found by task 101's sweep. `long_run_sd`, the delta-method standard
      deviation of a span's correlation, forms `σ_x² = E[x²] − E[x]²` from
      raw moments, and its centred series `x² − E[x²]` cancels against
      `2·m_x·(x − m_x)` through `D₂`; both lose the level's digits. At a
      level of 1e8, `x²` is 1e16 with a rounding step of 2, so a unit
      variance is noise. **Measured 2026-09-24**, on eighty rows whose
      deviations are multiples of 2⁻²⁰ so that adding a level is exact: `D̂`
      is off by 5e-10 of itself at a level of 1e3, by 1.2e-5 at 1e5, and NaN
      at 1e8, where the variance cancels to a non-positive number and the
      monitor flags nothing. **Design: centre and scale the span first.**
      `ρ̂` is shift-invariant and so is the delta-method variance of it: the
      shifted moments are an affine image of the raw ones and the Jacobians
      cancel, so the same number comes out of better-conditioned inputs.
      With the span centred at its means and scaled by its standard
      deviations, `m_x = m_y = 0` and both variances are 1, so `f = D₃D₂ =
      (−ρ̂/2, −ρ̂/2, 0, 0, 1)` and `D̂²` is the Bartlett long-run variance
      of the one series `ξ_t = x̃_t ỹ_t − (ρ̂/2)(x̃_t² + ỹ_t²)`: Wied,
      Krämer & Dehling's own form, at 1/25 of the 5×5 kernel's cost.
      `corr_of` already centres. **Built 2026-09-24.** The span's means are
      pairs (`crate::comp`): a second pass over the exact differences from
      the first pass's mean picks up what its division rounded away, since
      that rounding is a shift of the whole span, which `ρ̂` cancels to
      second order but a first-order term in `ξ` carries, 1e-10 of `D̂` at a
      level of 1e8. The kernel loop is one series, not five squared.
      **Tests**: level invariance to 1e-12 at 1e3, 1e5 and 1e8 on deviations
      that are multiples of 2⁻²⁰, so the shift is exact (it measured the raw
      form first); the units test and the paper's null, size and power as
      they were; the definition test's longhand stays the five-moment form,
      so it now checks the identity between the two to 1e-12. **Review
      2026-09-25.** The accurate `D̂` reaches its true 0 on a collinear pair
      (`y = 2x + 3`), where the raw-moment form's noise stood in for it, and
      the numerator's rounding divided by it flagged every span (29 of 30
      seeds in the reviewer's replica; a derived or duplicated column in a
      bank would flag every span); a pair within 64 ε of `|ρ̂| = 1` now has
      no verdict, as a constant column has none
      (`a_collinear_pair_has_no_verdict`, which fails on the guard removed).
      Pinned since: T = 3 and 4, the bandwidth override, a constant column
      with an inexact mean, levels 4e9 and 1e12 (deviations on a 2⁻¹² grid),
      the level test to 1e-14 and the statistic itself at a level;
      `bandwidth` at `usize::MAX` no longer overflows. A span with one far
      outlier now gets the verdict the formula gives, a flag, where the old
      form cancelled to nothing (docs/REGIMES.md §4). Left: the kernel
      weights lag `l` by `1 − l/(γ+1)` (Newey–West) where
      docs/ANSWERS-E54-E64.md transcribes the paper as `k((t−u)/γ)`, `1 −
      l/γ`; O(1/γ), inside the size and power bands; to check against the
      paper once.
