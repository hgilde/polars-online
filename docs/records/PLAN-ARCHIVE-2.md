<!-- kind: record -->
# The plan archive, part 2 of 8

Moved out of `docs/PLAN.md` on 2026-11-14 (task 398), text unchanged: the plan had
grown to 1.7 MB, more than a reader can hold, and most of it was finished work.
This part holds the full entry of each finished task and the text of each closed
section, in the order they stood in the plan. `docs/PLAN.md` keeps one line per
finished task, linking the entry here (`#task-N`), and every section heading with
a line saying where its text went (`#sNN`).

Earlier text is in [part 1](PLAN-ARCHIVE-1.md).

Later text is in [part 3](PLAN-ARCHIVE-3.md).

<a id="s11"></a>
## 11. Task list

<a id="task-104"></a>
- [x] 104. **Window expressions as model targets, under an embargo --
      built 2026-10-03** (the review below first, then the build; the
      "Built" record is at the end of the task). Depends on 78 (done) and
      143; its tests read task 152's clocks. Touches clock events, so it
      got the extra review task 120 calls for before it was built.

      #### A target

      - **A target may be a task-143 expression** containing at least one
        forward operator: `targets=[(po.rewm_mean("mid", half_life="10s",
        window_size="1m") - pl.col("mid")).alias("fwd_mid")]`. One with no
        forward operator is known at its own row, so it belongs in
        `with_windows` as a column, and is refused. Several targets in one
        spec share one `X'X` and one embargo; a buy-side, a sell-side and an
        all-trades VWAP are three targets.
      - **A formula target goes everywhere a spec goes** (decided
        2026-10-02, superseding the same day's "1a"): the spec, the saved
        state and TOML carry it as task 143's compact tree, and the bank
        evaluates it with the embedded Polars. "1a" had made formulas
        Python-only, re-supplied on resume, because Polars' serialized
        expression is unstable across versions; the tree reads seven
        stable node kinds at build time and rebuilds through the public
        builders, so that limit is gone. Considered and not taken: Polars'
        SQL expression parser as a shared
        text form -- it parses the element-wise part in both languages
        (arithmetic, `CASE WHEN`, `LN`, a quoted placeholder column) but not
        the operators: Polars SQL's windows are row frames only (no `RANGE`,
        no `FOLLOWING`, no exponential weights), Python's `pl.sql_expr`
        refuses a function of ours and cannot register one, Rust's
        `polars-sql` registers functions only through a full query (its
        expression parser with a registry is private), and it would add
        `polars-sql` and `sqlparser`, statically linked (rule 12); and no
        forward targets on the command line.
      - **Each forward operator states its own `window_size`.** The spec
        states the embargo.
      - **An operator's `partial` applies**: a window cut short by a gap past
        `max_dclock` or a session change is learned from under `"keep"` and
        not under `"null"` or `"drop"`; a reset discards it, unlearned.

      #### The embargo

      - **`label_delay` is renamed `embargo`** in this task, everywhere,
        with no alias: the delay between a row and the learning of its
        target, as `po.stream.embargo` writes it out as data.
      - **The embargo is the only release, and nothing defaults it**
        (decided 2026-10-02). A row is learned once its embargo has passed in
        elapsed time -- the clock column's own steps, and `session_gap` where
        a session change restarts the clock (task 153) -- and never before its
        target is known; with no embargo, a row is learned where it sits once its
        target is known. The embargo means the same for every target, a
        plain column or an expression: a forward expression's window
        cannot close before *t* + w, so an embargo covering the window loses
        nothing the bank could have learned sooner, and it states the delay
        the bank keeps rather than leaving it implicit. Considered and not
        taken: a forward expression learned when its window closes with no
        embargo asked for, and an embargo counted from when the label is
        known.
      - **`fit_predict`** (and `lf.online.fit_predict`, the command line's
        run) **refuses a forward target whose longest `window_size` exceeds
        the embargo**, no embargo included. Row *j*'s target is known only
        at *j* + w, and a target learned at row *t* − 1 covers rows up to *t*
        + w − 2, which overlap row *t*'s own target window. With an embargo
        of at least w, the state that scores row *t* holds only targets
        whose windows closed before *t*.
      - **`fit` takes any embargo, none included, and ends in the same
        state**: the bank learns each row once its target is known, and the
        embargo only moves learning relative to scoring, each held row
        replaying the clock step it arrived with. (Measured on today's
        `label_delay`: a fit with a delay of 5 rows and one with none over
        the same rows less the 5 still waiting predict new rows identically,
        bit for bit, with and without decay.)
      - **No cap on the embargo** (decided 2026-10-02). A break releases
        nothing early (task 153): the embargo counts elapsed time, and a
        break's events wait with the row after it. So an embargo of any
        length is honest for every kind of target, and a forward window may
        be longer than `max_dclock`. The rule that refused an embargo longer
        than `max_dclock` -- written because a break released every held row,
        so a plain forward column with a one-hour embargo and a five-minute
        cap learned, after a ten-minute gap, labels eleven minutes old -- and
        its plain-column exception are gone.
      - **The held rows are bounded by one embargo of each group's rows**,
        and the docs state it: there is no row cap, so a denser stream holds
        proportionally more.
      - **`predict` does not read the embargo**: after a fit the state holds
        only targets whose windows closed before its input ended, so rows
        after that are scored honestly, and rows inside the training history
        are in-sample under any embargo. So no in-sample prediction is ever
        emitted.

      #### How its rows move

      - **Each row is scored and emitted as it arrives.** It is held, with
        only its own columns the target expression reads, until the window
        core resolves it.
      - **The bank releases a row when its embargo has passed and the
        window core has resolved it**: the window closes, a gap or session
        change cuts it, or a reset discards it. The target expression,
        rebuilt from its tree (task 143), is evaluated by the embedded
        Polars on the rows each chunk resolves, then held until the embargo
        has passed. With an embargo of at least the window the
        embargo is the later of the two: with no break between, the core's
        steps are the clock column's, so elapsed time reaches *t* + w no
        sooner than the core does, and a break cuts the window. A break's
        events -- the lag rings' clear, the blend -- wait with its row and run
        when it is learned (task 153; docs/REVIEW-E54-E64.md L2). The case
        the review was to settle, a run of skipped rows the bank read as a
        capped gap and the core did not, is gone with the early release.
      - **The window core sees every row with a value**, including rows the
        spec skips for a null feature: a target depends on the prices ahead,
        not on whether the model could use a row's features. Clock events on
        skipped rows count (task 79).
      - **A row unresolved when the input ends** is never learned; with
        `save_state` it waits in the state for the next run.
      - **`group_close` is refused** with a forward target, as with
        `embargo`: a closed group cannot release the rows it holds.
      - `SCHEMA_VERSION` bumps (rule 5).

      #### The column shows the target the model used

      `po.stream.with_windows(lf, [expr], like=spec)`, with `expr` a target
      of `spec`, equals on every row the model learns from the target it
      learned, to the bit, and is null on every other row: rows a reset
      discarded, rows unresolved at the end of the input, and rows the spec
      skips. `like=spec` carries the spec's clock policy and its rule for the
      rows it learns from -- features and weight usable. `complete` marks a
      cut window; null marks a row the model did not learn from.

      #### Tests

      - Interleaved trades and quotes: three VWAP targets (all trades, buys,
        sells) equal to the column form fed back as plain targets under the
        embargo.
      - Parity, row by row, with skipped trade rows, on streams built to
        contain each event on a skipped row and on an accepted one: a gap
        over `max_dclock`, a session change with a finite `session_gap`,
        `session_gap="reset"`, a step back past `min_backwards_jump` under
        `"reset_state"`, several groups whose clocks restart together, rows at
        one timestamp, a row exactly one window later, a run of skipped rows
        totalling more than `max_dclock`, and the end of input inside a
        window.
      - `fit_predict` refusing an embargo below the longest `window_size`
        and none; `fit` taking both, ending in the state of an embargoed
        fit; `predict` unaffected; a target with no forward operator
        refused.
      - The embargo seen through task 152's clocks: on every row the scored
        row's clock less the learned row's is at least the embargo in elapsed
        time, and the next row the bank holds is less than one embargo back,
        across gaps and session changes alike.

      #### Reviewed before building (2026-10-03)

      The review task 120 asks for, done against the code of task 143's core
      and the bank's `apply_label_delay` (task 153), event by event. The
      names are task 144's (`gap_cap`, `restart_after_step_back`, `embargo`).

      - **One clock, two walkers.** The bank steps each group's
        `ClockState::advance` with the spec's `ClockCfg`; the core steps
        the same `advance` with the same cfg per group *and* once more on
        the stream's own clock across groups. Every event the bank sees --
        a gap past `gap_cap` (`capped`), a session change with a finite
        `session_gap` (`capped` or not by the gap's size) or `"reset"`
        (`reset`), a step back past `restart_after_step_back` (`reset`), a
        late row (refused) -- the core sees on the same row with the same
        verdict, because it is the same function on the same arguments;
        only `accept` differs (the bank folds a skipped row's time into the
        next accepted row, the core counts every row), and `accept` never
        changes a verdict. So a window cut or discarded by the core is cut
        or discarded exactly where the bank's `held_break` or `reset` lands.
      - **One core per group, as one stream per group.** `with_windows`
        runs one core over every group with a clock of the stream's beside
        each group's, which refuses a row of group B earlier than group A's
        last and cuts every group's open windows at a gap on the stream --
        neither of which the bank's per-group clocks see. The bank keeps a
        core per group instead, fed that group's rows: each group's clock
        is the spec's clock as its stream runs it, a stream interleaved on
        clocks of its own runs (as it does for a plain target), and a
        group's rows resolve in their own order, so a resolution is known
        at the row that made it. (Found in the build: one core over every
        group emits a row only once every row before it, of any group, has
        resolved, and that wait moved with the chunking.) A gap on the
        stream reaches each group at its next row, where its own clock
        shows the capped gap, and the cut value is the same -- the window
        saw no row of its group after the gap either way.
      - **Release is the later of the two, and chunk order decides it.** A
        resolution is applied in row order -- at the row that closed, cut
        or discarded the window, never earlier -- because with `embargo`
        equal to the window under `closed="right"` a row exactly one window
        later has the embargo run out while the window is still open (that
        row is a member); applying the chunk's resolutions up front would
        release it a row early in a coarse chunking and not in a fine one.
        Each resolution therefore carries the sequence number of the row
        that made it, and a pending row is released when its wait has run
        out *and* a resolution at or before the current row has reached it.
        Within a group, pending rows resolve in arrival order (windows close
        in clock order, a cut or a discard takes every open window, rows at
        one stamp share one), so the resolved rows are a prefix of the
        buffer and the release stays the `take_while` it is.
      - **A refusal leaves the bank as it was.** The core is fed after the
        clock check and before the streams, on a snapshot that is restored
        when the chunk is refused later (its own refusal, or the window
        pre-pass); the pre-pass replays with the chunk's resolutions.
      - **`fit` and `fit_predict` are one Rust method**, so the refusal of
        an embargo below the longest forward `window_size` is a bank flag
        the Python `fit` sets for its run (`set_learn_only`), computed per
        spec when the bank is built; the command line never sets it.
      - **Decided here, where the design above was silent:** a formula
        target's columns reach the bank as numbers or text (a temporal
        column other than the clock is refused, as for every role); its
        columns may be features, since the leak check guards a *plain*
        target's column read at the same row, and a forward window over
        `mid` is not the row's `mid` (`closed="left"` or `"both"` puts the
        row's own value in the window, which is the user's choice);
        `resid_<name>` is null on the scored row (the target is not known
        then; the diagnostics fold at release as under any embargo); for a
        target `"drop"` means what `"null"` means (not learned from), since
        a row of a model is scored and cannot leave; a row unresolved when a
        run ends stays in the state with the core's rows, so a resumed run
        learns it when its window closes; a row the spec skips is fed to the
        core with its value and is never pending; `SCHEMA_VERSION` 23 with
        the bank's minimum left at 22 (a 22 file has no formula target and
        loads as it was).

      #### Built (2026-10-03)

      - **A target is a `TargetDef` with a `formula`** (`targets.rs`): a
        table `{name, formula}` on every surface -- the spec's JSON, the
        state file, the CLI's TOML (`{ name = "fwd", formula = [...] }`) --
        refused without a name, with a column too, or with no operator
        looking ahead. `TargetDef::columns()` is what every reader of a
        target's columns now asks (the chunk's casts, the first-reader
        roles, the clock clash); `value_column()` is `None` for a formula,
        so the leak check passes its columns as features by design. In
        Python a `pl.Expr` in `targets` becomes the table (`_spec.py`,
        `formula_target`), `po.FormulaTarget` is its type, and
        `target_columns` walks the tree for the plan's projection.
      - **The bank resolves it** (`bank.rs`, `TargetWindows`): per spec,
        one `WindowsRun` per group over the formulas under the spec's
        clock policy (no `like=`: the core sees every row), fed each
        chunk's rows group by group in row order after the clock check and
        before the window pre-pass and the streams, on a snapshot restored
        when a later check refuses the chunk. `WindowsRun::feed_resolving`
        returns every resolved row with the formulas' values (null where
        a `"drop"` operator was partial), its number in the bank's stream
        (`@po:row`, from `rows_fed`) and the number of the row that
        resolved it, read off the core's ready count after each push. The
        chunk's resolutions reach each stream as `FormulaTargets` -- the
        formula slots, each row's number in the laid-out order, and its
        group's `Resolutions`, flat (row numbers, resolving rows, values
        row-major: no allocation per row; a map of a `Vec` per row cost
        the first build 1.7 times the column form) -- and
        `apply_label_delay` releases a row when its wait has run out *and*
        a resolution made at or before the current row has reached it,
        found by binary search over the resolved prefix of the buffer at
        every row of the group, skipped rows included (a skipped row
        closes windows as any row does, and the next accepted row may be
        chunks away: with one row a chunk, resolutions made at skipped
        rows were lost and the buffer stalled behind them); with no
        embargo the buffer runs with a wait of 0. `PendingRow` gains
        `seq` and `resolved`, both defaulted.
      - **`fit_predict` refuses, `fit` takes**: the longest forward
        `window_size` against the embargo is computed when the bank is
        built and refused at the first `fit_predict` unless
        `Bank::set_learn_only(true)`, which `ModelBank.fit` sets for its
        run; the CLI never sets it.
      - **The state carries the cores**: `BankFile.resolvers`, per spec
        and group, each `WindowsRun::save_bytes`; a loaded bank resumes
        each at its group's first chunk, which says what the columns are.
        Schema 23 (24 since review R4 and 25 since review R6, which changed the window core's state).
      - **Tests**: `crates/online-polars/tests/formula_targets.rs` (the
        column form fed back, bit for bit, in 1, 7 and 600 chunks; the
        short-embargo refusal and `fit`'s equal predictions; a state saved
        mid-window; groups on clocks of their own equal to each group run
        alone; what the spec refuses) and `tests/test_formula_targets.py`
        (the same parity on the three VWAPs over interleaved trades and
        quotes and through every clock event of the review -- a gap past
        the cap, a session change with a finite gap and with `"reset"`, a
        step back the policy restarts on, groups restarting together,
        rows at one stamp, a row exactly one window later, a run of
        skipped rows longer than the cap, the end of the input inside a
        window -- each on a skipped row and an accepted one, in one chunk
        and seven; the row exactly one window later under an embargo equal
        to the window, the one place the two forms part; `fit_predict`'s
        refusal and `fit`'s acceptance on every surface; `predict`; the
        embargo and the window seen through task 152's clocks; a state
        saved mid-window; the plan's projection; the TOML form; a formula
        beside a plain target).
      - **Measured** (2026-10-03, PERFORMANCE §34): the forward VWAP of
        §33 learned natively takes 1.5 times the column form's wall time
        at 4M rows (2.37 against 1.57 s) and 1.6 at 16M (9.37 against
        5.87), with less memory (1.09 against 1.23 GB). The first build
        was 1.7 times (a map of a `Vec` per resolved row; flattened).
        What remains is the overlap the query form gets for free: its
        window and bank are two sources the streaming engine runs as two
        stages, and its 1.57 s is below the 2.08 s of `with_windows`
        alone and the model alone run one after the other, where the
        native path runs both in one source, 0.29 s (14%) above that
        sum. One polars thread moves none of the four, so the overlap is
        the engine's source tasks. Closing it needs the next chunk before
        the current one returns, which a pull-based source has not got;
        the groups' cores on the pool would gain nothing on one group.
        Neither is built; the user's call whether the native form's
        convenience is worth the 1.5.

<a id="task-105"></a>
- [x] 105. **`po.prep` renamed `po.stream`, and the rules every function in
      it follows — split from task 78 on 2026-09-25; built 2026-09-28** (the
      user: "Do 105"). The rename (105a), and the rules applied to the two
      functions there are: both give back the kind of frame they are given
      (rule 1; `embargo` returned a `LazyFrame` for a `DataFrame`);
      `refresh_time` takes `clock=` and `group=` and refuses a step back, as a
      spec's default `on_clock_reset` does, comparing a temporal clock in
      nanoseconds (rule 2; task 120 settled its clock); and it resumes (rule
      5), with `load_state` read when the plan is built and `save_state`
      written once the input's last row is fed, atomically, a slice of the
      output not stopping the input early. Its state is its own file
      (`RefreshFile`, versioned msgpack with `names` and `pairs`, which a
      load must match), with a temporal clock kept in nanoseconds so it
      resumes on any unit. Rules 3 and 4 are the windows' (task 78). The user,
      2026-09-11: `po.prep` is to be renamed and its API defined "the way
      it should be done" — pre-1.0, so no aliases and no compatibility. No
      behaviour change; the one breaking change queued, so it belongs in a
      minor release. **The clock of `refresh_time` under the one vocabulary
      needs extra review** (task 120). The passages below moved from task 78
      word for word.

      **`po.prep` becomes `po.stream`**: the namespace for streaming
      transforms of a plan — `LazyFrame` in, `LazyFrame` out, O(state)
      memory, before or beside a bank. "Stream" is already a defined word in
      the README's glossary. `embargo` and `refresh_time` move; nothing
      stays behind under `po.prep`.

      Every function in `po.stream` follows five rules — what "the way it
      should be done" means here:

      1. **Frame first, the same kind back** — `LazyFrame` in gives a
         `LazyFrame`, `DataFrame` gives a `DataFrame`, as `po.fit_predict`
         does. Chaining is Polars' own `lf.pipe(...)`, and
         `with_windows` is also `lf.online.with_windows` and
         `df.online.with_windows`, as `with_columns` is a method.
      2. **One vocabulary**: the clock and its policy -- `clock`,
         `max_dclock`, `on_clock_reset`, `session`, `session_gap`, `group`
         -- have a spec's names, defaults and meanings, run by the same Rust
         clock code; a window operator takes Polars' names (task 144). `refresh_time`'s `time=` becomes
         `clock=` and `by=` becomes `group=`. With no `clock`, one unit is
         one row.
      3. **What the rows share goes on the call; what a window decides goes
         on the window.** The call takes the clock and its policy — `clock`,
         `max_dclock`, `on_clock_reset`, `session`, `session_gap`, `group` —
         and the state; each window operator takes its own input and
         parameters (task 143).
      4. **Output names as in `with_columns`**: from keywords and `alias`.
         Two outputs with one name, or an output named like an input column,
         are refused while the plan is built.
      5. **Stateful transforms resume** with `load_state` / `save_state`;
         `chunk_rows` as everywhere. `embargo` is a pure Polars plan
         (`merge_sorted`) with nothing to save and takes neither.

      #### Sub-tasks

      - [x] 105a. (was 78a) **The rename**, alone, first: `po.prep` →
            `po.stream`, `time=` → `clock=` and `by=` → `group=` in
            `refresh_time`, its tests, the reference page, the README,
            `llms.txt`, and the API surface file. No behaviour change, so
            every existing test passes with only its spelling changed.

<a id="task-106"></a>
- [x] 106. **Discarded by the user on 2026-09-25: a backward clock is not a
      session change.** "We no longer want to assume that a backward clock
      is a session change, this is an old idea that should be discarded."
      Not to be built or revived: `on_clock_reset` keeps its four policies,
      `session_gap` still needs a `session` column, and a step back stays
      what `min_backwards_jump` and `on_clock_reset` make it. The design
      below is kept as the record of the idea, split from task 78 on the
      same day, word for word.

      **A backwards clock is a session break.** The user's rules,
      2026-09-11: "When a forward window encounters a backward clock, we
      want the same effect as if there were a large clock gap where the
      window just runs out", and then "If a clock moves backward it must be
      a session break or a problem in the input data and we can't tell
      which so we assume session break." And then, for the decay too: "the
      decay should take the session gap as well since that would be needed
      anyway to completely recreate the target used by the model." Read against
      `ClockState::advance` (`crates/online-core/src/clock.rs`) and
      `apply_label_delay`, per `on_clock_reset`:

    | policy (today) | a waiting row today | as a session break |
    |---|---|---|
    | `"max"` | `capped = true` — the buffer releases, the window ends | unchanged |
    | `"reset_state"` | the buffer is discarded; the model starts over | unchanged — a reset, as `session_gap="reset"` is at a session change |
    | `"zero"` | a zero step, not capped: the buffer keeps waiting, and a window would run on across the jump, counting the rows after it as the row's own clock | **the buffer releases and every window ends** |
    | `"error"` | the run stops | unchanged |

      - **In both forms, a backward step on a group's clock does to the
        waiting rows what a session change does**: every open forward
        window of the group ends — no row at or after the jump enters,
        the value is over the rows before it, under `partial` — and the
        `label_delay` buffer releases in order. One line in
        `apply_label_delay`: `plan.backwards` joins `session_changed` and
        `capped` in the release condition.
      - **The decay takes it too: a backward step is a session change,
        everywhere.** `ClockState::advance` treats a raw step below zero as
        it treats a change of session value: `session_changed` is set, the
        step is `session_gap` (limited to `max_dclock`), and
        `session_gap="reset"` starts the model over. So everything that
        follows a session change follows a backward step — the decay, the
        `label_delay` release, the lag rings cleared, `session_shrink`'s
        blend, `group_close="session"` closing the group, and the windows
        ending. One code path, which is what lets the column form recreate
        the target the model learns from exactly.
      - **`on_clock_reset` shrinks to `"session"` (the default) and
        `"error"`.** The three policies it loses are each a `session_gap`:
        `"max"` is `session_gap = max_dclock`, `"zero"` is `session_gap = 0`,
        `"reset_state"` is `session_gap = "reset"`. Nothing is lost except
        choosing a different step for a backward clock than for a session
        change — which the rule says cannot be told apart anyway. `"error"`
        stays, for input where a backward clock can only be a bug.
      - **`session_gap` without a `session` column**, now meaningful: it is
        the step at a backward clock. With no `session` column it defaults
        to `max_dclock` — today's default decay step at a backward clock
        (`"max"`), so a stream with no session column and no `session_gap`
        decays exactly as before. With a `session` column it stays required,
        as now.
      - **What moves.** Pre-1.0, so no compatibility spelling: the three
        policy words are refused with a message naming the `session_gap`
        that replaces each. Numbers move for a stream with a backward clock
        under `"zero"` or `"reset_state"` (now session changes: a released
        buffer, cleared lag rings), under `"max"` where a `session` column
        gave a different `session_gap`, and for any spec with
        `session_shrink` or `group_close="session"`, which now also fire at
        a backward step. `SCHEMA_VERSION` bumps (the clock config's layout
        changes), shared with 78e's bump if they ship together, with a
        loader that maps a saved `"max"` / `"zero"` / `"reset_state"` to the
        equivalent `session_gap` when the file has none. A minor release,
        with the list above as its first CHANGELOG lines.
      #### Sub-tasks

      - [x] 106a. (was 78b, discarded) **A backward clock is a session
            change**, in `ClockState::advance`, before any window is built:
            every model's decay, `label_delay`, lag rings, `session_shrink`
            and `group_close` follow from it; `on_clock_reset` becomes
            `"session"` / `"error"`; `session_gap` allowed without
            `session`, defaulting to `max_dclock`; the README's clock
            section, the reference docstrings and every test that names a
            removed policy. Its own commit, its own CHANGELOG lines, since
            it changes numbers the windows do not depend on.
      #### Tests

      - A backward step and a session change at the same row, for every
        model (moved from task 78's parity test):
        **A backward step equals a session change at the same row, for
        every model**: one stream with a backward clock and no `session`
        column against the same stream with a `session` column that changes
        exactly there and the clock made monotone, identical output field by
        field — decay, `label_delay`, lag features, `session_shrink`,
        `group_close="session"`. Each removed policy word refused with the
        `session_gap` that replaces it; a v6 state saved under each loading
        with the equivalent gap. For `"reset_state"` and
        `session_gap="reset"`, the rows with `complete` false are exactly
        the rows the model never learned from.

<a id="task-107"></a>
- [x] 107. **Relative targets -- built 2026-09-25 for plain columns;
      windows take it as arithmetic in task 143.** A level -- a price, a
      VWAP -- is rarely what a regression should predict; where it goes from
      now is.

      - **A plain-column target** takes `relative_to=`, a column read at the
        target's own row (the mid, the last trade), and `relative=`:
        `"difference"` (default, `y − r`), `"ratio"` (`y / r`) or
        `"log_ratio"` (`ln(y / r)`): `po.target(column, *, relative_to,
        relative, name)`, and in TOML a table, `targets = ["ret_5m", {
        column = "price_5m", relative_to = "mid" }]`.
      - **No look-ahead is added**: `r` at row *t* is known when row *t*
        arrives. A null `r`, or for the two ratios a non-positive `y` or
        `r`, makes the target null on that row: not learned from, never a
        NaN in the state (hard rule 9).
      - **Everything downstream is on the relative scale**: `pred`, `resid`,
        `sigma`, the metrics and the conformal interval, so `hit_rate` is
        the direction of the move. The output does not add `r` back.
      - **A window target is relative by arithmetic** in task 143's
        expressions: `- pl.col("mid")`, `/ pl.col("ask")`, `(op /
        pl.col("bid")).log()`, in the column form and the target form alike.
      - **`"log_ratio"`** stores an `f64` like any target. `ln` can differ
        in its last bit between glibc and Apple's libm, so a log-ratio
        target is reproducible across OSes to the last few bits, and no
        `"log_ratio"` spec goes into a byte-identical cross-platform fixture
        (`state_schema*.rs`, the release workflow's write-on-macOS,
        read-elsewhere job).

      #### Tests

      - `po.target("p", relative_to="r")` against the same spec given
        `p − r` as a plain column, identical field by field, for each of the
        three ways, in `ewridge`, `kalman`, `huber`, `holt` and `marginal`
        (`tests/test_relative_targets.py`); a null `r` and a non-positive
        value giving a null target that is not learned; the TOML table
        loading to the same spec as the Python one.
        **Review 2026-09-26.** The IO plugin's projection set was built
        with `set.update(spec["targets"])`, so a table target -- relative or
        merely renamed -- raised `unhashable type: 'dict'` on every
        LazyFrame path, and a fix by name alone would have projected the
        reference away (D1, F1): `target_columns` gives a target's column
        and its reference, and `_spec_columns` keeps both
        (`test_the_lazy_plan_reads_a_relative_targets_columns`). `sgd` under
        `"logistic"` and `"poisson"` took a relative target `ftrl` refused
        (D2): refused by name, and `ftrl`'s guard names its default loss
        rather than everything but `"squared"`. A ratio target is positive
        by construction, so `hit_rate`'s sign test read 1.0 whatever the
        fit (D3): the hit test is about 1 for a ratio, `SlotMetrics::update_about`,
        pinned by a recursion recomputed from the output
        (`test_hit_rate_under_a_ratio_target_is_about_one`). The `bocpd`/`hmm`
        slot check compared names, so `{"column": "z", "name": "h"}` put
        column `z` in the hazard slot (D4): the slot must be the plain
        column; and the message printed `Targets`' derived `Debug` (D5).
        The leak check matched a target's *name* against the features and
        refused a target merely named like one (D7): the column alone. An
        empty `name`, `column` or `relative_to` was refused nowhere (D
        missing 5): refused by the builder and by the bank. The builders'
        `_matches` did not read a TypedDict, so a wrong table entry reached
        serde and was named by JSON path (F4), and a list's entries escaped
        the int floor (F7). Docs: a result the model cannot use is null too
        (D8); the pairs table and `describe` carry the *name* (E6); a table
        naming its own column comes back as the column (F6). New tests: the
        equality under a group, a label delay, both, a window and the
        diagnostics with the reference null on row 0; `predict` before any
        fit and with either column dropped; the specs and the JSON carry the
        table; a saved bank held to its targets' `relative`; a renamed table
        is the name in the fields, the Gram, `describe`, `summary`,
        `marginal` and a closed group's pairs; two views of one column; the
        role tables of the error-message suite name `relative_to`.

**The plan of 2026-09-25.** Three audits compared every open list with the
code (`docs/PLAN.md`, the other design docs, the release checklist, CI and
GitHub). The user's call on the result: everything that integrates a new
library or Arrow is parked (below the tasks); everything else is on this
plan to be reviewed and, the user expects, done now; and any work on rows
out of order or a clock that goes back **needs extra review before it is
built**: task 120 (what an audit of that area found), 104, the clock rules
of 78 and the clock of 105's `refresh_time`, and the items marked so below.
Task 106, the idea that a backward clock is a session change, was discarded
by the user the same day. In tests, any library under an open-source licence
may be used (task 121), and only licensed ones -- source-available or
commercial terms -- are parked (the user, the same day). Each task says its
size (S under a day, M days, L a week or more), what it waits on, and the
decision it needs, with a recommendation where there is one.

<a id="task-108"></a>
- [x] 108. **The docs say what the code does.** S–M, no code, no decision.
      **Done 2026-09-25:** a survey checked each item against the code,
      and 107 edits followed, each an exact match of the stale text: the
      removed surfaces marked as removed where they read as live (a note
      heads ENHANCEMENTS and IMPROVEMENTS for the done rows that name
      them, and code comments drop them); the Arrow import corrected to
      unbuilt and parked in ARROW-SOURCES, task 86's head and the two
      reviews; clustering and B2 recorded as built; the counts recounted
      (976 Rust tests, 3,265 pytest cases), the schema 16 and the tested
      Polars versions dated; the README's `min_periods` sentence replaced
      by a table of the defaults `Spec::default_min_periods` gives;
      WARMUP-AND-CONVERGENCE held to what was built for `ewridge`;
      REVIEW-2026-09-18's six "candidate" statuses closed with batch 3's
      commit and tests, and the 2026-09-12 review's open V entries struck
      with their closures; ANSWERS' D̂₃ corrected to the code's gradient;
      IMPROVEMENTS P3 noted as done by PERFORMANCE P11; TESTING's T-W rows
      marked, with T-W8's case-insensitivity the one open entry; and this
      plan's status line, task 61 split from task 68, §3 and §9.
      The audits found text the code has left behind: the removed expression
      plugin (task 85) and `po.run` (task 83) described as live in
      ENHANCEMENTS, IMPROVEMENTS, PERFORMANCE, CLUSTERING, STATE-WORKFLOW and
      `runner.rs`'s comments; the Arrow import called blocked (ARROW-SOURCES
      §4 and the head of task 86, both retracted 2026-09-22), and the two
      reviews that repeat it; clustering and B2 (built as `window`) recorded
      as undecided (ENHANCEMENTS, CLUSTERING §12, BEYOND-O-STATE, PLAN §11h);
      stale counts ("650 Rust tests and 2,200 Python cases"), schema numbers
      and the tested Polars version; the README's rule that every model but
      `ewridge` defaults `min_periods` to one per unknown, which is wrong for
      seven models; WARMUP-AND-CONVERGENCE's stale passages; the statuses of
      REVIEW-2026-09-18 (B1, S1, D1, P1, S4, D2 still "candidate") and of the
      2026-09-12 review's V-list; ANSWERS-E54-E64's D̂₃ marked "Verified";
      IMPROVEMENTS P3 "rejected" where PERFORMANCE P11 is done; TESTING's
      "Open entries: None" beside unmarked T-W rows; this plan's own status
      line ("tasks 1–58 done, released as 0.2.0"), task 61 fused onto task
      68, and the §3 table and §9 item 4 listing the clock policies.

<a id="task-109"></a>
- [x] 109. **Release 0.11.1** (planned as 0.11.0). S each; the push and
      the approval are the user's. **Tagged `v0.11.0` on 2026-09-26** on the
      user's word ("merge, push and tag the next minor"), after the review of
      task 132: the version in the six places and both locks, the
      comparison against 0.10.0 measured again (21 of 30 specs, 130 fields,
      median 3e-16, at most 1.4e-14), the CHANGELOG promoted. **Never
      published** (below); **the release is 0.11.1**, dispatched by the
      workflow of task 133, with the same changes and the fixes below. Not done: the suite at the Polars floor 1.34.0
      (last run 2026-09-02), the glibc floor, the 0.10.0-wheel state test,
      S6 and task 105. **The tag's first run failed, and nothing was
      published** (2026-09-27): the rehearsal on `main` (RELEASE-READINESS
      step 2) was skipped, and the branch's 21 commits reached CI for the
      first time on the tag. Two tests, both the tests' fault:
      `the_bandwidth_override_reaches_the_kernel` (task 103) failed on Linux
      and Windows -- at a bandwidth of `usize::MAX` the long-run variance is
      rounding, its sign set by Box–Muller's libm bits, and where it rounded
      to 0 the model said NaN (no verdict) and the longhand, folding with
      `f64::max`, said 0.0; it runs thirty streams now and takes the model's
      NaN rule (reproduced here at seed 13). And the licence check (task
      121) read sphinx's metadata where CI's `uv sync` installs `dev` alone:
      `docs` is a default group now. Because `cargo test` stopped at the
      first failing binary, Linux and Windows ran none of the integration
      tests and Windows no pytest; CI passes `--no-fail-fast` now. The
      Mutants jobs failed on the same unit test in their baseline. The gate
      of that fix found **G4** in the contract's proptest: `deco`'s
      `loglik` was `-inf` after rows weighted `0.01, 0.01, 1e100` and one at
      `1e100` (a spread near `5e-52` standardised the row to `2e151`, the
      density's `ρ` sat `1e-9` from singular, the quadratic form reached
      `1e311`); it is NaN now, no reading, as `hmm`'s is
      (`a_log_density_past_the_range_is_no_reading`). A randomly seeded
      proptest that finds such cases can fail CI at random, so the fix was
      followed by more runs before any push, and they found **G5**: rows at
      `1e100` before a window and one of weight `1e100` inside it left the
      live co-moments near `1e100` with the window's spread below their
      last digit, the subtraction returned noise near `1e84`, and the
      windowed `lasso` predicted `-inf`. `crate::truncated` now reads a
      variance no larger than `64ε` of the terms it is formed from as no
      spread, zeroing its row and column as a held feature's are
      (`a_window_the_live_state_cannot_resolve_reads_as_no_spread`, which
      read `1.3e84` without it, and
      `a_window_the_live_state_cannot_resolve_through_the_bound`). The tag
      could not be moved -- `v*` tags are immutable, by the ruleset -- so
      the user decided (2026-09-27) that the release ships as **0.11.1**
      and that the workflow creates the tag after everything runs (task
      133). Before the tag, the list as it stood:
      - Refuse `max_error_inflation` by name on a model that ignores it:
        today it is range-checked and dropped on every model but `ewridge`,
        where `emit_error_inflation` is refused (spec.rs:2417-2433).
        *Decision: refuse (recommended) or document it as ignored.*
      - **Done 2026-09-25.** `EwDiag::update`: drop the zero-weight loop,
        a no-op (`a = 1`, `b = 0`), which also removes an equivalent-mutant
        entry whose line matches the other branch too, where a real
        survivor would hide. Only finite values ever reached it: the
        stream skips a row with a missing feature before a model sees it.
      - **Done 2026-09-25.** The window test oracles keep `age < window`
        where the models keep `≤` (test_window.py:219, ewcov.rs, ewridge.rs,
        marginal.rs tests). None had a row exactly one window old, so none
        could tell: the Rust oracles now step the clock in quarter units,
        which land rows on the boundary (12 to 20 reads each, counted and
        asserted), and failed under `<` before the fix. The Python ridge
        oracle's data were noise-free, which hid it the same way; a noisy
        test now puts the model 6.6e-16 from the inclusive fit and 2.2e-3
        from the exclusive one.
      - **Done 2026-09-25.** `po.sim` in the API snapshot, which now reads
        the package's modules from its directory rather than a hand-kept
        list; FTRL's `zz < 0` equivalent mutant recorded.
      - The suite at the Polars floor, 1.34.0 (last run 2026-09-02); the
        Linux CLI binaries' glibc floor before GitHub's Ubuntu 26 runners
        (2026-10-19).
      - The release comparison against 0.10.0 again (the CHANGELOG's figure
        predates tasks 102 and 103); the version (RELEASE-READINESS.md:150's
        places, both locks); the CHANGELOG promoted; the README's pin.
      - *Decision, the schema:* keep the loaders for 14 and 15 that the
        CHANGELOG promises (recommended, with a test that loads a state
        written by the released 0.10.0 wheel), or raise the minimum to 16.
      - *Optional:* S6, the stream's persisted fields as one sub-struct,
        cheapest while schema 16 is unreleased.
      - *Recommended:* task 105 in this release, the one breaking change
        queued, so the API breaks once.
      - Then the user's: fast-forward `main` and push (the first CI on
        Linux 3.13/3.14, macOS 3.14 and Windows; the Mutants job will likely
        time out on a 3,700-line diff and does not gate a release), the
        rehearsal, the tag, the upload's approval; then a clean-venv install.

      **Review 2026-09-26.** `max_error_inflation` is refused by name
      on a model without a ridge system, as `emit_error_inflation` was
      (`test_is_refused_by_name_where_no_ridge_system_reads_it`). The
      schema: the loaders for 14 and 15 stay, as the CHANGELOG promises; the
      test that loads a state written by the released 0.10.0 wheel is not
      built. Task 105 is not in this release: its clock rules need the
      user's review (task 120). **2026-09-27, after 0.11.1:** the state test is
      built, `tests/test_released_state.py`, for 0.10.0 (schema 14) and
      0.11.1 (schema 17); `scripts/release_probe.py --states` writes the
      files under the released build (RELEASE-READINESS, "Compare with the
      last release"). The Linux CLI's glibc floor is measured: 2.39, a
      decision now (task 115 (i)). The suite at the Polars floor, 1.34.0, ran
      again: 3,408 of 3,425 passed, and the one failure in the package,
      `po.eval.seqtest` with `by` (a window inside a window, which 1.34.0
      refuses), is fixed; the rest are `scan_arrow_c_stream` (1.43.0 on,
      now said in ARROW-SOURCES), two duration oracles and the two version
      pins (RELEASE-READINESS, "The floor and the ceiling").

<a id="task-110"></a>
- [x] 110. **This week's review leftovers, as tests.** S. Held-value tests
      for `ew_class` and `hmm`; weight-0 rows for `bocpd`, `deco` and `hmm`;
      each model's state saved before the low parts, loaded; `ew_class`'s
      window reading a held feature; two targets under pairwise gaps with a
      held feature (task 101's review). **Done 2026-09-27.** In
      `held_values.rs`: `ew_class` classifies and `hmm` filters the same at
      every level (measured a thousandth of `steps_of` and less; `hmm`'s
      state spread decays at its share of the rows, 2.7e-22 of the stop,
      so it is held to level 0.5's rather than to `ew_cov`'s 1e-30); a row
      of no weight moves nothing in `bocpd`, `deco` or `hmm`, to the bit;
      two targets under `pairwise` keep their slopes at every level -- the
      second target's slope wanders under `pairwise` at every level,
      level 0.5 included, from before the feature stops (the all-row Gram
      against a third fewer cross-moments), so it is held to 0.5's. In
      `model_contract.rs`: `every_model_resumes_from_a_state_without_the_low_parts`
      drops every `_lo` field of every model's state, in msgpack (`rmpv`,
      a test-only dev-dependency, MIT: serde_json turns an infinite cfg
      value into null), and holds the continuation to the whole state's
      within 1e-11 (3.7e-13 measured), with each of the 15 models that
      keep a mean shown to have dropped parts; and
      `a_schema_16_state_of_offsets_loads_as_own_means`, the G3
      conversion's first test on a real schema-16 shape. In `ewclass.rs`:
      `the_window_reads_a_held_feature_as_no_spread_in_every_class`.

<a id="task-111"></a>
- [x] 111. **The model contract, checked where it is only listed.** S–M.
      EXTENDING's checklist misses four places a test checks (`BUILDERS`,
      the OUTPUTS meanings, llms.txt's count, the README's API lines);
      `ewridge`, `sgd` and `kalman` lack a wrong-shape refusal test; the
      bit-flip fuzz covers one spec; the save/load and realized-struct
      sweeps cover the regressions only; lists kept by hand stand in for
      checks (test_bank.py:330, test_coef.py:114-141). **Done 2026-09-27.**
      EXTENDING names all four and the new sweeps. Every model module has
      its wrong-shape test (`sgd` and `kalman` check in their `TryFrom`, so
      theirs go through the encoding). `summary.rs`'s
      `every_model_kind_refuses_or_loads_a_corrupt_file_never_panics` cuts
      and bit-flips a state of every kind (thinned: every 7th prefix, every
      5th byte), held to `ModelKind::KINDS`: no kind panics. `tests/test_every_kind.py`
      runs the realized-struct and the mid-stream save/load checks over the
      registry's `MINIMAL` itself -- `rcov`, `hmm`, `corrchange` and `bocpd`
      had neither. The hand lists: `test_coef`'s `COEF_SPECS` gained the five
      kinds with a `coef` it missed (`quantile`, `kmeans`, `ew_class`, `hmm`,
      `deco`) and a check against `coef_index`; the sparse-target test runs
      every regression model, so its per-target set is checked by the
      others; `test_error_messages.BUILDERS` gained the seven builders its
      inf and NaN sweeps never saw, held to `MINIMAL` (all seven were
      right). `test_kwargs_typing.BUILDERS` stays a list -- deriving it would
      import the registry that imports it -- and is held by
      `test_the_builder_list_covers_every_builder`, which EXTENDING now names.

<a id="task-112"></a>
- [x] 112. **Older review items that need no decision.** S each. C16's raw
      oracles; C21's r2 and coverage under a delay; S23's docs; D9, the dead
      `LassoCfg::combo_labels`; REVIEW-2026-09-18 §6 (a constant column in
      `deco`, `bocpd`'s 1e5-row truncation and its gaussian at d ≥ 3); T2's
      roll call on real states (4 of 20 kinds); U5, a Decimal parquet
      through the CLI (its test went with task 83); REGIMES: `hmm` through a
      regime switch, `deco` and `rcov` experiments, WKD's size at t5, the
      Epps sweep that ends at its best point; PERFORMANCE: the measurements
      owed (prefetch limits, §19's state sizes, §21's trivial scan, task
      101's cost) and a re-run of its timings. **The tests and fixes done
      2026-09-27, and the rest the same day.** REGIMES (run on 2026-09-27,
      every earlier figure unchanged): `hmm` through a switch holds each new
      state in a median of 12 rows from the true covariances and follows
      none seeded from the rows; `deco` settles at `E[u]`, two-thirds of the
      truth, and crosses in one halflife; `rcov`'s estimators each win one
      condition; the size study at 20,000 draws a cell puts the shared-scale
      `t_5` within 0.004 of WKD in every cell, so the 0.024 was noise and
      the independent reading is the one that misses; the Epps sweep to
      `L = 1024` levels off at 0.775 between 256 and 512. PERFORMANCE §26:
      the benchmark under 0.10.0, 0.11.1 and this build found two slowdowns
      0.11.0 shipped, bisected to tasks 101 and 132 and fixed bit for bit
      (task 101's record corrected); `micro`/`kmeans`' cost is task 102's
      default floor, kept; the CSV and NDJSON read-ahead limits change
      nothing on either surface; §19's two states measured; §21 re-measured
      by `scripts/plan_inspection_bench.py`. C16: both blend oracles are the
      centred mixture, run at 0 and 1e8. C21: `r2_y` under a delay is
      scikit-learn's `r2_score` over the matured rows, to 1e-9; and
      **`coverage_y` was not** -- a held row was scored, at release, against
      the conformal radius reached by then rather than the radius its
      interval was shown with (C21's other half): the record of a held row
      now carries the radius it was shown, `Conformal::update_against`
      scores against it and steps the current radius by that error
      (delayed-feedback ACI), and a row shown no interval counts for
      neither; schema 18, and a 17 file's rows are scored as before
      (`a_delayed_row_is_scored_against_the_radius_it_was_shown`,
      `test_coverage_is_the_share_of_matured_rows_inside_their_interval`).
      S23: `emit_averaged`'s docs say its weights are a mean loss ratio that
      does not sharpen, and a slot without a prediction drops out of the
      row's average. D9: deleted. §6: a constant column in `deco` freezes
      every block (a decision now, task 115 (h)); `bocpd` in a stationary
      stream prunes little -- 2,001 runs kept after 2,000 rows and 7,036 at
      most by 20,000 under the defaults, so `max_run` (10,000) is the real
      bound on its memory and its per-row cost, now in its docs -- and a
      test holds a stream at `max_run = 1,000` for 3,000 rows, finite
      throughout, with a 4σ shift found by `run_mode` within 3 rows (the
      pruned and exact runs agree after the shift; `p_change`, `P(r ≤ 1)`,
      peaks at the shifted row and falls, so it is not the reading to wait
      on). A 100,000-row test would take four minutes a debug run. Its
      `gaussian` emission at d = 3 is a textbook
      multivariate-t longhand in Algorithm 1, to 1e-10. T2: every kind is
      read off a real model's state. U5: a Decimal and a `UInt8` column
      through the CLI give the bank's numbers on the same `Float64`s.

<a id="task-113"></a>
- [x] 113. **CI and tooling.** S–M. S7, one composite action for the Linux
      prep step (five copies now); a coverage run, and whether Rust coverage
      joins CI; a fresh mutation baseline and its triage (the weekly job
      starts 2026-09-27; `lasso` and `ewridge` first); WRITING's step-5
      structure checks as a script. **S7 and the script done 2026-09-27.**
      S7: `.github/actions/linux-build-prep`, called by all five workflows
      (SIMPLIFICATION S7 has the rest). The script:
      `scripts/doc_structure.py` parses each file as CommonMark with
      GitHub's tables (markdown-it-py, a new dev dependency), counts a row's
      cells on its source line as GFM splits it, and makes GitHub's anchors;
      its anchors, tables and rows matched GitHub's own rendering of every
      tracked file (1,066 anchors, 373 tables), and its tests' expectations
      were each rendered by GitHub first. `tests/test_doc_structure.py`
      runs it over every tracked file. It found three rows whose unescaped
      `|` dropped cells (ENHANCEMENTS E23 and E54, MARGINAL-LAGS-AND-BINS'
      `E[y | x]`, escaped now), two example links whose target was `...`
      (PHRASING, WRITING, escaped), and eleven tables in this plan that
      GitHub showed as code, moved from six spaces to four. **For the
      user:** the same cause shows most of this plan as code on GitHub. An
      entry's text starts at column 2, after `- `, so a paragraph that
      follows a blank line at six spaces is an indented code block: 35
      blocks, 2,662 lines. Four spaces would render them as text; the diff
      is whitespace across the whole file, so it is not made without a word.
      **The rest done 2026-09-27.** The mutation baseline over `lasso.rs`
      and `ewridge.rs`: 906 mutants, 104 missed and 152 timed out; two
      `--iterate` rounds leave none that is neither caught nor recorded
      with its reason (14 equivalents; TESTING, "Mutation survivors"). The
      new tests use third-party oracles (the user asked, 2026-09-27: "is
      there no third party oracle library?"): `faer` solves the Rust
      oracle's normal equations, and scikit-learn's `Ridge` holds every
      `ewridge` solve in the Python suite, 20 cases to 1e-8; `cargo mutants`
      sees only the Rust tests, so both. The changed lines of task 112's
      performance fixes: 42 of 43 caught, the other equivalent. Coverage
      (`scripts/coverage.sh`): 96% Python, 93.9% region and 92.6% line
      Rust, from 75% and 73% on 2026-08-30. **Rust coverage does not join
      CI:** it sees `cargo test` alone, so `online-py` reads 0% and
      `online-polars` less than the Python suite drives; it would add an
      instrumented build a run; and the mutation jobs, on every push and
      weekly, are the sharper measure of the same thing. CI keeps the
      Python figure it reports.

<a id="task-114"></a>
- [x] 114. **Four formulas checked against their papers.** S–M; needs the
      papers (none is in `.cache/research/papers`). WKD 2012's Bartlett
      kernel (`corrchange` weights lag `l` by `1 − l/(γ+1)`; the transcription
      reads `1 − l/γ`; a change moves `corrchange`'s goldens); CKP 2010 §3's
      exponent for `rcov`'s `psd` window; ABK 2023's robust posterior for
      `bocpd` (*decision if `robust`'s numbers move, or it becomes a new
      name*); Wied & Galeano 2013's sequential detector (paywalled; *a new
      `corrchange` kind, a decision*). **Checked 2026-09-28 (the user: "Check
      against papers that are not paywalled"; then, of Wied & Galeano,
      "there seem to be many others that describe it exactly").** Every
      paper is in `.cache/research/papers` with its text (PDFKit): WKD from
      the Ruhr-Universität Bochum copy, CKP as arXiv 2602.19645, ABK as arXiv
      2302.04759, and Wied & Galeano from their own open preprint, SFB 823
      Discussion Paper 12/2012 (TU Dortmund Eldorado, version of 12 March
      2012; the journal version may differ in detail), cross-checked against
      Dette & Gösmann (arXiv 1802.07696, §5.1) and Pape, Galeano & Wied (SFB
      823 DP 7/2017), which restate its boundary. Four results:
      (1) **WKD's kernel: the code departs, the transcription was right.**
      Appendix A.1 writes `D̂₁ = ΣₜΣᵤ k((t−u)/γ_T)VₜVᵤ'`, `k(x) = 1 − |x|` on
      `|x| ≤ 1`, `γ_T = [log T]`: lag `l` at `1 − l/γ`, lag `γ` at 0.
      `corrchange` uses `1 − l/(γ+1)` over lags `0..=γ` (Newey–West), in
      `long_run_sd`, `scalar_stat` and the tests' oracles. Measured with an
      independent numpy statistic that reproduces the library's `stat` to
      1.7e-14 on real spans, 20,000 shared draws a cell: the size at WKD's
      Table 1 cells (shared-scale `t_5`, 5 %) is 0.0397/0.0398, 0.0336/
      0.0334, 0.0389/0.0394 at `T = 500` and 0.0388/0.0396, 0.0353/0.0358,
      0.0382/0.0384 at `T = 1000` for `ρ = −0.5, 0, 0.5` (current/paper;
      WKD 0.040, 0.035, 0.041, 0.038, 0.034, 0.039); the power at Table 2's
      break (0.5 → 0.7 mid-span) is 0.5479/0.5509 at `T = 500` (WKD 0.587)
      and 0.8132/0.8151 at 1000. *Decision:* follow the paper (recommended:
      the model is documented as WKD's test, and the change is neutral to
      slightly better) -- `corrchange`'s goldens move, and `bandwidth = 1`
      becomes lag 0 alone -- or keep Newey–West and say so, as the module
      docs now do. (2) **CKP §3.4: the code agrees; nothing moves.** Eq. 16
      `k_n/n^{1/2+δ} = θ + o(n^{−1/4+δ/2})`, `0 < δ < 1/2`; Eq. 17 has no
      bias term; Theorem 4 (ii) calls `δ = 0.1` optimal (rate `n^{−1/5}`).
      `rcov`'s `⌈θ·n^{0.6}⌉` with the bias dropped is that; the "second-hand"
      caveat in `window_for` is gone. (3) **ABK: `robust` is not it, and the
      exact posterior does not fit a streaming model.** Theirs is Gaussian
      over the natural parameters, `Σ⁻¹ += 2ωΛ(x)`, `μ = Σ(Σ⁻¹μ − 2ων(x))`
      (Prop. 3.1, §3.4), robust through `m(x)` built from a reference `θ*`
      (Prop. 3.2) that they take as the MLE on the whole data set, with `ω`
      tuned by KL-matching the standard posterior on the first `t*` rows by
      automatic differentiation; the predictive is closed-form for a Gaussian
      with a changing mean, and with mean and variance both unknown (every
      `bocpd` emission) they sample it (App. C.1). *Decision:* keep `robust`
      as the β-power-weighted variant it is documented to be and drop the
      "swap ABK in" follow-up (recommended), or build ABK as a new emission
      with a warm-up `θ*`, a given `ω` and a seeded sampled predictive.
      (4) **Wied & Galeano: the detector, exactly.** `V_k = D̂·(k/√m)·
      (ρ̂^{m+k}_{m+1} − ρ̂^m_1)` (Eq. 1; `D̂` the inverse long-run standard
      deviation, WKD's A.1 on the `m` historical rows), stop at `τ_m = min{k
      ≤ [mT] : |V_k| > c·w(k/m)}` (Eq. 2), `w(b) = (1 + b)(b/(1 + b))^γ`,
      `0 ≤ γ < 1/2` (Eq. 5; Dette & Gösmann and Pape et al. add a floor
      `max(·, δ)` inside), `c(α)` from `P((T/(1+T))^{1/2−γ} sup_{0≤s≤1}
      |W(s)|/s^γ > c) = α` (Eq. 7), their Table 1 at 5 %: `γ = 0`: 1.2870,
      1.5578, 1.8158, 1.9980; `γ = 0.25`: 1.8001, 1.9924, 2.1684, 2.2467;
      `γ = 0.45`: 2.6282, 2.6844, 2.7215, 2.7660, for `T = 0.5, 1, 2, 4`; the
      changepoint `argmax_j D̂(j/√τ_m)|ρ̂^{m+j}_{m+1} − ρ̂^{m+τ_m−1}_{m+1}|`
      over the monitoring rows alone (Eq. 8). Their Table 2 (GARCH pairs, 1000 draws): sizes
      0.047–0.087 at `γ ≤ 0.25`, 0.106–0.174 at `γ = 0.45`. *Decision:* a
      `kind = "sequential"` (a training span of `m` rows, then the detector
      over at most `[mT]`), or not. It shares (1)'s kernel.
      **Decided 2026-09-28 (the user: "Your recommendation on WKD 2012 and
      abk 2013, build wied & galeano 2013 has a corrchange be sure to
      document well") and built.** (1) The kernel is WKD's, `1 − l/γ` over
      lags `0..γ` (`long_run_sd`, `scalar_long_run_sd`); the oracles in
      `the_statistic_is_its_definition` and
      `the_scalar_statistic_is_its_definition` failed on the old kernel;
      REGIMES §2–4 re-measured (size moved by at most 0.001 a cell, power by
      0.009), with dated notes keeping the old figures. (3) `robust` stays,
      ABK is not built; `bocpd.rs` says why. (4) `kind = "sequential"`:
      `span_rows` rows of history (`m`), then up to `monitor_rows` rows
      (default `span_rows`, `T = 1`) each tested as it arrives, `stat =
      max_pairs |V_k|/w(k/m)`, the cycle ending at a flag or its last row;
      `boundary_gamma` (default 0); `scalar` supported (the mean of `u`);
      `crit` overrides. The critical value is Eq. 7 with `q_γ` from
      `crates/online-core/src/boundary.rs`: the series of `sup|W|` at `γ =
      0`, and above it the law solved, not simulated -- `U(t) =
      e^{t/2}W(e^{−t})` is a stationary OU process, so `Z_γ ≤ c` is an OU
      path inside the corridor `±c·e^{(1/2−γ)t}`, a Fokker–Planck equation
      on `[−1, 1]` with absorbing ends (Scharfetter–Gummel, Crank–Nicolson;
      matches the `γ = 0` series to 3e-6), within 0.03 of W&G's Table 1 and
      above it in 11 of 12 cells, cached per `(α, γ)` and recomputed on
      load (not state). Eq. 8 dates a flag as `since_change`, a new output
      of every kind (`"monitor"`: after the CUSUM's argmax; `"window"`: the
      second window). Size on W&G's GARCH design within two standard errors
      of their Table 2 in all 24 cells (REGIMES §9, `regime_experiments.py
      sequential`). With it, a parameter that belongs to another kind is
      refused (`crit`, `reset` and the permutation knobs under `"monitor"`
      were taken and ignored, though the docstring promised `reset` raised),
      and a `scalar` monitor saved mid-span now loads (the shape check held
      its `u` rows to the feature count). `SCHEMA_VERSION` 20 carries the
      monitoring period.

<a id="task-115"></a>
- [x] 115. **Decisions on behaviour that is built.** Each is S–M once
      decided; the evidence goes with it. (a) `share_p`: VALIDATION §4 shows
      sharing better on both targets, and this plan keeps `False` because it
      hurt one — re-measure, then decide. **Decided 2026-09-28 (the user:
      "Keep share p off"):** `False` stays; the re-measurement (VALIDATION
      §4, 1.215e-6 against 1.279e-6 and 4.963e-6 against 4.987e-6) is on
      data with no predictive signal. (Task 172's relative prior moved them
      to 1.214e-6 against 1.277e-6 and 4.941e-6 against 4.984e-6, sharing
      still better on both.) (b) The solve cadence by
      accumulated weight: `halflife=1e12` solves once (a state field; the
      next schema bump). **Decided 2026-09-28 (the user: "Do 115b") and
      built:** with no `solve_every` under a finite halflife, `ewridge`,
      `lasso`, `huber` and `quantile` solve once the weight learned since
      the last solve reaches `ln 2 / 50` of the weight the fit holds
      (`online_core::DEFAULT_SOLVE_SHARE`; `n_eff`, and `robust`'s raw
      weight), which on evenly spaced rows in steady state is `halflife /
      50` of clock (at halflife 500, 10.007 rows against 10). An explicit
      `solve_every` keeps its clock; `lam` and an infinite halflife still
      solve every row; `max_rows_between_solves` still caps. The share is
      configuration the bank sets from the spec on build and on restore,
      and the counter is state: `SCHEMA_VERSION` 20, a 19 file loading
      with it at 0 (`tests/test_solve_cadence.py`; the oracles in
      `tests/reference.py` and `tests/reference_paths.py` run the rule).
      Measured on VALIDATION §1's data: the clock cadences of 9, 10 and 11
      rows give MSE 1.1741, 1.1617 and 1.1606 (×1e-6), the weight rule
      1.1704 -- phase noise on a stream whose every fit has a negative
      R², so the report cannot rank them; solving every row is the worst,
      1.2255. (c) §12: a weight-0 row after the decay underflows
      keeps the history, where `decay(0)` forgets it (recommended: forget)
      — *a clock gap: reviewed with task 120*. **Decided (forget) and built
      2026-09-28 with task 120**, which records `hmm`'s subnormal edge,
      left as is by the user's call. (d) Task 80's raised calls: the
      256 MiB window budget (**decided 2026-09-28**, the user: "256 mib is
      a good default window budget and on failure it should explain how to
      raise it" -- kept, and the refusal now names the default when the
      spec set none and spells each way out, `{"refuse": MiB}` larger or
      `inf`, a larger `window_every`, `{"thin": MiB}`;
      `over_budget_tests`); a budget-refused bank refusing every later
      call (the user asked how the other budgets handle it: every other
      one -- `gram_block_rows`, `marginal`'s `bin_budget` -- is sized from
      the configuration and refuses before a row is learned; only the
      window's depends on the rows. Options raised: keep the broken bank,
      or a pre-pass over the chunk's clock, as the backwards-clock check
      makes, that predicts the ring and refuses the chunk untouched --
      recommended, with a test holding the prediction to the ring;
      **decided 2026-09-29**, the user: "115.1 follow your suggestion", the
      pre-pass, **and built**: `Stream::window_prepass` replays the chunk's
      schedule on shadows of every ring -- `online_core::WindowShadow`, the
      ring's own `offer` and `trim` on byte counts -- before any stream
      runs, so the chunk is refused whole and the bank goes on; held to the
      ring's own verdict in `crates/online-polars/tests/window_prepass.rs`,
      seven cases at four chunkings, the same piece and the same words
      against a bank run without it, whose reset and close cases fail with
      either path switched off; under `drift_action = "reset"`, whose
      resets depend on the residuals, it abstains and the broken bank
      remains); C24 part 2, `ftrl`'s penalties under a halflife (the user asked
      whether river covers it: only at `halflife = inf`, where we are river
      to the bit; river has no forgetting, so a finite halflife is ours, and
      the build is river's formula with river's constant penalties on
      decayed sums, whose cost is the shrinkage C24 measured; undecided.
      **A second oracle, 2026-09-29** (the user: "find an alternative
      oracle for ftrl that supports more options"): Vowpal Wabbit's
      `--ftrl`, in the dev group, holds `pred` and `coef` on every row to
      1e-5, its single precision (measured 2.6e-6), under both losses, the
      intercept, row weights with zeros, null targets, two targets and `l1`
      with `l2`, where river's covers the logistic loss and the penalties,
      on the state recursion alone (`TestFtrlIsVowpalWabbits`). The user
      allowed it a CI leg; none is needed, since 9.11.9 ships wheels for
      3.12 to 3.14 on all three OSes (reported as missing for 3.14, read
      from the classifiers, which stop at 3.13). None of river, VW or
      Keras's `Ftrl` forgets, so the penalty question stays open. **Decided
      2026-09-29** (the user asked whether to change the behaviour only
      under the EWMA, then: "Build it"): not a switch keyed on the decay,
      which is discontinuous in the halflife -- at `halflife = 10^6` the
      weight-scaled penalties would be near zero for 1.4M rows, full at
      `inf` -- but a per-target scale `m = W/W*` on `beta/alpha`, `l1` and
      `l2`, `W*` the target's weight on a clock that runs only on the rows
      that teach it; the steady state stays the prior against the window
      (4.65 at `halflife = 100`, 0 bias at `inf`), and only the drift of a
      fit with no data goes. **Built:** a row that teaches nothing -- the
      target absent, weight 0, a refused label -- leaves every coefficient
      and prediction to the bit, through a total gap and 100,000
      halflives, where 100 clock units had taken the fit to 0.748 of
      itself at `halflife = 100` (`a_row_that_teaches_nothing_leaves_the_fit`);
      the decay waits per target until the next row that teaches it, so
      the sums cannot underflow; held to the decayed form written out, to
      1e-12 across gaps of 1 to 2,000 clock units and weights of 0 to 2
      (`the_penalty_scale_is_the_longhands_across_gaps` (renamed `the_fit_is_the_longhands_across_gaps` by task 215, which dropped the scale)), and to
      `ftrl_ref` moved to the same rule (the Python oracles, with nulls and
      zero weights under halflives of 150 and 100). At `inf` the fit is
      river's and VW's as before; the two golden signatures at halflife 40
      moved; state schema 20 gains three vectors per target, a 19 state
      loading with `m = 1`); S30,
      `holt` with no level-only mode (**decided 2026-09-29**, the user:
      "115.3 do your suggestion", **and built**: `holt(trend=False)` holds
      the trend at zero for a flat forecast, simple exponential smoothing,
      whose level is the EW mean of the observations -- held to pandas'
      `ewm(times=)` at 2.2e-12 and to `DescrStatsW` at 5.6e-16
      (`TestALevelOnlyHoltIsAnEwMean`), and to its definition in `holt.rs`;
      `trend_halflife` is refused beside it, and a config without the field
      loads with the trend on, so no schema bump); S31, `n_eff` counting rows with a null target
      in `ftrl`, `pa` and `sgd` (hard rule 8). **S31 decided 2026-09-28
      (the user: "n eff should do what it does everywhere else") and
      built:** measured first, every model emits the shared weight as
      `n_eff` and every regression model but these checks a target's
      `min_periods` against its own weight (ten null-target rows, then the
      target, `min_periods = 2.5`, halflife 200: `ewridge`, `lasso`,
      `huber`, `quantile` and `kalman` first predict on row 13, `holt` too
      once its level is set); `pa`, `sgd` and `ftrl` predicted from row 3 on
      coefficients no target had moved, and `rls` from row 11. So `pa`,
      `sgd`, `ftrl` and `rls` now keep `w_target`, the weight of the rows
      that carried each target (for `rls` the rows it learned from, all
      targets present; for `ftrl` not a label `strict_binary` refuses), and
      gate each target on it in `step` and `predict`; the emitted `n_eff` is
      unchanged. `SCHEMA_VERSION` 20 (with (b)); a 19 state loads with each
      target at the shared weight. Pinned by
      `test_a_target_counts_toward_min_periods_only_on_rows_that_carry_it`
      (the whole sweep) and a test in each model's file; the oracles
      (`rls_ref`, `ftrl_ref`, `rls_paths_ref`, `pa_ref`, `sgd_ref`) gate the
      same way. `sgd`'s `inv_scaling` rate still reads the shared weight.
      (d) is done: `ftrl`'s penalties, its last item, are built above. (e)
      The 2026-09-12 review's
      sign-offs (P4's calls; decisions 1, 3, 4 and 5; D1's caveat to rule
      5) and REVIEW-2026-09-18's unrecorded S1, D2 and B7. **Approved
      2026-09-29** (the user: "115.5 approve all"): decisions 1, 4 and 5
      stand, 3 went with task 136, P4's calls are (d)'s two budget items,
      and S1, D2 and B7 stand; D1's caveat is in `CLAUDE.md`'s rule 5, and
      each review document records its sign-off. (f) S18, Bartlett
      weights for `serial_rule`; S19, windowed target moments; C18, a
      windowed lag ring — record as tasks or decline. **Decided 2026-09-28
      (the user: "Accept tasks s18, s19 as long as it doesn't add much
      extra size, c18 all to be done after the current work"):** tasks
      135, 136 and 137. (g) C8, the CLI's
      NDJSON on the system allocator: write it from one thread (the
      mimalloc option is parked with the new libraries). **Built
      2026-09-28** as low-hanging (the only fix not parked), after
      re-measuring at HEAD on the three-spec example bank over 3M rows: the
      slices ran 4.98–5.50 s with 2.1 s of system time and no tail, polars'
      `BatchedWriter` on the writing thread 4.65–4.78 s with 1.1–1.2 s, and
      parquet 4.65–4.73 s; so the one-thread writer is both the simpler and
      the faster now (`ndjson_write`). (h) `deco` with a
      column that has no spread (found by task 112's test, 2026-09-27): a
      row learns only when every standardized value is finite, so while one
      column is constant -- from its first row, the only way its variance is
      exactly zero -- no block's `rho` moves, its own or any other, and `u`
      and `loglik` are null for every block; `n_eff` advances and every block
      learns again once the column moves
      (`test_a_constant_column_poisons_nothing_and_every_block_learns_once_it_moves`
      pins that much). Whether the other blocks should go on learning, and
      the constant column's own block too (from its other columns), is the
      decision. **Decided 2026-09-29** (the user: "115.4 your
      recommendation"): each correlation value keeps its own weight, a
      column with no standardized value is left out of the row's sums, and
      `loglik` is null on such a row. **Built 2026-09-29:** while a column is
      flat its block reads what the model without that column reads, `u`
      and `rho` to the bit, and a block's value is the one-block model over
      its own columns on every row
      (`a_column_without_spread_is_left_out_of_its_block`,
      `each_value_keeps_its_own_weight`,
      `test_a_constant_column_is_left_out_and_every_other_value_learns`);
      a value with no estimate neither learns nor decays, the rule the
      whole row had. `rho_w` is one per value in schema 20, and a 19
      state's single weight reads as that weight on each. Also (review
      2026-09-28, pre-existing): under `dynamics =
      "linear"` a zero-weight row moves `rho` (`deco.rs`: `(1-α-β)·bar +
      α·u + β·rho` is recomputed whether or not `b` is 0), against hard
      rule 9; whether the linear form should hold `rho` still on such a row
      is a decision. **Decided 2026-09-28 (the user: "Hold rho still on a
      zero-weight row") and built after 0.12.0:** `rho' = rho` at `w = 0`
      under `"linear"`; positive weights still reach `rho` only through
      `rho_bar`, which the docstring states (options not taken: scaling the
      `alpha` step by the weight, or documenting only). (i) The Linux CLI's glibc floor (task 109's leftover,
      measured 2026-09-27 on 0.11.1's assets): both binaries need
      `GLIBC_2.39`, the runner's own -- `pidfd_getpid` and `pidfd_spawnp`
      from Rust's standard library, weak symbols under a version requirement
      that is not -- and hard `2.34` and `2.35` (`__libc_start_main`,
      `hypot`); so no Ubuntu 22.04, Debian 12 or RHEL 9, and the floor rises
      with `ubuntu-latest` (RELEASE-READINESS has the table). Options: build
      the CLI in the manylinux2014 container the wheels use (glibc 2.17,
      still dynamic, nothing newly linked; recommended); build it on
      `ubuntu-22.04` (2.35, until that image retires); or a static musl
      binary (static linking, rule 12, a raise). Each changes the release
      job, which only a dispatched run tests. **Decided 2026-09-28 (the
      user: "follow your recommendation on the next release") and built:**
      the Linux CLI is built by `docker run` in
      `quay.io/pypa/manylinux2014_$(uname -m)`, and
      `scripts/glibc_floor.py` fails the build job above 2.17 before any
      upload (`tests/test_release_workflow.py` holds the shape; the script
      refuses 0.11.1's binary at 2.39). The 0.12.0 rehearsal built it (run
      36448809775 on `36e468c`: 2.16 on x86_64, 2.17 on aarch64), and 0.12.0
      shipped it.

      **Done 2026-09-29:** every item, (a) to (i) and S30, is decided and
      built.

<a id="task-116"></a>
- [x] 116. **Readiness beyond `ewridge`** (WARMUP-AND-CONVERGENCE §7). M–L;
      defaults move, so *the scope is the user's*. A readiness statistic for
      `rls`, `kalman`, `lasso` and the scalar EW models; `support_coef` for
      `lasso`, `ew_cov` and the robust models; `n_eff_settled` in the
      summary; a warning where `min_periods` can never be met, against the
      ceiling `1 / (1 − λ^d)` (task 148 corrects the docs' figure); the
      inflation notice's halflife figure; §7.1's coefficient
      standard errors (its CUSUM is task 146's); §7.7's flicker; §7.9's CLI
      closing count. *Checked 2026-10-07*: task 198 (`54975d9`, D8) built
      the summary's `weight_sum_settled`, the warning where `min_weight`
      can never be met and the inflation notice's half-life figure, and
      declared the current floors final; task 146 closed without the
      CUSUM, which nothing owns. The rest is open; the user wants it built
      before 1.0, scope to follow from a summary (§18, 2026-10-07).
      *Done 2026-10-08* (the user, 2026-10-07: "Your reco on 116"; branch
      `task116-readiness`, integrated after review 5): A `max_error_inflation`
      and `emit_error_inflation` on `rls` (`s₂` in the state, schema 45; the
      gate `√(1 + k/n_Kish)`, the row's leverage) and `kalman` (each row's
      exact `√(1 + z'P⁻z/R)`, the gate per row), the gate on `lasso` (`df` =
      the active count plus the intercept); B `support_coef` on `huber` and
      `quantile` (`1 − λ(A⁻¹)_jj` on the band system, persisted); F `se_coef`
      under `emit_se_coef` on `ewridge`, `rls` and `kalman` (`diag(T Cov Tᵀ)`,
      a report); G the support warning under the notices' one 95%-settled
      rule (a bug: 8 seeds of 12 warned falsely at the first row through);
      H the CLI's closing line. The "cannot be met" notice is off for
      `kalman`: its gate is per row and `P` settles on `coef_half_life`'s
      clock, not the spec's decay, so a notice tested at ±5% cannot hold
      (§19, 116-K; WARMUP §7.11). `tests/test_window_budget.py`'s spec was not
      §2.2's true positive: on the base it read `support_coef` 1.00 and never
      warned, so its comment and §2.2 were corrected and no code changed for
      it. `kalman`'s steady-state floor, measured against the brief's
      `√(1 + k·p)`: 1.0715 against 1.0675 at `coef_half_life` 50 (`h` 6%
      above `k·p`), 1.84 against 1.58 at 5 (60% above). Costs a row: `huber`
      +5% and `quantile` +4% by default (`support_coef`); the `kalman` gate
      +115-120 ns when set; `rls`'s `s₂` within the noise; `se_coef` +10 ns
      on `ewridge` at the default cadence.

<a id="task-119"></a>
- [x] 119. **Housekeeping.** S; *the user's call*. The 13 remote branches
      merged into `origin/main` (three point at release tags) and about 40
      local branches, all merged. **Done 2026-09-29** (the user: "Do task
      119"): the 39 local branches went first; the 13 remote ones, each
      checked merged into `origin/main` and with no open pull request, were
      deleted with `git push origin --delete`. The tags three of them
      pointed at, `v0.2.0`, `v0.9.0` and `v0.9.1`, remain; the remote holds
      `main` alone.

<a id="task-120"></a>
- [x] 120. **Rows out of order and clocks that go back: what an audit of
      2026-09-25 found. Needs extra review before any of it is built (the
      user, 2026-09-25).** One read-only audit traced every mechanism that
      detects or absorbs disorder — `on_clock_reset` in `ClockState::advance`,
      `min_backwards_jump`'s chunk pre-check, sessions, the query check, the
      per-group clock, `group_close="monotone"`'s key check, `label_delay`'s
      release, the summary's `clock_backwards`, `refresh_time` and `embargo`
      — and the two findings marked *reproduced* were run on the build
      (scripts in the session's scratchpad, not in the repository).
      - **An infinite cap turns a clock step back into an infinite step —
        CONFIRMED, reproduced.** Under `max_dclock=inf` the order check is
        off by default, and `"max"` hands the models a step of `+inf`
        (clock.rs:356-362, 398). `holt` predicts null on every row after
        the step, for good (0·inf in its level and trend, holt.rs:208-235);
        `kalman`, `ewridge` with or without `label_delay` keep a non-finite
        value in their state, which the JSON export refuses (`decay_time`
        and `settled_frac`, stream.rs:3686; `pending_clock` inf − inf,
        3689-3691); the readiness warning then claims every prediction is
        withheld for good while later rows still predict. `session_gap=inf`
        under an infinite cap does the same. REVIEW-2026-09-18 V2 raised it
        and closed it because `max_dclock` is required — which `inf` meets.
        Input: `clock="t"`, `max_dclock=inf`, t = 0, 1, 2, 3, 1, 2, 3, 4.
        test_clock_order.py:152-155 accepts this pattern for `ewridge` and
        checks only the row count.
      - **A minimum equal to the clock's smallest step never fires —
        CONFIRMED, reproduced.** The comparison is strict (clock.rs:346)
        and the "would never fire" refusal tests `v < tick` (arrow.rs:595):
        a `Date` clock with `max_dclock="1d"` absorbs the days 1, 3, 2
        silently, where `"2d"` refuses the same stream. Its refusal message
        reports the step in seconds (`86400`), not as a duration.
      - `max_dclock=0` turns the check off by default too (spec.rs:2065-2071);
        the README names only `inf` (CONFIRMED).
      - V14's test was never written: nothing steps the clock back at a
        session change with `session_gap` unset (clock.rs:325-328) (moved
        here from task 112).
      - Chunk invariance with a step back that a policy absorbs is untested:
        the property generators make forward clocks only, though
        test_properties.py:4 says otherwise; one chunk boundary is tested,
        and a refusal whose earlier row sits in the previous chunk only
        under `"error"`.
      - Docs and comments that contradict the code: PLAN:202-203 and
        ENHANCEMENTS:41 ("errors loudly"); bank.rs:528, 2698-2700, 2842-2847
        and stream.rs:2334, 2393, which call the pre-check `"error"`-only;
        ARROW-SOURCES:601-604, 684-686 (out-of-order rows refused — only
        small steps are); README:2703; WARMUP:582; "jitter check" in test
        comments.
      - An error's row number counts from the start of the chunk in the plan
        form, the CLI and `refresh_time` (_frame.py:648-651, runner.rs:763,
        refresh.rs:247), where the README implies the input row.
      - Resuming onto input that overlaps the saved state learns it twice:
        the rerun's first row is at least `max_dclock` behind, and `"max"`
        absorbs the step silently; STATE-WORKFLOW says nothing of it.
      - PLAUSIBLE: the query check takes any `Sort` as fixing the order
        (_frame.py:325-326), but `sort` without `maintain_order` leaves ties
        unordered and equal clocks pass the clock check; `embargo` merges
        by clock alone, so it needs the global clock order where a stream
        needs each group's, and its docstring says "as a stream must be";
        `refresh_time` casts the clock to f64 (refresh.rs:228), so on a
        `Datetime(ns)` a step back of under 256 ns can round to a tie.

      **The planned work that touches this area**, all designed before
      `min_backwards_jump` existed (2026-09-20), none of which mentions it
      (task 106, which would have made a backward clock a session change,
      is discarded): task 78's column form (it closes
      every window at a step back in any group, where the bank checks each
      group; `advance` records the new clock even when it refuses, so the
      window core must work on a copy and commit it; an IO source that
      refuses mid-stream has already emitted rows); task 104's parity tests
      (their streams must step back by at least the cap or switch the check
      off); task 105 (one vocabulary run by the same clock code, and
      `load_state` on `refresh_time`, imply an exact temporal clock there
      and a saved last time per group, checked on resume, which no sub-task
      owns).

      **Decided 2026-09-28 (the user): `"max"` is removed from
      `on_clock_reset`.** `max_dclock` is a cap on a step and nothing else
      (the user: "Max dclock is meant to be a cap on dclock, where is it used
      as a default?"). It had two other uses, and both confirmed findings
      above come from them: the step a backward clock took under `"max"`
      (clock.rs:356-361), infinite under an infinite cap, and the default of
      `min_backwards_jump` (spec.rs:2202-2210).

      **Decided 2026-09-28, every clock question (the user: "Resume should
      get a helper. Remove inf as an option for max dclock. Remove zero
      option. The rest as you suggest."):**
      - `max_dclock` is required with a clock and must be finite and
        positive: `0` is refused (it turned decay off, and every positive
        gap then read as a break, so `label_delay` released every held
        label on the next row -- measured: learning from row 1 where a
        delay of 3 starts at row 3), pointing to `halflife="inf"` or no
        clock; `inf` is refused (the cap is also what a gap is a break
        against), pointing to a large finite cap.
      - `on_clock_reset` keeps `"error"`, now the default, and
        `"reset_state"`; `"max"` and `"zero"` are removed and refused by
        name.
      - `min_backwards_jump` has no default derived from the cap: it is
        required with `"reset_state"` (the caller says what a late row is)
        and refused with `"error"`, where every step back is refused anyway.
        A step back no larger than it is a late row: the comparison is
        inclusive, and the message states the step as a duration.
      - `session_gap` is finite or `"reset"`; `inf` is refused, pointing
        to `"reset"`. No model ever receives a non-finite step, which a
        contract test holds for every model.
      - A zero-weight row after the decay underflows forgets, as the decay
        does (task 115 (c)).
      - Resuming is the next chunk: an overlap takes the policy, which by
        default refuses it. A helper returns each group's last learned
        clock so a caller can skip what the state has learned, and
        STATE-WORKFLOW says so.
      - For later tasks: the column form (78, 104) checks disorder per
        group, as the bank does; task 104's layout change gets no loader
        (the pre-1.0 waiver of 2026-09-14).
      - `sort` without `maintain_order` in the query check, and `embargo`'s
        global order: a test first; changed only where it fails.
      - With no decision needed: the session-change test (V14), property
        tests whose clocks step back, the docs and comments that contradict
        the code, error row numbers counted from the input's start, and
        `refresh_time` reading a `Datetime` in integer nanoseconds.

      **Built 2026-09-28**, every decision above but the two recorded for
      later tasks (the column form's per-group rule, 78/104; no loaders for
      104). The user added, while it was built: "Do not worry about old
      specs" -- so a bank file before schema 19, which names `"max"` (the
      old default, always written), is refused by its version
      (`MIN_BANK_SCHEMA_VERSION`, bank.rs); a model's own state from 14 on
      still loads, no model state having changed layout.
      - The clock (clock.rs): `OnClockReset { Error (default), ResetState }`;
        the late-row test is `back <= min_backwards_jump` under
        `ResetState` only; `advance_scoring` gives a step back a step of 0
        and never refuses or resets, which `predict` uses. A proptest,
        `finite_steps`, holds every step finite and within the cap over
        every configuration the spec lets through.
      - The spec (spec.rs `clock_cfg`): the cap finite and above 0 (0 names
        `halflife = "inf"`), `session_gap` finite or `"reset"`, the minimum
        required with `"reset_state"` and refused with `"error"`. A clock's
        refusal message gives a temporal step as a duration (`1d`, not
        `86400`). `clock_scale` names a rate (`lam`, `q`) before a plain-number
        cap, since a finite cap now always sits beside it.
      - Resume: `ModelBank.skip_learned(frame)` (DataFrame or LazyFrame),
        from `Bank::last_clocks`, exact in nanoseconds; STATE-WORKFLOW
        "Resuming on input that overlaps the state".
      - 115 (c): the zero-weight row after an underflowing decay forgets, in
        `EwCov`, `EwDiag`, the gaps accumulators and `deco`; a contract test
        over every model compares it with the row one halflife short.
        **Found and raised, not fixed:** `hmm` one halflife short -- a
        zero-weight row at a subnormal decay factor, from about 1025
        halflives -- leaves its co-moments and precision prior a few bits,
        and its densities NaN for up to 39 of the next 40 rows (measured at
        1030 and 1074 halflives; 1000 and 1075 are fine). A threshold that
        forgets a subnormal share (2^-1022) fixed 1030--1074 but not 1025;
        one that would make it impossible (2^-53) changes shipped output at
        about 57 halflives, a weekend on an hourly halflife. **Left as is
        (the user, 2026-09-28: "very much at the edge")**: it needs a
        zero-weight row about a thousand halflives on. The contract names
        `hmm` where it skips the comparison.
      - Rows are counted from the input's start: `ArrowChunk::row_base`,
        `Bank::fit_predict_from`/`predict_from`, the plan form, `fit`,
        `fit_predict_batches`, the CLI and `refresh_time`.
      - `refresh_time` compares a temporal clock as the column's integer.
      - Test first, then: a sort by **several keys** without
        `maintain_order` reordered 2,109 of 10,000 rows against the stable
        sort (polars 1.44.2), so the plan check reports it; by one key none
        moved, pinned by `test_the_measurement_behind_the_sort_warning`, so
        a change in polars shows. `embargo` on a frame sorted only within
        its groups returned a group out of order: its docs now say it needs
        the clock order across all rows and name the remedy; the bank
        refuses the result by default, so it is not silent.
      - V14's session-change test and the stepping-back property tests
        (`test_properties.py`) are written.
      - Reviewed 2026-09-28 (two readers over both commits): the clock and
        accumulator changes held; three fixes followed in the review commit:
        `skip_learned` compared a temporal clock scaled to nanoseconds in
        polars integer arithmetic, which wraps past 2262 (compared in the
        column's own unit now); `refresh_time`'s state did not record its
        grouping, nor check the clock's kind on a new group's row (both in
        the file now, held on load and on every row). One divergence for the
        user: under a pushed-down `head()` with `save_state`, the bank saves
        the state after the rows the query pulled while `refresh_time` feeds
        the whole input and saves that; task 105 asks for one rule.
        **Decided 2026-09-28 (the user: "Yes") and built after 0.12.0:** the
        bank's rule for both, the state after the input behind the rows
        returned. `refresh_time` stops at the tick that completed the n-th
        point (`RefreshTime::feed_limited`), so the state is the same at any
        chunk size and a resumed run goes on with point n + 1; with
        `pairs=True` a tick completing several points is taken whole.

<a id="task-121"></a>
- [x] 121. **Test libraries under an open-source licence.** S–M. **Done
      2026-09-25:** scikit-learn 1.9.1 in the dev group; `huber` against
      `LinearRegression` in the exact limit and `HuberRegressor` under
      outliers (T-S4), `marginal`'s bins against scipy's `binned_statistic`
      with values on the edges and its split against a stump (T-S12), the
      reverting `kalman` against filterpy (T-S5 in full), and `sgd`
      against `SGDRegressor` live; proptest 1.11 over all 21 models in
      `model_contract.rs` (a deep run of 2,000 streams a model found
      nothing); and a licence check over the dev and docs groups and the
      crates' dev-dependencies. One measurement worth keeping: at one row
      in ten a gross outlier, `huber`'s intercept sat halfway to least
      squares (1.5 against `HuberRegressor`'s 0.66, truth 0.5), because
      its scale is the plain EW residual spread -- review D4, documented in
      `robust.rs`, not changed here. The user,
      2026-09-25: "we do want to enable every unlicensed library in tests
      and park using licensed libraries" -- read as: a library under an
      OSI-approved open-source licence (BSD, MIT, Apache, MPL, ...), which
      asks for nothing to be bought or accepted, may be used in tests; one
      under source-available or commercial terms (the Business Source
      License, the SSPL, a commercial licence) is parked. Already in the dev
      group and open source: numpy, pandas, statsmodels, river, hypothesis,
      pyarrow, duckdb, the ADBC drivers.
      - scikit-learn as a live oracle: the T-S second opinions (T-S4's
        `HuberRegressor`, T-S12, T-S5 in full), TESTING's scikit-learn
        comparison, and the PERFORMANCE measurements that needed it
        installed by hand (`scripts/sklearn_comparison.py`).
      - Rust `proptest` for T-D2, property tests in `online-core` beside the
        Hypothesis ones. A dev-dependency links into the test binaries only,
        never the package, and hard rule 12 covers production deployments,
        not tests (the user, 2026-09-25), so it needs no raise.
      - The rule where it lives (TESTING, "Libraries the package does not
        depend on") and a check in `tests/test_dependency_policy.py` that
        every dev-group library's metadata names an open-source licence, so
        a licensed one is refused rather than remembered.

**Requests of 2026-09-25 for `marginal` at width** (E70–E74,
`docs/MARGINAL-AT-WIDTH.md`, from the same caller as E66–E67; the user:
"examine its feature requests and integrate them into the current
planning cycle"). Examined against the code, and against a benchmark of
the model alone at `p = 10,000` (a scratch crate outside the repository):
the per-target cost is the model's — `3.3·T` ns per feature per row for
the moments, `19.8·T` with six lags and sixteen bins, against the caller's
`3.2·T` and `19.1·T` — and the caller's fixed 14–17 ns per feature per row
is not, since the model alone has `0.0` and `3.5` there.

      **Review 2026-09-26.** The proptest's `value()` said "near the
      input bound" and reached 1e50 of 1e100, its weights never the bound,
      and no windowed or blocked configuration was generated (C8): values
      now sit at the bound, weights at `1e-100` and `1e100` (which
      `INPUT_BOUND`'s doc names as legal), and `ew_ridge`, `lasso`,
      `ew_cov`, `ew_class` and `marginal` run windowed at both cadences,
      `ew_ridge` blocked with pairwise gaps. That found **G1**: `micro`
      reported an infinite distance and *absorbed* the row. A row of weight
      `1e-100` then one of `1e100` leave a standardized spread of `4e-200`;
      the next row at the bound is `5e199` scaled units away, whose square
      overflows to `inf`; `merged_radius2` read a non-finite square as
      "radius unchanged", so `admits` took the row, and `dist` reported
      `inf`. Fixed in `summary.rs`: an infinite square merges into an
      infinite radius, and `dist` computes the distance without squaring
      (scaled by the largest deviation), read only where the square is not
      finite so ordinary rows keep their bits; `micro` and `kmeans` report it,
      and break a tie of overflowed squares by it -- `kmeans`'s search
      started from infinity, so an infinite square never won and the
      runner-up was never set (`a_row_at_the_bound_against_a_vanishing_spread_is_far_not_absorbed`,
      `an_overflowed_distance_is_reported_as_the_distance`,
      `an_overflowed_distance_is_far_and_still_measured`). And **G2**, the
      same test on the quantile `robust`: a target of `1e100` lifted the
      mean the first off-zero row's slope was read against, the fit
      extrapolated to `-1.5e199` at `x = 1e100`, that residual set the
      scale to `5.8e148`, and the next row outside the band -- one term of
      the score, none of the Hessian -- nudged the slope by `h·dev/var`
      against a Gram holding no curvature that far out, to `1e248`; the
      prediction at `1e100` read `-inf`. The nudge is bounded by the row's
      leverage under the Gram's diagonal, so the fit at the row moves by at
      most its residual (the row is brought at most to its target); the
      linearisation holds inside the band and nothing past it is justified.
      (Task 160, TC1b: the diagonal under-reads the leverage of a row
      against correlated features, and such a row moved 6 to 127 times its
      residual; the bound is the full leverage `1 + u'A⁻¹u` against the
      band's own system since, at about half the default schedule's
      throughput: 2.6M rows/s against 5.0M, 10 features.) The 24 robust
      tests, the QuantReg oracles among them, hold; the quantile golden's
      three values moved by about 1e-3 of themselves, the bound binding on
      early rows where a row's leverage exceeds its weight share. Measured
      on the Python oracle's stream (250 rows, halflife 300): `tau = 0.5`
      and `0.9` unchanged to the bit; at `tau = 0.1` one row (44) is
      bounded, and the fit that follows has the lower pinball loss, 0.10852
      against 0.10940, at coverage 0.170 against 0.161 -- the reference
      recursion in `tests/reference.py` mirrors the bound
      (`a_quantile_fit_stays_finite_through_the_bound`,
      `robust_quantile_through_the_bound`). And **G3**, the same test on the
      windowed `lasso`, though the window had no part in it: `Cross` kept
      each target's own feature mean as an offset `δ_j` from the all-row
      mean `m` and reconstructed it as `m + δ_j`; a target absent on a row
      whose feature stood at `-2.6e99` left both at `1e99`, their sum
      resolved nothing below `1e83`, the next present row's deviation of
      `0.4` read as `1e83`, the cross-moment as `1e132`, and the prediction
      at `1e100` as `-inf`, through `ew_ridge` too. Each own mean is now a
      pair of its own (`Cross::mj`), updated from the target's own rows by
      the Gram's steps, so under `own_rows` it is the Gram's mean to the
      bit and a row the target misses leaves it exactly where it was; the
      `pairwise` readers take it directly. Schema 17; `offsets_to_means`
      converts a 14-16 state's live accumulators and every window snapshot
      once (`a_target_absent_on_a_row_at_the_bound_leaves_the_fit_finite`,
      `a_target_absent_on_a_row_at_the_bound_through_the_bound`). The sklearn
      comparison kept sklearn's default `l2` penalty while our spec has none
      (F9): `penalty=None`. The licence check was read for ways round it and
      found none (F10). The seed: proptest draws afresh each run; a failure
      prints the shrunk stream, and `PROPTEST_RNG_SEED` repeats a run.

<a id="task-122"></a>
- [x] 122. **E71: bin each feature once per row.** S; no API, no state,
      bit-identical. **Done 2026-09-25:** `MarginalBins::update_row` forms
      the row's indices once, in a scratch buffer outside the state, and
      the warm-up replay goes through it too. Held to the per-target update
      to the bit (`the_row_update_is_the_per_target_update_to_the_bit`), and
      to the build before by `marg_bench`'s checksum, absent targets
      included. At nine targets the bins cost 2.6 times less; at one, a
      fifth less, which the request did not expect (PERFORMANCE §22). Found
      on the way: the docstring promised the warm-up replay to the bit,
      which a row of weight zero inside the warm-up breaks by about 1e-15
      of the data's scale. That is the trade the design doc records, so
      the docs now say it and a test pins it. Verified: `update_target` runs `bin_of(&self.edges[j],
      x[j])` once per present target (margbins.rs:253-261), and the edges
      are the feature's. Form the row's `p` indices once, before the target
      loop, and the same on the warm-up replay. Tests as the request lists,
      and `marg_bench`'s shape at `T = 1` and `T = 9`. First: the smallest,
      and exact.

      **Review 2026-09-26.** The row update took the low parts'
      slice for every present target but sized the vector only when some
      feature binned, so a histogram restored from a 0.10.0 state (which
      carries none) panicked on its first all-NaN-feature row with a target
      (B1, a crash from the Rust API; the bank's null rule shields the CLI
      and Python): sized on the first row that could write a cell, which is
      also the rule the sharded path uses, so the two paths' states agree
      byte for byte (A7). `has_shape` now holds the offsets to the edges and
      the low parts to the cells (B2). The scratch row's four-slot minimum
      is counted, so `histogram_bytes` is the buffers to the byte at every
      width (B5). The design doc's cell count, the per-feature search and the
      "other two" said three values a cell (A4, B6). Tests: the brute-force
      tests drive `update_row`, the production update, where they drove the
      test-only per-target one; `the_split_gain_survives_a_fold` pins the
      product-of-ratios claim to 1e-9; `edges_from_gives_edges_a_histogram_accepts`
      is a proptest over junk, ties and a dominant weight;
      `fixed_edges_collapse_against_a_large_level`; and `shard_stream` now
      carries NaN features, all-NaN rows and all-absent targets, so the
      sharded `NO_BIN` paths run under the bit test.

<a id="task-123"></a>
- [x] 123. **E70: `marginal(cross_lags=...)`.** S–M. **Done 2026-09-25**,
      on schema 16 (unreleased), through the spec, the bank's two tables and
      Python; refused by name at construction. The default is bit-identical
      to the build before and 0.9% faster; one loop for both cases had made
      it 1% slower, until the `cross_lags` loop was kept out of line
      (PERFORMANCE §23). One cross lag of six takes 30% off the lags' cost. Verified: `n_serial`
      reads the autocorrelations alone (`serial_factor`, marginal.rs:368-420,
      from `lagcorr_xx` and `lagcorr_yy`), and `cxy`/`cyx` feed only
      `lagcorr_xy`/`lagcorr_yx`. The default, `None`, keeps every lag as
      today, and the kept moments are bit-identical. A state field with a
      serde default is still a layout change under hard rule 5: it rides
      schema 16 if it lands before 0.11.0, 17 after.

      **Review 2026-09-26.** `MarginalLags::has_shape` let a ring longer
      than the deepest lag, or one with fewer target rows than feature rows,
      through: the first read every lag a row too recent for good, the second
      panicked (B3, a corrupt state only); both refused. The module doc said
      the cross terms were two thirds of the lag work; §23 measured 30% (A6).
      Tests: `a_lag_state_with_an_overlong_or_uneven_ring_is_refused`,
      `a_compact_state_without_cross_lags_reads_with_every_lag` (the
      positional upgrade the comment claimed), the spec layer's
      `cross_lags_write_back_as_given_and_need_lags`, and the bins test that
      `cross_lags=[]` drops the two cross columns and moves nothing else.

<a id="task-124"></a>
- [x] 124. **The stream's fixed cost per feature per row at width** (from
      E74's measurement). S–M, and it helps every wide model. **Done
      2026-09-25** (PERFORMANCE §24): it was not the cast or the transpose
      first, but seven lookups by column name that scanned every column,
      quadratic in the width, and the marginal model keeping every pair's
      run for a window it did not have. At 10,000 features on one thread
      the moments-only call went from 818.7 to 192.2 ms at one target and
      from 2,141.5 to 617.7 ms at nine; the fixed part is `2.8` ns per
      feature per row where it was `13.1`, the transpose and the data
      summary. Tiling the summary was measured slower and left out. The caller's
      14–17 ns per feature per row that does not grow with `T` is outside
      the model (above). Profile the bank's path at `p = 10,000` — the
      cast, the tiled transpose (`feature_rows`, bank.rs:169), the per-row
      validity check, the data summary, the output — and fix what
      dominates; at `T = 1` it is four fifths of the row.

      **Review 2026-09-26.** The hash lookups were read against the
      scans they replaced (`find`, `key`, `has`, `first_readers`) and found
      the same answers; `first_readers`' doc now says where a reference
      sits in the role order (D12). `check_clocks`' "a temporal column can
      only be a clock" loop read target names and knew no reference, so a
      Datetime clock used as a table target's column or as its reference was
      refused later, with another message (D6): by column now, and a
      reference is "a relative_to reference"
      (`test_a_temporal_clock_is_not_a_target_or_a_reference_either`).

<a id="task-125"></a>
- [x] 125. **E72: `feature_moments="shared"` — a decision first.** M–L. The
      request says only `var_x` changes where a target is absent; not so.
      The batch identity it cites holds, but `sxy` is kept by a recursion
      centred on the feature's pre-row mean over the target's rows, and
      stepping it from the shared mean gives another `cov` wherever a target
      is absent on some learned rows: 1.8e-3 relative with a random tenth
      absent at `lam = 0.99` (a simulation of both recursions,
      2026-09-25), bit-identical only where every target is on every
      learned row. Under `"shared"`, `cov`, `var_x`, `corr`, `beta`, `t` and
      `n_serial` then all differ there: a different estimator, sound where
      the absence is not informative, which the docs would have to say.
      *Decision: build it as that estimator; build it only where it is
      exact (every target on every learned row, else per-target moments);
      or decline.* Either way the shared means are pairs (task 101), the
      runs go per feature (task 94's windowed read), and a `window` needs
      its own answer. **Decided 2026-09-29** (the user: "Do task 125",
      on the recommendation to build it as that estimator and document
      it). **The pair moments are built:** `feature_moments="shared"`
      (`FeatureMomentLayout::Shared`) keeps `p` means (pairs, task 101) and
      variances over every learned row and `p·T` covariances, unsplit and
      sharded, bins as they were; the pairs are `"per_target"`'s to the bit
      where every target is on every learned row
      (`shared_feature_moments_are_per_target_to_the_bit_where_every_target_is_present`,
      at one target and three, and the sharded stream); where one is
      absent, the feature's moments are its column's `EwCov` over every
      row to the bit and the covariance the stated recursion
      (`shared_feature_moments_are_the_features_over_every_learned_row`),
      measured moving `corr` by at most 0.0065 at a tenth absent. 2.7 times
      as fast at ten targets, 3.2 at thirty, level at one (PERFORMANCE
      §27). A window is refused by name: its subtraction reads a covariance
      centred on the pair's own mean. **The lags are built too:** the
      feature's autocovariance at each lag per feature with the row's mix
      (`MarginalLags::update_shared`, and the sharded job the same way),
      the cross terms and `cyy` per target with its own; the pairs are
      `"per_target"`'s to the bit with every cross lag, some and none, at
      one target and three, and sharded; with a target absent, the
      feature's lag correlation is a `"per_target"` model's whose target
      is on every row, to the bit
      (`shared_lag_moments_are_the_features_over_every_learned_row`). With
      lags at ten targets it runs 4.6 times as fast, 2.5 with one cross lag
      (PERFORMANCE §27).

<a id="task-126"></a>
- [x] 126. **E73: a wide `marginal` sharded across the pool (`shards`).**
      **Done 2026-09-25**, as a batch of rows per fork-join, not one per
      row (§11a has why). `marginal(shards=)` takes a count or `"auto"`,
      off by default; it is a setting, not state. `Marginal::step_sharded`
      advances every target's own numbers as the row arrives and holds
      each pair's share of it; a flush steps the pairs through the held
      rows in order, one `MarginalShard` per range of features, which the
      stream runs on the bank's pool. `online-core` keeps no threads: the
      caller passes the runner. Flushed when 256 rows (or 8 MB of features)
      are held, before a window snapshot, before the bins fold their scale,
      and at the end of each run of rows. The pair update, the lag updates
      and the bins' cell update became slice kernels shared by both paths,
      and the unsplit row got 7–50% faster with every checksum equal. Tests:
      `a_sharded_step_is_the_unsplit_step_to_the_bit` and four more in
      `marginal.rs`, `a_sharded_marginal_reads_the_unsplit_pairs` through
      the bank, and `tests/test_marginal_shards.py` (every count, chunked,
      saved and resumed, `fit(lf)`, the CLI). Measured (PERFORMANCE §25):
      at 10,000 features the model alone ran 1.8× at one target, 4.6× at
      nine and 5.4× with lags and bins; through the bank, 1.2×, 2.0× and
      4.9×, where the bank's own row work does not split. Not split: the
      bins' warm-up replay, once per group.

      **Review 2026-09-26.** Three findings changed behaviour. A window
      with the default `window_every` made `takes()` true on every row, so a
      sharded windowed model flushed -- one fork-join -- per row, the regime
      §25 measured as slower than unsplit, and `"auto"` did not know (A1):
      `auto_shards` sizes a windowed model's flush by the snapshot cadence,
      every row at the default, which no width can keep busy
      (`auto_does_not_split_a_window_snapshotted_every_row`, and a counting
      runner in `a_window_snapshotted_every_row_flushes_every_row` shows the
      per-row flush as a fact). `state()` and `pair()` with rows held were
      `debug_assert`s: a release-build caller got a state whose targets had
      advanced and whose pairs had not, unrecoverable after a restore (A3);
      `assert!` in every build, and the bank cannot reach it (every exit of
      `run_instance` flushes). `load` compared `shards` with the saved specs
      and refused a bank resumed under another count, which the docstring
      promised (F3): the count is left out of the comparison and the bank is
      built from the specs given, so it runs under theirs; given none, the
      saved one (`a_bank_saved_under_one_shard_count_loads_under_another`,
      `test_a_saved_bank_resumes_under_another_count`). Smaller: every flush
      copied the whole lag ring through `to_vec` and `set_ring` (A8, B7) --
      the ring keeps its newest rows and the held ones are pushed; a dead
      `decay(pending_lam)` in `freeze` (A9); `Deferred` is boxed like `lag`
      and `bins` (A10); the batch-size comment's "within 6%" holds at nine
      targets only (A5); the bench's checksum left out `lagcorr_yx` and
      `bin_var_y` (E3) and hashes every pair field now; §25's "about one
      held batch per thread" is not a bound under rayon's stealing, and says
      so (E2). Tests that could not fail: `"auto"` at 60 features is one
      range in three of the four shapes, so those legs compared the model
      with itself (F2) -- `test_auto_splits_at_a_width_that_keeps_the_pool_busy`
      runs at 1,000 features, where `auto_shards_at_the_python_suites_widths`
      pins that every shape splits; the CLI test compared `n_eff` alone,
      which no shard computes (E1) -- it saves the state and compares the
      pairs, under `"auto"` and `4`; the Rust `"auto"` assertion failed on a
      one-thread pool (D9, E4). A panic inside a shard leaves the model
      inconsistent without a `broken` mark (E5); noted, not built: a panic in
      the kernels is itself a bug. New edge tests: a label delay, a halflife
      grid and a session reset under a window; closed groups at a session
      change and by a monotone key; one-row chunks; forty groups on two
      threads under `"auto"` (a child process); scoring holds no rows; a save
      inside the bin warm-up; a cadence longer than a batch; and in Rust a
      restore mid-warm-up, a refusing budget with rows held, the count
      changed with no flush between, `p = 1` with two shards, a total gap
      with rows held, all-absent targets after a wipe.

<a id="task-127"></a>
- [x] 127. **E74: a chunk run pair-major — not as specified.** Its reason
      is the intercept, which is not the model's (task 124, which took it
      from 13.1 to 2.8 ns per feature per row); the bank
      already hands the model one contiguous row (`feature_rows`), so
      there is no per-row gather to remove; and its lag, `x_j[r − ℓ]`, is
      not the model's: the ring holds only rows that taught something
      (weight above 0, every feature finite, marginal.rs:1030-1040), and
      the caller's weight is 0 on its warm-up and purge rows. What may
      remain is the state traffic behind the slope, `p·T` pairs' state in
      and out of cache on every row: measure it after 122, 123 and 125,
      before designing a loop order that every clock event, reset, window
      snapshot, group close and `label_delay` replay would have to split.
      *Recommended: wait for that measurement.* **Done 2026-10-10**
      (PERFORMANCE §39): the slope is the model's (0.58 ns a pair-row,
      moments, `p = 10,000`) and the intercept the stream's; the state's
      traffic is about 14% of the slope (20% with lags): resident against
      streamed, 2.06 against 2.37 cycles a pair-row, against a floor of
      1.75. E74's register-resident run measured about 5% faster at best
      and half as fast narrow: not built. An L1 tile in the unsplit `step`
      (+16% at T >= 10, -5% at T = 1, tuned to this machine's 128 KB L1
      where many x86 cores have 32-48 KB) not built either; a user who wants
      the tiled order gets it by setting `shards` to a count, even on one
      thread. Built, bit for bit: a sharded flush steps one target at a
      time (`run_by_target`, 0.76-0.81 of the time on one thread at 30
      targets) and cuts its ranges on even features (`shard_cuts`,
      0.85-0.91).

<a id="task-128"></a>
- [x] 128. **`EwCov` keeps its runs without a window** (found by task 124,
      PERFORMANCE §24). S–M, a confirmed regression of task 94, unreleased.
      **Done 2026-09-25:** `Runs::off`, a state flag read as on where a
      state lacks it, and `EwCov::without_runs`, taken by every owner
      without a window (`ew_cov`, `ewridge`, `lasso` and `ew_class` without
      one; `huber`, `quantile` and the slot metrics always); `marginal`
      builds its runs off too, so a wide one without a window no longer
      allocates 16 bytes a pair for them. A test per owner pins the choice.
      `core_bench`'s `ewridge`: +15% at 5 features, +17% at 20, +4% at 50,
      +10% solving every 25 rows; the goldens are unchanged.
      Every covariance accumulator tracks each slot's run on every learned
      row, and only a window reads them (`crate::truncated`); `core_bench`'s
      `ewridge` is 18–20% faster at 5 and 20 features without the tracking,
      4% at 50. The owner knows whether it has a window, the accumulator
      does not: a state flag with a default of *on* (so an old state keeps
      tracking, the safe side), set off at construction by every owner
      without a window, kept through resets and the blend's rebuild
      (`gaps.rs`'s `empty`), ahead of `pending`, which must stay last. Rides
      schema 16. Tests: a model without a window keeps no runs, one with a
      window reads a held slot as today, and the numbers are bit-identical.

      **Review 2026-09-26.** No `restore` re-derived the flag from the
      cfg (C4): a 0.10.0 state, which has no flag, resumed without a window
      kept tracking for the rest of its life, this task's saving lost after
      every resume; and a state whose runs are off restored under a window
      read every held slot as moving, the task-94 defect back with no error.
      Every owner's `restore` now sets the runs off without a window and
      refuses a windowed state whose runs are off (`ew_cov`, `ewridge` and
      its twin, `lasso`, `ew_class`, `robust`, `hmm`); `EwRidge::restore`
      also gained the ring-against-cfg check its three siblings had (C5); and
      `hmm`, a windowless owner this task's "every owner" missed, keeps no
      runs (C3). The docs of `held_from`, the `runs` field and a test still
      described the weight rule the 2026-09-25 review replaced (C6), and
      `window_every` was documented as learned rows where the cadence counts
      every row, weight-zero rows included (C7): the docs say so now
      (`a_zero_weight_row_counts_toward_the_snapshot_cadence`). Two entries
      of `scripts/mutants_equivalent.toml` excused mutants that are not
      equivalent: `EwCov::skip`'s `*`→`/` skipped the prior's ageing on an
      empty blocked Gram (C1), `Lasso::step`'s `<`→`<=` moved `lam_selected`
      on a tie two saturating penalties give (C2); a test each kills them
      and the entries are gone.

<a id="task-129"></a>
- [x] 129. **`marginal`'s bins budget counts three values a cell where
      there are four** (found 2026-09-26, checking the memory caps at the
      user's question). S, a confirmed regression of task 101, unreleased.
      `BinCfg::validate` refused a histogram past 256 MiB (`MEMORY_BUDGET`,
      `margbins.rs`) counting `3 × cells` doubles: the weight, mean and
      spread a cell kept in 0.10.0. Task 101 gave each cell a fourth, the
      mean's low part (`mean_lo`), and left the count at three, so a
      histogram the check allowed could take 4/3 of the budget, 341 MiB. No
      test held the estimate to the allocation. Three older gaps of the same
      kind, all in 0.10.0: the warm-up hold counted a held target at 8 bytes
      where an `Option<f64>` takes 16, and a held row's own size not at all;
      the peak at the warm-up's last row is hold plus histogram, each allowed
      256 MiB; and the cap is per model, which no document said.
      **Done 2026-09-26**, as the user approved after
      the plan's first version: `CELL_VALUES`, the per-cell vector count,
      is what `MarginalBins::new` takes its vectors from, in one array, and
      what `histogram_bytes` counts, beside the edges, their lists, the
      offsets and the row's offsets; `hold_bytes` counts a held row from its
      types, `HeldRow` moved beside it; the hold is reserved at exactly
      `bin_warm_rows` rows, again after a restore; learned edge lists are
      trimmed to the edges found. The peak is documented, not refused: a
      check on the sum would refuse 10,000 features, 50 targets and 16 bins,
      E73's full profile, whose histogram alone fits (244 MiB). "Per group,
      and per halflife" is in the docstring, the error message and
      MARGINAL-LAGS-AND-BINS. A bank-wide bound was not built: groups arrive
      with the data, so a build-time check cannot count them, and every
      model's state grows with them, not the bins' alone. Tests:
      `the_histogram_budget_is_what_its_buffers_take` and
      `the_hold_budget_is_what_the_held_rows_take` hold each estimate to the
      bytes the buffers report, to the byte;
      `the_state_holds_nothing_the_budget_does_not_count` fails when the
      state gains a number no count covers; `the_budget_refuses_at_the_real_size`
      pins the boundary. Restoring the three-value count, the eight-byte
      target or the growth by doubling each fails them. Visible: a spec
      whose histogram needs between 256 and 341 MiB, or whose hold
      undercounted its targets past the budget, is refused where it was
      allowed.

      **Review 2026-09-26.** Reserving the hold at exactly `warm_rows`
      on the first held row charged every group the whole hold up front --
      64 KB a group at the default, 640 MB across ten thousand short groups
      -- where doubling had cost at most twice the rows held (A2): the hold
      grows by doubling, capped at `warm_rows`, so `hold_bytes` stays the
      most it can reach and a short group holds only its rows
      (`a_short_group_holds_only_its_rows`). The spec doc said the budget was
      per group; it is per model instance, every halflife of a grid included
      (D11).

<a id="task-130"></a>
- [x] 130. **Two window snapshots undercount what they hold** (found
      2026-09-26, checking the other memory estimates after task 129). S,
      confirmed regressions of tasks 101 and 102, unreleased. A window's
      ring thins or refuses past its budget (`window_budget`, 256 MiB a ring
      by default) by adding up each snapshot's `Footprint`, a sum over its
      fields written by hand beside the struct. Two sums fell behind their
      structs: `Cross` (`gaps.rs`), which every `ewridge` and `lasso`
      snapshot clones whole, gained `m_lo` and `my_lo` in task 101, `k + T`
      doubles uncounted; and `MarginalMoments` gained `rows`, one count per
      target, in task 102. **Done 2026-09-26**, at the user's word: both
      footprints count them. `window::assert_footprint_counts_every_vector`
      walks a snapshot's state and holds its footprint to every vector of
      numbers in it, from two snapshots of different sizes, so a vector left
      out shows as a difference that grows with the vectors. That second
      snapshot was needed: from one alone, the scalars `ewridge`'s
      footprint counts equalled the low parts it missed, and the check
      passed. A test per snapshot type (`ew_cov`, `ew_class`, `ewridge`,
      `lasso`, `marginal`); the `ewridge`, `lasso` and `marginal` ones fail
      without the fix, and `ew_cov`'s and `ew_class`'s guard a footprint
      that needed none; the spread's ring (`resid_window.rs`) has its own. A ring near its budget now thins or
      refuses a little sooner than before the fix: what it counts is what it
      holds.

      **Review 2026-09-26.** The two-size check documented "every
      vector longer in the larger" and did not enforce it, so a vector sized
      by a dimension neither snapshot varied could still hide under the
      scalar slack (B4): it walks both snapshots in step and refuses a vector
      that does not grow (`the_footprint_check_sees_a_vector_that_does_not_grow`),
      which made the `lasso` test vary the path and the two-Gram `ewridge`
      test (`the_window_footprint_counts_every_gram`, C missing 7) vary the
      targets. The ridge and lasso snapshots cloned the means' low parts,
      which the window's subtraction never reads (C9): cleared in
      `Acc::snapshot`, `k + T` doubles a snapshot the ring no longer holds.
      `Moments::footprint` counts its three numbers (C10). This entry said a
      test per snapshot type failed without the fix; `ew_cov`'s and
      `ew_class`'s did not need one (C11).

<a id="task-131"></a>
- [x] 131. **`marginal(bin_budget=...)`: the bins' memory limit, set per
      spec** (the user, 2026-09-26, after task 129). S. The warm-up hold and
      the histogram were each refused past a fixed 256 MiB. `bin_budget`
      sets that limit in MiB for both, per spec: 256 when absent,
      `float("inf")` (`"inf"` in JSON and TOML) for none; it needs `bins` or
      `bin_edges`, and a budget that is not a positive number is refused by
      name. **Done 2026-09-26:** `BinCfg::budget_mib`, the last field and
      the only one that skips, so a model without it keeps its bytes
      (`a_bins_budget_round_trips_in_both_encodings`); the spec's
      `bin_budget` a `Num`; the Python builder and the CLI's TOML take it.
      The error names the budget and the setting. `gram_block_rows`'s cap,
      the same kind of build-time check, stays fixed at 256 MiB: the user
      asked about the bins'. Tests: `the_budget_can_be_set` in both
      directions, infinity and the bad values;
      `test_bin_budget_sets_the_limit` through the bank, with a hold of 313
      MiB refused by default, built at 400 MiB and refused again at 300.

      **Review 2026-09-26.** Tests at the spec layer, which had only
      `check()` being `Err` for a budget without bins:
      `bin_budget_is_held_to_what_it_can_mean` (zero, negative, `-inf`, NaN
      and a string refused by name, beside given edges accepted, `"inf"` no
      bound), `shards_read_a_count_or_auto_and_refuse_the_rest` and
      `shards_round_trip_through_json_and_msgpack` (D missing 7, 8), and the
      Python refusal of a tiny budget beside given edges. The builders'
      finiteness check did not walk a dict, so `bin_edges={"x": [inf]}` was
      named by serde as `model` (F8).

<a id="task-132"></a>
- [x] 132. **Review 2026-09-26 of the unpushed changes** (the user:
      "a deep review iterating through each file and reviewing the code in
      full while examining the changes as well; also examine the tests,
      find edge cases that may be missing"). Scope: every change since
      `ae34d91` -- tasks 101-103, 107a, 108, 109's items, 121-124, 126,
      128-131. Protocol as docs/REVIEW-2026-09-18.md §0: six reviewers by
      area (A `marginal.rs`; B bins, lags, runs, window; C the covariance
      accumulators and the mutant excusals; D spec, targets, arrow; E stream
      and bank; F the Python package and tests), each finding re-derived,
      reproduced with a failing test before its fix, fixed in two batches
      (core, then the crates above it), gated unpiped. **Done 2026-09-26**;
      each finding is recorded under its task by the reviewer's letter and
      number, with the test that now pins it. Two crashes (B1, D1), one
      silent wrong output (D3), two infinities from inputs inside the bound
      (G1, a row absorbed from infinitely far; G2, a quantile nudge past its
      row; G3, an own mean rebuilt from two level-sized numbers -- schema
      17; all found by the widened proptest), a per-row flush regime `"auto"` did
      not know (A1), a memory charge per group (A2), a documented resume that
      refused (F3), eight lost guards or wrong excusals (A3, B2, B3, C1, C2,
      C4, C5, D4), four misleading messages or refusals (D2, D5, D6, D7,
      F4, F7), a test helper that did not enforce its rule (B4), three tests
      that could not fail (E1, F2, D9) and a dozen doc mismatches. Nothing
      changed in the numbers a model returns but `hit_rate` under a ratio
      target, the distances `micro` and `kmeans` report where their squares
      overflow, a quantile `robust` fit's step on a row whose leverage
      exceeds its weight share, and the last bits of a gappy target's fit
      under `ew_ridge` and `lasso`. Left as noted: E5 (a panic inside a shard leaves no
      `broken` mark) and task 105.

<a id="task-133"></a>
- [x] 133. **The release workflow tags what it published, after
      everything ran** (the user, 2026-09-27, after the 0.11.0 tag failed:
      "change the release workflow so that GitHub tags a release after
      everything runs. We want to keep the rehearsal for testing"). S–M.
      **Done 2026-09-27.** `release.yml` has no tag trigger any more: it is
      dispatched on `main`, and a `publish` input (off by default) turns
      the rehearsal into a release. A `version` job runs first
      (`scripts/release_version.py`): the six places agree, the CHANGELOG
      has the section, and, publishing, the run is on `main` and the tag is
      new -- refused before the hour and a half of builds, since the tag of
      a spent version can never be reused. A `ci` job calls the whole of
      `ci.yml` (now `workflow_call`-able, its concurrency group per calling
      workflow so a release and a push on `main` do not cancel each other),
      so the suites that failed 0.11.0 hold back the upload with the wheels,
      the state hand-off and the Polars legs. The PyPI upload keeps its
      approval; the `tag` job then tags the tested sha, annotated with the
      CHANGELOG's section, and `release` makes the GitHub release from the
      same notes. So a tag names exactly what was published. The README's
      PyPI links point at the tag the run will create (the sha when
      rehearsing). `docs` in `ci.yml` needs `pages` and `id-token`, which
      the call must grant; it runs on a push alone, so the grant is inert,
      and `tests/test_release_workflow.py` holds the workflow's shape and
      the version script to all of this (the schema check with
      `check-jsonschema` passed). RELEASE-READINESS's steps now read:
      compare, version, changelog, gate, commit and push, dispatch with
      `publish`, approve, verify. No rehearsal before a release (the user
      asked whether one was needed, and it is not): the publishing run does
      all a rehearsal does before the approval, and a failure there leaves
      nothing to undo. The rehearsal stays for testing a change to the
      workflow. Neither tries `tag` or `release`, which run after the
      upload; a failed one is re-run on the same sha.

<a id="task-134"></a>
- [x] 134. **E75: `marginal`'s lead/follow reading does not hold against a
      forward-looking target — correct the docs, change no code.** S; no
      decision. **Done 2026-09-27:** (1)–(6) below as written, the test as
      `test_a_timely_feature_beats_corr_one_row_on_against_a_forward_target`
      in `tests/test_marginal_lags.py`, and the dated note in
      MARGINAL-AT-WIDTH's table. Asked 2026-09-27 by the caller of MARGINAL-LAGS-AND-BINS and
      MARGINAL-AT-WIDTH (factor_selection, `~/dev/model`), in a request file
      that was checked, recorded here and deleted. **The reading:** seven
      places say a feature whose `lagcorr_xy[0]` exceeds its `corr` follows
      the target (a late-sampled column): the README's comment on
      `bank.marginal()`'s columns (1842–1843), the `marginal()` docstring in
      `_bank.py` (1042–1044), the spec's docstring in `_spec.py`
      (2980–2982), MARGINAL-LAGS-AND-BINS' table of columns (114) and its
      paragraph after the tests (131–134), E66's entry in this plan
      (919–921) and E66's row in ENHANCEMENTS (497). **Why it fails:** a
      forward-looking target is built from the rows after its own,
      `y_t = Σ_{k≥1} w_k r_{t+k}`, and a timely feature holds each row's
      news from that row on, `x_t = Σ_{k≥0} v_k r_{t−k}`. `corr` is
      `C(x_t, y_t)` and shares no return; `lagcorr_xy[0]` is
      `C(x_t, y_{t−1})`, and `y_{t−1}` starts with `r_t`, so it gains
      `v_0 w_1 Var(r)`. Every timely feature built from the target's own
      news shows `lagcorr_xy[0]` above `corr`, at any halflife, so the
      inequality separates nothing. It holds for two series that describe
      the same moment. **What it did:** the caller's late-sampling screen
      flagged 527 of 6,015 pairs on sixty days of Binance one-second klines,
      every one a correctly sampled return or signed-flow factor (BTC's 2 s
      return average against the 10 s target: `lagcorr_yx[0]` 0.075, `corr`
      0.105, `lagcorr_xy[0]` 0.305); read at four offsets for where the
      slope jumps, it flags none. **Checked here 2026-09-27:** the algebra;
      each of the seven places says what the request quotes; and its test
      passes against this build, in 0.75 s, with the closed forms 0.2544
      and 0.4173 it quotes.
      **To do.** (1) The README comment and both docstrings, one text: "the
      feature now against the target `l` rows back, and the target now
      against the feature `l` rows back, at each of `cross_lags` (every lag
      by default). For two series that describe the same moment, a feature
      whose `lagcorr_yx[0]` exceeds its `corr` leads the target, and one
      whose `lagcorr_xy[0]` does follows it. That reading does not hold
      against a forward-looking target, one built from the rows after its
      own. There the target `l` rows back is built partly from the
      feature's newest `l` rows, so a feature built from the same news shows
      `lagcorr_xy` above `corr` however it is sampled. Absent under
      `cross_lags=[]`." (2) MARGINAL-LAGS-AND-BINS' row for `lagcorr_xy`:
      "`C(x_t, y_{t−ℓ}) / (sd_x · sd_y)` — the feature *now* against the
      target ℓ rows *ago*. For two series that describe the same moment, how
      far the feature follows the target. Against a forward-looking target
      it also holds what the two share: the target ℓ rows ago is built
      partly from the feature's newest ℓ rows." (3) Its paragraph after the
      tests: the pair is read with the target's construction in hand; for
      two series of the same moment `lagcorr_xy[0]` above `corr` means the
      feature follows; against a forward-looking target it says nothing
      about timing, since every timely feature built from the target's news
      shows it; telling late from timely takes more offsets and a judgment
      over independent blocks, and one caller keeps `cross_lags=[1, 2]` and
      asks where the slope of the correlation jumps. (4) A dated note, not a
      rewrite, in E66's entry here and its ENHANCEMENTS row: "2026-09-27:
      the lead/follow reading holds for two series that describe the same
      moment, not against a forward-looking target (E75)." (5) A test, the
      caller's: `r` i.i.d. normal; `x` the EW average
      `x_t = (1 − a) x_{t−1} + a r_t`, `a = 1 − 2^(−1/2)`; `y_t =
      Σ_{k=1..L} w_k r_{t+k}`, `w_k = (1 − λ) λ^(k−1)`, `λ = 2^(−1/10)`,
      `L = 70`, seven of its halflives; 100,000 rows through `marginal(lam=1.0, lags=[1, 2],
      cross_lags=[1, 2])`; then `|corr| < 4/√n`, `lagcorr_xy[0]` within
      that of `√(a(2 − a)(1 − λ²)/(1 − λ^(2L)))` (0.2544),
      `lagcorr_xy[1]` of that times `1 − a + λ` (0.4173), and every
      `lagcorr_yx` below it. (6) A Docs line in the CHANGELOG. **Not
      asked:** a late-sampling statistic in the library (the rule needs a
      block design and a multiple-testing margin, the caller's choices), a
      change to the terms or their orientation (they are right, and the
      test holds them to the closed form), or a new `cross_lags` default.
      **On the caller's side:** MARGINAL-AT-WIDTH line 66 says it reads
      `lagcorr_xy[0]` and `lagcorr_yx[0]`; it now runs `cross_lags=[1, 2]`,
      about 3 % more state a pair at its defaults, the time not yet
      measured. A dated note there fits.

<a id="task-135"></a>
- [x] 135. **S18: Bartlett weights for `marginal`'s serial rule.** S. The
      user accepted it 2026-09-28 (task 115 (f)). `"truncated"` sums
      `rho_x(l)·rho_y(l)` over lags at unit weight, which is not positive
      by construction, so a mixed-sign pair drives the factor to or below
      zero, where it now reports NaN (review 2026-09-12, S18). Newey–West's
      weights `1 − l/(L+1)` keep it positive: a new `serial_rule`, held to
      `statsmodels`' `cov_hac` with `weights_bartlett` in
      `tests/test_second_opinion.py` and to a `faer`-free longhand in Rust.
      No state: the weights apply where the factor is read. **Built
      2026-09-28:** `serial_rule = "bartlett"`, `L` the longest kept lag,
      sparse lags at their own `l`; positive by construction only for a
      positive-definite sequence of products over every lag `1..=L` (the
      factor is then `1ᵀC1/(L + 1)`), so a factor at or below zero is NaN
      as under `"truncated"`. `the_bartlett_factor_is_its_definition`
      (S18's pair, `+0.8`/`−0.8`, reads 0.36 where `"truncated"` has none)
      and `TestTheBartlettSerialFactor`, which takes the weights from
      `statsmodels`' `weights_bartlett` and the autocorrelations from its
      `acf`.

<a id="task-136"></a>
- [x] 136. **S19: the Gram's target moments under a `window`.** S–M. The
      user accepted it 2026-09-28 "as long as it doesn't add much extra
      size". Under a window, `target_means`, `target_vars` and
      `target_n_kish` are `None`, since the snapshots hold no target
      moments; snapshotting them restores them as the window's. The bytes a
      snapshot gains are measured against its Gram before building, and
      the entry records both. **Built 2026-09-28:** each snapshot carries
      the target moments decayed to it (`TargetMoments::decayed`: the
      means and variances as they stand, `Q` at `lam²`), and the view
      truncates them by the pooling identity the Grams use, `m_R = m −
      r·d`, `v_R = g·v − r·v_u − r·g·d²`, `Q_R = Q − f²·Q_u`
      (`TargetMoments::truncated`); the export's Kish count reads each
      part's own, windowed, target weights. The cost is exactly `3·T`
      doubles a snapshot (`the_target_moments_add_three_doubles_a_target_to_a_snapshot`):
      +0.1 % at `k = 50`, `T = 1`; +0.8 % at 50 and 10; +1.7 % at 10 and 1;
      +4.5 % at 5 and 1; +5.5 % at 10 and 5; +15.8 % at one feature, whose
      snapshot is 152 bytes. Held to a longhand over the in-window rows,
      a gappy target included (`the_windowed_target_moments_are_the_rows_inside_it`),
      and to `statsmodels`' `DescrStatsW` (`TestTheGramIsTheWindowsToo`). A
      snapshot from before carries none, and the export says `None` until
      the ring rolls over.

<a id="task-137"></a>
- [x] 137. **C18: a lag ring under a `window` in `marginal`.** M. The
      user accepted it 2026-09-28, with the same condition on size. Today
      `window` with `lags` is refused (review 2026-09-12, C18), since the
      lagged co-moments would come from the whole history beside windowed
      variances. The lag moments are normalized moments on the same decay,
      so a snapshot of them truncates as everything else does (V12). The
      bytes a snapshot gains, `lags × pairs` and the target side, are
      measured first; held to `statsmodels`' `acf` on the in-window rows,
      as C18's test reads. **Measured 2026-09-28, and held for the user's
      word on size:** a windowed lag moment is the sum of the increments
      made inside the window, so every lag moment must be in the snapshot
      -- `L·T` for `cyy`, `L·p·T` for `cxx` and `2C·p·T` for `cxy` and
      `cyx` (`C` the cross lags, every lag unless `cross_lags` names fewer)
      -- beside today's `MarginalMoments` of about `(3p + 5)·T` doubles.
      The lag moments add about `(L + 2C)/3` of the snapshot's size: +100 %
      at one lag, +500 % at five with the default cross lags, and +167 % at
      five with `cross_lags=[]`, which `n_serial` does not read. Options:
      build it at that price; build it only with `cross_lags=[]` under a
      window (the serial correction without the lead/lag by-product); or
      keep the refusal. **Decided 2026-09-28 (the user: "For c18 implement
      but with an api parameter that documents the impact") and built:**
      the spec's `window_lags` (`po.spec.marginal(window_lags=True)`,
      default `False`) accepts the price, documented with the formula and
      two examples; without it, `window` with `lags` is refused with the
      spec's own numbers (`4 doubles beside the 8 it holds without them,
      1.5 times the size` for one feature, one target, one lag), and
      `window_lags` without both is refused. The core takes the pair: each
      snapshot holds the lag moments (`LagMoments`, `marglag.rs`; counted
      in `window_budget`), and a pair's lag moment under the window is `(W·C
      − f·W_u·C_u)/W_R` for its target. Held exactly to an unwindowed
      twin's sum form (`a_windowed_lag_moment_is_the_increments_inside_the_window`),
      to `statsmodels`' `acf` of the in-window rows at 0.02 across a `phi`
      change (`TestAWindowedLagIsTheWindows`), to the documented size
      (`the_lag_moments_add_what_window_lags_says`), and bit for bit sharded
      (`a_sharded_step_is_the_unsplit_step_to_the_bit`, a new window-with-
      lags case). A state's snapshots are now checked for shape on load, the
      pair moments' included, which were not.

<a id="task-138"></a>
- [x] 138. **README rewrite against the current API, requested 2026-09-29.**
      The user's words: "After the gate commit rewrite the readme after
      analyzing the current state of the api and features. Be sure to
      follow phrasing and writing guidelines". Task 89's rewrite is six
      days old, and 363 lines have come in since, one task at a time. So
      this pass runs `docs/WRITING.md` §6 again, against the code as it
      stands at `5e96018`.

      *Measured before* (a counting script to §6's rules: paragraph
      boundaries, a list item its own paragraph, a bold rule split from the
      sentence after it, a sentence allowed to open with a lowercase name):
      13,282 prose words, 709 sentences averaging 18.7 words, 34 of 35+
      words and 2 of 45+, no cost words (the one hit is "the
      point-biserial correlation"), 43 tables, 61 python blocks, 10
      top-level sections.

      *The map.* The ten sections stay; three pieces move, each to the
      section whose heading it serves (§2), and `←` says where from.

    | section | subsections |
    |---|---|
    | Introduction | The idea · Four words · Install · A first fit · What you can rely on |
    | How a bank sees a stream | What a spec names · Time and decay · A hard window (← `ew_cov`'s section: five models take `window`, and a reader of `ewridge`, `lasso`, `ew_class` or `marginal` never looked there; it sits beside the decay it is compared with) · A local fit along any feature · Convergence without a decay · Groups · Weights · Warm-up · Labels that arrive late · Nulls, and three ways to hold a row back · Row order and the two guarantees · Series that tick at their own times (← Running a bank, whose heading is the three ways to run one: this is how several series become one stream, beside `embargo`) |
    | Running a bank | As a query · In a loop (+ `ModelBank.fit` and `fit_predict_batches`, which the memory table uses before any section shows them) · Output as Arrow · Outside a live Python process |
    | Saving, loading and serving | Save and load · Serving without learning · What a state file holds · Reading a state without this library |
    | Reading the fit | Coefficients · Output field names · The running sums behind a fit · One row per finished group (its accumulate-only pass ← `bank.fit`) · Reading a correlation matrix |
    | Diagnostics, selection and evaluation | Per-row diagnostics (+ which models take them: the ten that predict a target; + each switch's tuning keyword) · Conformal intervals · Evaluating an output frame · Evaluating a stream too large to hold · Data whose truth is known |
    | Models | the two tables · Linear models · Moments and correlation (`ew_cov` without `window`, which it now links to) · Clustering and classification · Sequential tests and regimes |
    | Performance | Throughput (re-measured: the tables date from 2026-09-06) · Memory: which calls stream · Tuning memory with Polars' own settings · Chunk size · Parallelism · Against scikit-learn |
    | Scope and integrations | What this is not · Pathway · Databases: DuckDB and ADBC |
    | Versions, testing and development | Versioning and the Polars pin · Testing · Development · License |

      *Constraints held.* Every anchor another file links to keeps its
      heading text (`llms.txt`: introduction, how-a-bank-sees-a-stream,
      models, against-scikit-learn; `docs/RUNNER.md`, `docs/REGIMES.md`,
      `docs/OUTPUTS.md`, `docs/PERFORMANCE.md`, `docs/README.md` and
      `scripts/outputs_doc.py` use eighteen more). Every `####` under
      *Models* is a model, with its *API:* and *Rust:* lines, and the model
      table keeps its header. Every python block runs in the namespace
      `tests/test_production_hardening.py` gives it.

      *The analysis.* Four agents checked the model families, Performance,
      Scope and Versions against the code, and every finding acted on was
      re-derived first; the introduction and the shared sections were
      checked by hand. What the README said that the code does not:

    | where | it said | the code |
    |---|---|---|
    | the model table | `huber` and `quantile` "solve", so converge to the batch answer in any row order | each row's weight comes from the fit before it: 9.2e-03 between two orders with outliers. A sixth kind, *reweight*, is theirs |
    | the model table, `ewridge` | halflives are fitted from the same running sums | each halflife keeps its own |
    | the accumulators | every one is a mean, and every second moment centred | `rls`, `ftrl` and adagrad keep decayed sums; `rls` a raw one |
    | `ftrl` | river's no-halflife recursion as the rule, "agrees with river to 1e-12" | the penalty scale `m` under a halflife (C24); river and Vowpal Wabbit hold without one |
    | `holt` | "let `emit_selected` choose" against a real model | `emit_selected` ranks one spec's own slots; `seqtest(a=, b=)` compares two specs |
    | `sgd` | `l2` on every coefficient; `coef_min=inf` for no bound | not on the intercept; `-inf`, and `inf` is refused |
    | `quantile` | the band's inside is `\|r\| ≤ h`; no warm-up | `\|r\| < h`; least-squares rows until three per coefficient |
    | `kalman` | the halflife drives the noise estimate | and the standardization |
    | `deco` | sums over every feature; one weight | a column with no spread left out; a weight per value; `rho` held at weight 0 under `"linear"` |
    | `marginal` | `n_serial`'s bracket as one rule; one search per pair; 4.6× with lags; no `window` | three rules; one per feature; 4.6× with no cross lags; `window`, `window_lags` |
    | `rcov` | `preavg` over `⌊θ√n⌋` less the bias | that is `psd=False`; the default is `⌈θ·n^0.6⌉` with no bias term |
    | `ew_class` | "full" factorizes every class every row, 0.9M rows/s at 400k rows | only the row's class, outside a window; the figures were 200k rows, before the caching |
    | `micro` | the nearest summary that can take the row; ARI 1.000 "with the `eps` above" | the nearest established one, then the nearest other; `eps` per shape |
    | `corrchange` | size and power "held to" WKD's tables; `γ = 0` "keeps the size" | measured against them (0.031, 0.552), tested near 0.05; 0.04 to 0.09 |
    | `hmm` | `state` the most probable filtered state; one limitation | the argmax of `p1`; the docstring's two |
    | `bocpd` | `prune_below` makes it finite, flat at the default; the run length dates a correlation break in one to three rows | `max_run` bounds a stationary stream; 98 rows under `"gaussian"`, never under `"diag"` |
    | `kmeans`, `micro` | a quiet feature counts 58 times its history | `2^(20/8)/0.1` is 56.6 |
    | the output records | `ewridge`'s fields without `settled_frac`, `withheld_reason`, `support_coef`; `summary()` without its five readiness columns | both, since 0.9.0 |
    | the grammar | no `__l{lambda}` | lasso's path |
    | Throughput, Parallelism | 2026-09-04 and -06 figures | `ewridge` k=20 4.12M → 2.53M; the grid 12.3 s → 32.7 s (PERFORMANCE §28) |
    | Install | 19 MB to download, 59 installed | 8–10 and 26–37 (0.12.0's wheels) |
    | Testing | 976 and 3,265 tests; numpy and river as oracles | 1,149 and 3,597; eight Python libraries and `faer` |
    | Versions | `~=0.9.0`; a cap is a patch *and* narrowing is breaking | `~=0.12.0`; narrowing is breaking but for the cap |
    | Scope | the Pathway operator runs in a pipeline; ADBC's trap is a second query | the example runs the operator over batches; re-executing a cursor, both queries wrong |

      Missing features, now documented: `ModelBank.fit` and
      `fit_predict_batches`, `closed_groups(drop=)`, the diagnostics' tuning
      keywords and which models take the diagnostics, `po.eval.unpack`,
      `po.corr.equicorr_row`, `window` on five models, `window_lags`,
      `serial_rule="bartlett"`, `strict_binary`, `select_halflife`, `p0`,
      `share_p`, `max_clusters`, `update_every`, `psd`, `transition_prior`,
      the Gram's target moments under a window. Docstrings with the same
      stale facts were corrected with them: `ewridge`, `sgd`, `holt`,
      `ew_class`, `marginal`, `deco`, `corrchange`, `bocpd` (its example
      dated a run on a global row index across interleaved groups),
      `ModelBank.gram` and `ModelBank.summary`, and two comments on `rcov`'s
      window. `huber`'s and `lasso`'s order exemption is task 139.

      **Done 2026-09-29.** Measured the same way before and after, on
      paragraph boundaries:

    | measure | before | after |
    |---|---:|---:|
    | prose words | 13,282 | 14,776, the added facts |
    | sentences of 45+ words | 2 | 0 |
    | sentences of 35+ words | 34 | 2, both 35 words |
    | mean sentence, in words | 18.7 | 18.1 |
    | cost words | 0 (one false hit) | 0 (the same false hit) |
    | tables, as rendered | 43 | 48 |
    | top-level sections | 10 | 10 |
    | python blocks, all running | 61 | 61 |

      Accounted as a diff: every number, backticked name and link the old
      text had is in the new, in PERFORMANCE §12 or §28, or replaced by a
      measurement or a correction named above. `scripts/doc_structure.py`
      passes on every tracked file, and GitHub's renderer shows 48 tables,
      88 code blocks and every in-page link landing.

<a id="task-139"></a>
- [x] 139. **`ModelBank.fit`'s order exemption, narrowed to `ewridge` and
      `rls`.** Found by task 138's analysis. `fit` skips
      `OrderNotGuaranteedWarning` where the state it leaves cannot depend on
      the row order, and `huber` and `lasso` at `lam=1.0` were on that
      list. `huber`'s 6.7e-16 was measured on rows no residual reached
      `delta * sigma`, so its reweighting never ran: with one row in ten
      lifted by 5, a shuffle moves its coefficients by 1.05e-02. `lasso`'s
      path points commute to rounding, but `lam_selected` ranks them by
      out-of-sample error, and shuffles moved the penalty it selects from
      0.01 to 0.1 and to 0.001. Tests first: both specs join
      `test_what_is_not_order_free`, which failed for both on the old list,
      and two premise tests show the reweighting firing on dozens of rows
      and the selection moving while every coefficient holds to 1e-12. The
      list, its measurement comment, the warning's docstring, `fit`'s
      docstring, the README and ARROW-SOURCES say the same now, and the
      keys only those two models take left the table.

<a id="task-140"></a>
- [x] 140. **Task 87's inverse diagonal, on every solve** (the user, after
      task 138 found the README's throughput 40–50% stale: "find which
      change caused the throughput drop"). S–M; *the user's call*. The
      bisect is PERFORMANCE §29: task 87 took `ewridge` 26% at k=20
      (`11e3ccb` 3.65M rows a second, `3e0b359` 2.69M), the clock checks
      (`7965035`), compensated means (task 101) and the floored cluster
      metric (task 102) the rest, each the price of a correction. Task 87's
      is not: `EwRidge::solve` forms `A⁻¹`'s diagonal on every solve, for
      each coefficient's data share and the noise gate's `edf`, and at HEAD
      that costs 7% at k=5, 21% at k=20 and 46% at k=50 (a build with the
      diagonal replaced by zeros). A cheaper direct method recovers
      nothing: the squared column norms of `L⁻¹`, written two ways, ran no
      faster than faer's inverse. Options:
      (a) **Compute it when it is read**, from the factor the solve keeps:
      on `coef` rows (by default one per group per chunk), in `summary()`
      and the `ReadinessWarning`, and for the noise gate only while it can
      bind. `edf` is at most one per kept column plus one, so once
      `sqrt(1 + (1 + k) / n_kish)` is below `max_error_inflation` the gate
      is open whatever `edf` is. Every value read is the same bits as
      today; the memory is a `k × k` factor per system, which
      `emit_error_inflation` keeps already. Recommended.
      (b) Every m-th solve: values stale between, and a new setting.
      (c) Keep it, and let the README's throughput say so, as it now does.

      **Built 2026-09-29, option (a), on the user's "Build the fix".** A
      solve no longer takes the diagonal. It keeps its factor in a
      `Pending`, shared by the slots the system answers, and the shares
      are computed on the first read and kept (`OnceLock`). A `coef` row
      reads them, and so do `summary()` and a save, which writes the shares
      a read would give, so the state layout is unchanged and needs no
      schema bump. The end of each run stores them and drops the factor,
      so a factor lives only while its stream runs. The noise gate reads
      `error_inflation_gate_into`. Where a solve is unread and the ratio of
      the bound, `sqrt(1 + (edf0 + k) / n_kish)`, is below
      `max_error_inflation`, the bound stands in, since the computed `edf`
      rounds to no more than it. Where the ratio could reach the limit, the
      gate reads the exact value, and so does the unreachable-gate
      warning's worst ratio. **The bound needed a guard these options
      missed.** A share can be NaN: with no ridge, an inverse that
      overflowed gives `1 − 0·∞`. The exact ratio is then infinite and the
      row is withheld, where the bound would have passed it. Features near
      `1e-155` fitted through the origin do it, and the test for it failed
      before the guard. `SpdFactor::inverse_is_finite` bounds every
      quantity the two triangular solves form, in `O(k²)` from the
      comparison matrix, and the bound stands in only where that holds. In
      its test it certified 2,181 factors and none of the 1,675 whose
      diagonal was not finite. **No value moves**: 18 `ewridge` workloads
      against `80df0ac`, every float of every frame, the state saved
      mid-stream and at the end, and every warning, the same to the bit.
      **Speed** (PERFORMANCE §29): +16% at k=20 and +41.5% at k=50, 82% and
      92% of the stub's ceiling; k=5's +3.5% is within noise; +42% where
      every row solves. `coef_every=1`, which reads the shares on every row,
      pays 1 to 3% for the bookkeeping. The README's `ewridge` rates are
      re-measured, with `rls`'s beside them and the `predict` ratio, and so
      are three solve-every-row figures stale since 0.9.0. Its Parallelism
      figures at k=20 were not: at load 4, a 14-thread run times the other
      jobs. Found on the way, not investigated: where every row solves, even
      the stub runs 23% below §19's rate of 2026-09-08.

<a id="task-141"></a>
- [x] 141. **Solve-every-row speed** (the user, after task 140 found §29's
      stub 23% below §19's rate where every row solves: "Solve-every-row
      speed"). S–M. PERFORMANCE §30 has the bisect: at k=20, 0.5.1 ran
      546k rows a second, the 2026-09-12 review's library batch took
      10.6% (N1's centred solve and the rest, spread over its commits),
      task 87 took 40%, and after task 140 it was 412k. A profile put 90%
      of a row in the solve: its own copy of the centred system and `k²`
      divisions by scales of 1 (23%), faer's per-call matrices for one
      right-hand side (13%), and the factorization's input copy (8%).
      **Built 2026-09-29:** unstandardized, the solve reads the Gram
      straight, with no divisions by 1; standardized, it reads only `C`'s
      diagonal for its scales; `SpdFactor` solves in place, as faer's own
      `solve` does after its copy; `ln det` is taken when read. No bit
      moves: 20 workloads against `80df0ac`, `coef_prior` among them. Every
      row solving, +8% at k=5, +13.5% at k=20 and +19% at k=50, with k=20
      back at 0.8.0's rate; at the default cadence, +3 to +10%. Left: about
      twenty allocations a solve.

<a id="task-142"></a>
- [x] 142. **NumPy's newest and next versions, tested as Polars's are** (the
      user, 2026-09-30: "Since numpy is an optional dependency we should
      test on the next release candidate of that too as we do with
      polars"). S. NumPy is the one optional extra (`numpy>=1.24`, no
      ceiling), and every job ran on the locked NumPy: 2.5.2, where PyPI has
      2.5.3. **Built:** `release.yml` gains `next-numpy`, two legs like
      `next-polars`: the newest NumPy blocks the publish, and its next
      release candidate (`--prerelease=allow`) is advisory. Only NumPy is
      upgraded, so a red leg names it. `polars-canary.yml` gains the same
      advisory run weekly. On 2026-09-30 NumPy's candidates on PyPI,
      2.4.0rc1 and 2.5.0rc1, are older than 2.5.3, so both legs resolve to
      2.5.3 until the first 2.6 or 3.0 candidate. `tests/test_release_workflow.py`
      holds the legs, and `test_ci_cost_policy.py` now names the prep
      action's callers. A workflow change, so the next release rehearses
      first, with publish off. Left as it is: the extra has no ceiling, so a
      NumPy 3.0 that breaks this library blocks every release until it is
      fixed or capped, the trade the Polars range makes too.

<a id="task-143"></a>
- [x] 143. **Window expressions: factors and targets as Polars formulas
      over exponentially weighted operators -- the column form built
      2026-10-03 ("Built" at the end); the target form is task 104's.**
      Size M to L. Depends on 78 (done); the target form on 104. Replaces
      `po.window.ewm` and `po.window.lookahead_rewm` (pre-1.0, no aliases).
      Constraints: the speed of task 78's core, and no memory beyond what
      each operator needs.

      #### The design

      Everything stateful is a kernel in the core -- direction, clock,
      `half_life`, `window_size` -- applied to any number of inputs.
      Everything row-local is a Polars expression, evaluated by Polars a
      chunk at a time. Each operator returns a genuine `pl.Expr`: a column
      reference whose name encodes the operator (`@po:{...}`, its JSON), so
      it composes with `pl.col`, `pl.lit`, numbers, `when/then` and every
      element-wise Polars function, on either side of an operator, and
      `pl.col` means the current row.

      ```python
      dpv, dv = po.increment("cum_notional"), po.increment("cum_volume")
      notional = pl.col("price") * pl.col("quantity")
      lf.online.with_windows(
          mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
          fwd_mid=po.rewm_mean("mid", half_life="10s", window_size="1m") - pl.col("mid"),
          fwd_vwap=po.rewm_sum(notional, half_life="10s", window_size="1m")
                   / po.rewm_sum("quantity", half_life="10s", window_size="1m") - pl.col("mid"),
          buy_vwap=po.ewm_sum(pl.when(pl.col("side") == "buy").then(notional), half_life="10s")
                   / po.ewm_sum(pl.when(pl.col("side") == "buy").then("quantity"), half_life="10s"),
          vwap_from_sums=po.ewm_sum(dpv, half_life="10s") / po.ewm_sum(dv, half_life="10s"),
          past_cvwap=(pl.col("cum_notional") - po.ewm_mean("cum_notional", half_life="10s", window_size="1m"))
                     / (pl.col("cum_volume") - po.ewm_mean("cum_volume", half_life="10s", window_size="1m")),
          volume_rate=po.ewm_rate(dv, half_life="30s", window_size="5m"),
          clock="ts", max_dclock="5m", session="date", session_gap="5m", group="symbol",
      )
      ```

      #### The operators

      Each is defined once, by the library that has it, and held to it:

    | operator | definition | oracle |
    |---|---|---|
    | `po.ewm_mean` (back), `po.rewm_mean` (forward): time-weighted mean | `y_i = α_i x_i + (1 − α_i) y_{i−1}`, `α_i = 1 − λ^(Δt_i)`: each value weighted by the decayed time of its interval; forward, the mirror, each value held until the next row | Polars `ewm_mean_by` (agrees to 1.3e-14 on irregular times) |
    | `po.ewm_sum`, `po.rewm_sum`: exponentially weighted sum | `y_i = x_i + λ_i y_{i−1}`, anchored at the row's own time | Polars `ewm_sum_by` |
    | `po.ewm_rate`, `po.rewm_rate`: rate per unit time | the sum over its decayed time mass, `∫ λ^age ds` over the time the window covers | the definition, integrated |
    | `po.increment(col)` | `x_i − x_prev` within the group and session; null on a session's first row; seconds on a temporal column | Polars `diff().over([group, session])`, in memory |

      - **A level is a time-weighted mean**, so a burst of rows does not
        outweigh a quiet stretch. No mean counts rows; pandas' `ewm(times=)`
        is that form, and is not offered.
      - **A weighted mean is a ratio of two sums**: a VWAP is decayed
        notional over decayed volume, the time mass cancelling. It equals
        `ewm_sum_by(p q) / ewm_sum_by(q)` to 2.8e-14, and is exactly
        unchanged by zero-volume rows or by a print split in two. A side's
        VWAP puts `when/then` inside both sums; there is no `weight=`.
      - **Recipes are compositions, not operators.** Each operator is held
        to its oracle; a factor or a target is a formula over them, and the
        design excludes no recipe. Which recipe a quantity should use is
        settled by measuring them against their definitions and each other,
        not by this plan. The ones known so far:
      - **A rate from running sums** can be `(C(t) − po.ewm_mean(C)) / (t −
        po.ewm_mean(t))`, the clock's own running value in the denominator:
        `Σ K dv / Σ K dt`, the same tail kernel on the flow and on time, so
        the two are attributed in the same steps and the attribution largely
        cancels. `t − po.ewm_mean(t)` is the window's mean age, near `half_life /
        ln 2` (measured, half-life 10 s: 14.18 s on even rows, 11.19 s on
        bursty ones, the spread of the order of the gaps). `po.ewm_mean(dt)` is
        no time mass -- under the time-weighted mean each interval counts by
        its own length, 7.1 s on bursty rows with a 0.5 s average gap -- and
        is never a rate's denominator.
      - **A rate can also be the rate operator**, the sum over its decayed
        time mass: the point kernel, as the ratio of sums is for a VWAP. It
        takes any input, running sums' increments and squared mid changes
        alike, so every rate can be written both ways.
      - **Running sums** give a VWAP two ways. `po.ewm_sum(dpv) /
        po.ewm_sum(dv)` is the per-trade VWAP, each trade weighted by its
        own decayed time; where the sums step exactly at trades it equals
        the VWAP from prices and quantities. `(C(t) − po.ewm_mean(C)) /
        (C_v(t) − po.ewm_mean(C_v))` weighs each trade by the decayed time mass
        on its far side within the window -- before it backward, after it
        forward -- a smooth taper to 0 at the window's edge; the two agree
        as the window grows against the half-life. Either way an increment
        carries the day total's rounding: about 2e-6 at a notional of 1e10;
        an integer volume's increments are exact.
      - **A volume clock is a clock column**: with the running volume as the
        clock, `half_life` and `window_size` are in shares, and weights
        decay with the volume traded since. **Today only ungrouped**
        (measured 2026-10-02): the core keeps a shared clock across groups
        for the silent-group cut, and a per-symbol running volume steps
        back at every interleaved row, so `group="sym"` on such a clock is
        refused at row 1. A clock has to say whether it is shared across
        groups (time) or each group's own (volume), and a group's own clock
        skips the shared advance and the silent list.
      - The windowed and forward forms, which no library has, are held to a
        brute-force loop from the same definition and to the time-reversal
        identity: a forward window is a backward one over the reversed
        stream, less the row itself.

      #### `with_windows`

      - **`lf.online.with_windows(*exprs, **named, clock=..., ...)`** reads
        like `with_columns`: names come from keywords and `alias`. It finds
        the operators with `meta.root_names()`, computes each distinct one
        once as a hidden column while streaming, evaluates the expressions
        with Polars on each chunk it emits, and drops the hidden columns.
      - **A formula is kept in a compact tree of this library's own**
        (decided 2026-10-02, the user: "Pl.col definitions can be
        serialized, see if there is a super compact form that can be stored
        in the spec"; then "bump up the polars floor if needed"). At spec
        build time Python reads the expression's own tree
        (`expr.meta.serialize(format="json")`, with the caller's Polars) and
        walks it into nested lists: `["-", ["rewm_mean", "mid",
        {"half_life": "10s", "window_size": "1m"}], ["col", "mid"]]`, about
        70 bytes for that target, plain JSON. The operators are nodes with
        their parameters, not placeholder columns, so the tree is
        self-contained; it goes into the spec dict, the saved state and
        TOML (`expr = [...]`) as it is. Seven node kinds are read and no
        other: `Column`, `Literal`, `BinaryExpr` (its named operators),
        `Function` (log, exp, abs, sqrt, pow, clip, fill_null, is_null,
        negate), `Ternary` (when/then/otherwise), `Cast` and `Alias`.
        Measured 2026-10-02: every one of them has the same shape under the
        floor Polars (1.34.0) and the pin (1.44.2), so the floor stays
        1.34.0; the serialized tree as a whole is unstable across versions,
        which is why only these nodes are read, at build time, under the
        caller's own Polars, and an unknown node is refused by name. Both
        sides rebuild the formula through Polars' public builders, which
        are stable where the tree is not: Python's `pl.col`, `pl.lit`,
        `pl.when` and the operators; Rust's `col()`, `lit()`, `when()`, the
        `Expr` methods and arithmetic, all in the embedded polars 0.55.2
        under the `lazy` feature already on, no new crate. A round-trip
        test covers every node kind in both directions, on the floor
        Polars too.
      - **Element-wise only, by node kind.** `shift`, `diff`, `cum_sum`, a
        rolling window, `over` or an aggregation would depend on the
        chunking (hard rule 3); each arrives as a `Function`, `Window` or
        `Agg` node outside the seven, so the build refuses it by name. No
        probe-frame evaluation.
      - **An operator's input** is a column or an element-wise expression of
        the row, `po.increment` included: the core computes increments as
        hidden columns from each group's previous row, then Polars evaluates
        the expression around them.
      - **Names, boundaries and equal timestamps follow Polars**: task 144.
      - **The call's own keywords are not output names** (decided
        2026-10-02, the user: "3a"). `with_windows(session=po.ewm_mean(...))`
        is refused, with a message pointing at `.alias()`: the clock policy
        (`clock`, `gap_cap`, `restart_after_step_back`, `session`,
        `session_gap`, `group`), `like`, `chunk_rows`, `load_state` and
        `save_state` are reserved. Polars' `with_columns` has no such clash,
        since it takes no policy. Considered and not taken: the policy only
        through `like=`.
      - **The clock and its policy** -- `max_dclock`, `on_clock_reset`,
        `min_backwards_jump`, sessions, groups -- are task 78's, shared by
        every operator in the call. **A clock per operator, Polars' `by=`,
        waits on a need** (decided 2026-10-02, the user: "Follow the reco";
        task 118 lists it), so that volume-clock and time-clock windows
        could share a call. Analysed: possible, since every
        queue item already carries its own policy time and nothing in the
        core assumes one clock but the clock fields themselves. The core
        keys each kernel by its clock; the clock state, the policy time, the
        restart flag and the silent list become one per clock in `Windows`
        and in each `Group`, a waiting row carries a policy time per clock,
        a break on one clock cuts the windows on that clock and a session
        change (one column) cuts every window, emission stays in input
        order since each clock's closures are monotone, and the window
        state file bumps its version. The policy goes per clock -- the gap
        cap, the step-back rule, `session_gap`, and whether the clock is
        shared across groups -- as `po.clock(...)` objects on the call, the
        operator naming its clock by `by=` and taking the call's single
        `clock` by default, and each window's `half_life` and `window_size`
        checked against its own clock's dtype. About two days on top of
        143, a second clock advance per row (unmeasured, expected under
        5%). Two things it does not give: a forward target off the spec's
        clock, since the embargo cannot be shown to cover a window in
        another unit (refused), and on a volume clock Polars' equal-stamp
        rule (task 144) holds every quote row's backward window until the
        next trade, since the quotes between two trades share one volume.
        Two calls compose for the column form, a formula over the first
        call's outputs in the second, at the cost of a second pass and a
        second state file. Not built until a need appears; then inside 143,
        where the kernel keying is being rewritten anyway, if 143 is still
        open.
      - **As a target** (task 104), the same expression in `targets=[...]`,
        carried as the tree above: the bank rebuilds it with the embedded
        Polars and evaluates it on the rows the window core resolves, so
        the column form and the target form share one evaluator, the
        command line takes a formula, a saved state resumes without being
        handed it again, and `fit_predict_arrow` takes one too.
      - **Against the current quote** is arithmetic: `- pl.col("mid")`,
        `/ pl.col("ask")`, `(op / pl.col("bid")).log()`; task 107's
        `relative_to` stays for plain-column targets.
      - **Open, the user's call: a ratio target's hit test.** A plain-column
        ratio target carries `relative="ratio"`, which centres `hit_rate`'s
        sign test at 1 (task 107's review, D3); an expression `op /
        pl.col("bid")` carries nothing, so its test centres at 0 and reads
        1.0 whatever the fit. Recommended: the docs say a ratio target is
        written as a log ratio or a difference; the alternative makes
        `hit_rate` null on every expression target.
      - **The operators take Polars' names** (decided 2026-10-02, the user:
        "4a"). Polars 1.44 has `ewm_mean`, `ewm_sum` and their `_by` forms, so
        the time-weighted mean is `po.ewm_mean` and the sum `po.ewm_sum`, and
        their forward mirrors `po.rewm_mean` and `po.rewm_sum`. The rate,
        which Polars lacks, is `po.ewm_rate` and `po.rewm_rate` on the same
        pattern, and `po.increment` keeps its name. Considered and not taken:
        `po.ewma` and `po.rewma`.

      #### Memory

      - **One queue per kernel**, however many operators share it: an item
        holds its clock and one value per operator.
      - **A row enters an operator's queue only when it carries a value**:
        a sum of a notional that is null on quotes queues the trades alone;
        a mid's time-weighted mean queues every quote, which it needs.
      - **`po.increment`** keeps one value per input per group.
      - **The column form** holds what task 78 holds: one window of input
        rows as the input's own chunks, and the queues.
      - **The target form** holds, per waiting row, only the row's own
        columns the target expression reads.

      #### Numerics

      A running sum less the window's mean of it cancels: a day's notional
      of 1e10 against a window's 1e6 costs about four digits, as does a
      price of 100 less a mid of 100 against a window's 0.01. Polars does
      that subtraction, so the core cannot help unless the operator is
      itself relative: an option `relative_to=` that returns `mean − x_t`,
      computed from sums centred on each segment's anchor row (`S' = Σ w
      λ^d (x − x_anchor)`, shifted by the anchor's step times the weight
      when segments combine, `mean − x_t = S'/W + (x_anchor − x_t)`). Built
      only if a brute-force comparison on day-scale running sums shows the
      plain form losing digits a factor reads.

      #### Tests and measurement

      - Each operator against its oracle above, on irregular times, and
        against a brute-force loop in both directions with a window; the
        time-reversal identity; an operator on either side of a Polars
        operator; identical operators computed once and operators sharing a
        kernel sharing a queue (by `queued()`).
      - Every recipe above against a brute-force loop of its own
        definition: both VWAPs and both rates. Then the recipes against each
        other, on synthetic streams and on the real quotes and trades: how
        far apart they are as the window grows against the half-life and as
        the row spacing turns bursty -- the evidence for which recipe a
        factor or a target should use.
      - The VWAP as a ratio of sums against the prices-and-quantities form;
        a session change restarting `po.increment`.
      - Each non-element-wise expression refused while the plan is built,
        and every element-wise one in the tests accepted.
      - The target form equal to the column form fed back through the
        embargo, row by row (task 104).
      - Real interleaved data: Binance USD-M futures `bookTicker` (best bid
        and ask) with `trades`, one symbol-day, downloaded and cached under
        `.cache/` by `tests/data.py` (hard rule 1).
      - `scripts/windows_bench.py` extended: 1, 4 and 16 operators on one
        kernel against as many windows, at 16M rows; memory at the
        scan-and-sink floor, as task 78 measured.

      #### Built (the column form, 2026-10-03)

      - **The core** (`windows.rs`, rewritten): a kernel per distinct
        (direction, `half_life`, `window_size`, `closed`), any number of
        operators on it sharing one two-stack queue whose items carry one
        value and one accumulator per operator; the queue keeps suffix
        sums on its front stack and prefix sums on its back, so a window
        less its oldest rows is read without a subtraction. That is what
        a held value cut at the window's edge needs: the first attempt
        subtracted the boundary row's full mass from the sum, and at a row
        exactly a window old the two cancelled to rounding dust that set
        the mean (a `"left"` mean read 6.27 where the definition gave
        6.34). The forward side closes a window before the kernel's open
        row -- the last one, held until the next -- joins, and counts that
        row directly, so no queued row is ever cut.
      - **Each operator's intervals are its own** (measured 2026-10-03:
        Polars' `ewm_mean_by` on `[1, null, 3]` at `t = [0, 1, 2]` gives
        2.5, the recursion with `Δt` from the last *valued* row; the row
        form gives 2.0). Looking back a value is held from the operator's
        last valued row (`Group::last_valued`, a start per operator in
        each item); looking ahead the last value is held until the next
        (`Group::held`, forward-filled), and a held value counts only
        where the row holding it is in the window -- a forward mean starts
        at the first in-window row with a value of its own, since the rows
        before it hold the row's own value or one its stamp excludes
        (found by the brute force and the time-reversal identity both).
        `min_samples` counts rows that carried a value.
      - **Window boundaries in one form everywhere**: ages compared with
        the window as differences (`tau - t > w`), in the core and the
        brute force, since the sum form `t + w` and the difference form
        disagree at an exact boundary by one ulp (a row exactly a window
        later fell on different sides).
      - **The frame runner** (`windows_frame.rs`): four passes a chunk --
        Polars evaluates the increments' inputs, the runner takes each
        increment one row back within the group's session (a step back
        past `restart_after_step_back` starts it over; nanoseconds for a
        temporal input, seconds out) as hidden columns; Polars evaluates
        every operator input to a number; the core runs; Polars evaluates
        the formulas over the held rows with the operators as hidden
        columns, which are then dropped. Increments live in the runner,
        not the core, because an operator's input may contain one and the
        inputs are evaluated before the row loop. The output schema is the
        formulas' dtypes over an empty frame. Windows state version 2.
      - **The formula tree** (`formula.rs`, `_formula.py`): the list form
        of the design, read on both sides, refused by name otherwise; the
        operators are nodes with their parameters; `Over` is what Polars
        calls an `over` node, `RollingExpr` a rolling function.
      - **The call**: `with_windows(lf, *exprs, **named, clock=..., ...)`,
        an expression named by its keyword or `.alias()` (an unnamed one
        whose name would be an operator's is refused), the clock keywords
        refused as expressions (3(a)), `like=` as before. `po.window`,
        splits, weights, totals, `unlisted`, `complete` and name templates
        are gone (a side's VWAP is `when/then` inside both sums); so is the
        `complete` column -- `partial="null"` shows a cut window as null.
      - **Tests**: Rust, every operator on nine kernels against a brute
        force from the definitions on five streams (gaps, repeated stamps,
        sessions, missing values, groups), chunking bit for bit, the
        time-reversal identity, the Polars `rolling_sum_by` membership
        table under each `closed`, the recursion, resets and cuts, the
        state at every row; Python (`tests/test_windows.py`, 58), each
        operator under each `closed` against a loop from the definition,
        the recursions written out, Polars' `ewm_mean_by`, `ewm_sum_by`
        (1.44.1 on) and `rolling_sum_by` as second opinions, the mirror,
        the three VWAPs as ratios of sums (unchanged by a print split in
        two or a zero-volume row), the VWAP from running sums' increments
        equal to the one from prices and quantities, compositions on
        either side of a Polars operator and their dtypes, the round trip
        of every node kind, every refusal by name, the clock events,
        `like=`, chunk invariance, resuming, slices and projections, the
        namespaces, a model fed in one query, durations, and one real
        Binance day (2.55M rows, 52% of trades sharing a millisecond with
        a quote) through four formulas.
      - **Measured** (`scripts/windows_bench.py`, 2026-10-03; PERFORMANCE
        §33): memory at the scan-and-sink floor at every size, sixteen
        operators on one kernel in 57% of sixteen kernels'
        time; against task 78's core rebuilt from `c061b7a` in a worktree
        and run beside it at one load, 1.5 times its time at 16M rows (2.44 and 2.49 s against 1.60 and 1.65 s, alternating), memory equal -- the constraint
        "the speed of task 78's core" is not met on the one case the old
        core could run, and the general engine is the reason. Three
        costs were profiled (an unstripped build under the sampler) and
        removed on the way: seven `Vec`s a row in the first queue (a
        struct-of-arrays queue with flat arenas), the nanosecond clock's
        `i128` division and conversion on every step (an `i64` fast path,
        bit-identical, library-wide), and a Polars plan for a bare-column
        input or a bare-operator formula (skipped; it measured nothing).
        The user's call whether the remaining gap is worth a further
        restructuring (the open-row bookkeeping and the two clock advances
        a row are the next items).

<a id="task-144"></a>
- [x] 144. **Window expressions take Polars' names and semantics -- the
      names built 2026-10-02 (the two tables below, "Built" at the end);
      the window semantics built 2026-10-03 inside task 143's core:
      `closed` (right, the default; left; both; none), `min_samples`, a row
      exactly `window_size` later counted under right and both, every row at
      one stamp sharing one backward window and waiting for the next
      distinct stamp under right and both, held to Polars' `rolling_sum_by`
      membership table and to the brute force in both languages (task
      143's "Built").** Where
      Polars has a parameter for the same thing, task 143's operators take
      its name, values, default and meaning, and the specs take
      `half_life`; the rest of the public names follow (below). Size L,
      with 143.

    | Polars (`rolling_mean_by`, `ewm_mean_by`) | the window operators |
    |---|---|
    | `half_life` | `half_life` |
    | `window_size` | `window_size` |
    | `closed="right"`, or `"left"`, `"both"`, `"none"` | the same: backward `(t − w, t]`, forward `(t, t + w]` |
    | `min_samples=1` | the same: fewer counted rows gives null |

      - **The forward window counts a row exactly `window_size` later.** As
        a target, the model learns row *t*'s label when the first row at or
        past *t* + w arrives, counting that row when it lands on the
        boundary; its value is known then.
      - **Equal timestamps follow Polars**: a window is a set of timestamps.
        A backward window counts later rows at its own timestamp, so its
        output waits for the next distinct one; under `"right"` a forward
        window counts none at the row's own timestamp. Stream order at one
        timestamp is an opt-in (`same_clock="include"`), for data whose order
        within a timestamp is known; in Binance's files the order of a trade
        and a quote update in the same millisecond is not.
      - **Where Polars has no counterpart, task 78's rules hold**: a gap
        past `max_dclock` or a session change cuts a window and a reset
        discards it, under `partial`; a forward row whose window has not
        closed when the input ends is null, where Polars would compute a
        partial window.
      - **`halflife` becomes `half_life` library-wide**, with no alias, in
        every parameter that spells it: `half_life`, `long_half_life`,
        `coef_half_life`, `revert_half_life`, `select_half_life`,
        `level_half_life` and `trend_half_life`, on every spec builder, in the
        Rust `Spec`, the command line's TOML (`examples/bank.toml`), error
        messages, docstrings and docs (prose says half-life). The output
        fields' `@h` instance label is unchanged. A saved state stores its
        specs, so a state saved before the rename does not load: raise the
        bank's `MIN_SCHEMA_VERSION` (pre-1.0 state compatibility is waived).
        Ships in the release that renames `label_delay` to `embargo` (task
        104), so specs break once.

      #### The rest of the public names (decided 2026-10-02, the user: "Add all")

      Each name was checked against Polars, against the rest of the API,
      and against what it does. All ship in the same release as
      `half_life` and `embargo`, with no aliases.

    | now | becomes | why |
    |---|---|---|
    | `max_dclock` | `gap_cap` | "dclock" is internal shorthand. It caps how much clock one step counts, and a capped step is a break; "gap" is the library's word, as in `session_gap`. Not `max_gap`, which reads as a limit on the gaps allowed. |
    | `window` (specs) | `window_size` | Polars' name, and the window operators'. `window_every`, `window_budget` and `window_lags` keep theirs. |
    | `n_eff` | `weight_sum` | It is a weight, not a sample size (task 148); in statistics n_eff is Kish's effective sample size, which this library calls `n_kish`. In every output field (`weight_sum@h<h>`), the `closed_groups` columns (`pair_n_eff` becomes `pair_weight_sum`), the `gram()` keys and CLAUDE.md's rule 8. |
    | `min_periods` | `min_weight` | It compares that weight. Polars renamed its own `min_periods` to `min_samples`, which counts rows, so neither name says what ours does. |
    | `resid_z_<t>`, `emit_resid_z` | `zscore_<t>`, `emit_zscore` | `resid_` is a prefix of `resid_z_`, so a target named `z_y` made `resid_z_y` two fields (measured 2026-10-02: the builder accepted the spec and listed it twice, and `fit_predict` refused it). The one prefix collision in the output grammar. |
    | `pcorr_<a>_<b>` (`ew_cov`) | `partial_corr_<a>_<b>` | The stat is asked for as `"partial_corr"`. |
    | `scale_features` (`sgd`) | `standardize` | Every other model's name for it. |
    | `on_clock_reset` + `min_backwards_jump` | `restart_after_step_back` | One rule in two vocabularies, the second required by one value of the first and refused by the other. Unset, a step back is an error; given a clock amount, a step back larger than it restarts the state and one no larger is still refused (inclusive, as built; this row once said "at least that large", corrected in review R1). In the specs and in `with_windows`. A clock rule, so it gets the extra review task 120 calls for. |
    | `ridge_decay` (bool) | `ridge_scale="mean"` or `"sum"` | Names the difference: a penalty on the mean moments, permanent, against a prior on the sums, fading (task 151's two families). Default `"mean"`. |
    | `lam_selected_<t>` (`lasso`) | `penalty_selected_<t>` | `lam` is the decay factor. It keeps that name: Polars' alternative to `half_life` is `alpha`, which `ftrl`, `deco` and `corrchange` already use. |
    | `add_intercept` | `fit_intercept` | scikit-learn's name for the same switch. |
    | `max_cd_iters`, `cd_tol` (`lasso`) | `max_iter`, `tol` | scikit-learn's `Lasso`. |
    | `reset` (`corrchange`) | `reset_on_flag` | "Reset" already names a clock event. |

      - **Everywhere a name appears**: the spec builders, the Rust `Spec`,
        the command line's TOML (`examples/bank.toml`), `with_windows` and
        its frame methods, error messages, output fields, `closed_groups`
        columns, `gram()` keys, docstrings, the README, the reference pages,
        `llms.txt` and CLAUDE.md. The rest of this plan keeps the old names
        in entries written before this one; what they build takes the new.
      - **An old name is refused with a message naming the new one**
        (`max_dclock was renamed gap_cap`), in Python and in TOML: no alias,
        and no bare "unknown field". Outputs simply change (the user,
        2026-10-02: no backward compatibility there): nothing reads an old
        field name back, so there is nothing to refuse.
      - **The builder refuses two outputs that render to one field name**,
        as the bank already does at `fit_predict`; `output_fields` listed the
        duplicate. The message names the inputs that collided, features
        included: it said "rename a target or grid label" whatever the
        cause, and for `pc0_var` the cause was a feature.

      #### The outputs (decided 2026-10-02, the user: "Yes")

      Beyond the outputs in the table above (`weight_sum`, `zscore_`,
      `partial_corr_`, `penalty_selected_`):

    | now | becomes | why |
    |---|---|---|
    | `n_eff` in `closed_groups` (and `pair_n_eff`), `coef()`, `last_row()`, `marginal()`, `gram()` keys and `po.eval`'s sums | `weight_sum` (`pair_weight_sum`) | `summary()` already calls this number `weight_sum`, so it had two names. |
    | `withheld_reason` value `below_min_periods` | `below_min_weight` | Follows the parameter. |
    | `coef()` column `lambda` | `penalty` | Follows `penalty_selected_<t>`. |
    | PCA loadings `pc<j>_<feature>` | `pc<j>_loading_<feature>` | A feature named `var`, `share` or `score` collided with the component's own fields (measured 2026-10-02: `pc0_var` twice, refused at `fit_predict`). |
    | `bocpd`'s `logscore` | `loglik` | `hmm` and `deco` call the same quantity, the row's log predictive density, `loglik`. |
    | `hmm`'s `p_<k>`, `p1_<k>` | `filtered_<k>`, `predicted_<k>` | `p1` is cryptic; the docs call them the filtered and the predicted state. |
    | `absresid_q<p>_<slot>` | `abs_resid_q<p>_<slot>` | The one fused word in the grammar; matches `mahal_q<p>`. |
    | `micro`'s field `micro` | `micro_id` | The id of the summary the row goes to, not its model's name. |

      Kept, checked: `pred_`, `resid_`, `sigma_`, `lo_`, `hi_`, `coverage_`,
      `ic_`, `r2_`, `hit_rate_`, `autocorr_`, `drift_`, `selected_`,
      `settled_frac`, `support_coef`, `error_inflation_`, the slot labels
      (`__<set>`, `_r`, `__l`, `@h`), `describe()`'s columns (Polars' own
      `describe` names), `closed_groups`' counters and clock range, and the
      standard names of `rcov` (`omega2`, `iq`) and `seqtest` (`log_e_`).
      - Kept, deliberately: `clock`, `session`, `session_gap`, `group`,
        `group_close`, `weight`, `coef_every`, the `emit_*` family, the `@h`
        labels, the models' standard notation (`q`, `p0`, `obs_var`; FTRL's
        and DCC's `alpha` and `beta`; DenStream's `beta_mu`; PA's `c`), and
        the model names (`ew_cov`, `ew_class` against Polars' `ewm_`: a
        model rename breaks more than any parameter).
      - Specs are stored in saved states, so the bank's
        `MIN_SCHEMA_VERSION` rises once for all of it.

      #### Built (the names, 2026-10-02)

      Every row of both tables, plus: `po.eval.rolling_metrics` takes
      `window_size` too (the same concept, Polars' name); `ridge_scale` is
      a two-value enum (`"mean"` default, `"sum"`), where `ridge_decay` was
      a bool. Schema 22: the bank refuses a file older than 22 by version
      (`MIN_BANK_SCHEMA_VERSION`); the models' own states did not change
      and the core minimum stays 14. The Rust accessor keeps the name
      `n_eff` (CLAUDE.md rule 8 says so): only the emitted field is
      `weight_sum`. An old name is refused naming the new one from the spec
      builders (`_RENAMED` in `_spec.py`) and, through
      `online_polars::name_renamed`, from a spec dict, a windows config and
      the command line's TOML, at the spec's level and the model's;
      `with_windows`' own Python keywords refuse an old name as any Python
      function does, by `TypeError`, since its signature is explicit. The
      builder refuses two outputs rendered to one field name, naming the
      inputs (`duplicate_field`): the one collision the columns' own names
      can still make is `ew_cov`'s `corr_a_b_c` over `a_b, c` and over `a,
      b_c`. The rename ran as a script over the live tree (code as
      identifiers; prose outside code spans says half-life), with history
      files and this plan's earlier entries left as written.
      `docs/OUTPUTS.md` and `tests/api_surface.txt` are regenerated. Tests:
      `tests/test_renames.py` (the builders' table against this one, every
      old name refused from a builder and from a dict, no old output name
      in the release probe's workload, the new names, the old collisions
      gone, the duplicate refusal naming the inputs, the same number under
      one name in every frame, `restart_after_step_back` as one rule), the
      Rust `an_old_name_is_refused_naming_the_new_one` (JSON and TOML),
      `spec_defaults.rs` for the one clock rule, and every existing test
      under the new names. Cost: none -- no hot path changed. Measured
      2026-10-02 against the task 152 build (`ewridge`, k=20, 400k rows,
      load 4.8): 3.01M rows/s against 3.01M, 3.02M with `emit_clocks`
      against 3.02M, 2.01M under `embargo=10` against 2.00M; state 10,029
      bytes, unchanged; peak RSS 531 MB (the first RSS figure recorded, the
      baseline for the next task).

      Tests: each `closed` value in both directions against Polars'
      `rolling` on the same rows, equal timestamps included; `min_samples`
      against Polars'; a backward window's output waiting for the next
      distinct timestamp, chunked anywhere. For the names: the API surface
      snapshot's diff is exactly this table; every old name refused naming
      the new one, from Python and from TOML; no old name left in the live
      docs (a search, never truncated); a target named `z_y` beside `y`
      accepted with `emit_zscore`; two outputs with one name refused by the
      builder, naming the inputs; a feature named `var`, `share` or `score`
      accepted beside `pca`; the same number under one name in every frame
      (`weight_sum` in the fields, `summary()`, `closed_groups`, `coef()`,
      `last_row()`, `marginal()`, `gram()` and `po.eval`).

<a id="task-145"></a>
- [x] 145. **`session_shrink` mixes data, not weight -- built
      2026-10-02.** Size S. `ewridge`'s blend toward its slow twin at a
      session change gave the twin the share `f W_slow / ((1 − f) W_fast + f
      W_slow)` and the weight `(1 − f) W_fast + f W_slow`. The twin's weight
      is many times the fast sums', so every `f` above 0 reverted almost
      fully: measured on 0.13.0 with a 100-row half-life against a
      10,000-row twin, the slope moved from 2.008 to 1.113 at `f = 0.25`
      (1.083 at `f = 1`), and `n_eff` from 145 to 2,855.

      - **The blend mixes the moments as a mixture of the two data sets**,
        in fixed shares `1 − f` of today's and `f` of the long run's: `m' =
        (1 − f) m_fast + f m_slow` for the feature and target means, and
        `C' = (1 − f) C_fast + f C_slow + f (1 − f)(m_fast − m_slow)(m_fast
        − m_slow)'` for the centred moments, the cross-moments alike. The
        fit is the ridge regression on that mixture.
      - **Every weight-like quantity is kept**: `W' = W_fast`, each
        target's own weight, the Kish sum of squared weights, the
        `ridge_decay` prior scale and the weight learned since the last
        solve, so `n_eff`, the Kish size, the warm-up gates and the solve
        cadence are unchanged across the blend and the fast decay carries on
        from the open. Today the inflated weight also delays the next solve:
        measured, the first re-solve after the open came 32 rows in at `f =
        0.25`, against 3 at `f = 0`, so the session's first rows are scored
        with stale coefficients.
      - So `f = 0` keeps today's fit, `f = 1` gives the twin's moments at
        today's weight -- the twin's fit exactly, unless `ridge_decay`, whose
        prior keeps today's scale -- and `f` between fits on that share of
        the long run. The coefficients move by about that share where the
        feature distributions match; they are not linear in the moments.
      - **In `Grams::blend` and `Acc::blend`**
        (`crates/online-core/src/gaps.rs`) and the moment blends they call:
        the shares `(1 − f, f)` in place of the weight shares, and the fast
        weights kept; `TargetMoments::blend` leaves `Q` as it is. The state
        layout is unchanged, so no schema bump. Numbers move for any `f`
        above 0: a CHANGELOG line, and the docstrings state the formulas.

      Tests: the blended coefficients from the raw rows, the two kernels
      each normalised and mixed, solved by `faer` independent of
      `solve.rs`, at `f` of 0.25, 0.5 and 1 (`f = 1` is the twin's fit), the
      weight bit-equal across the blend; under `ridge_decay` the prior scale
      bit-equal and `f = 1` the twin's moments at today's weight and prior,
      from the rows; the mixture term by term at the origin and at 1e8, each
      target's own weight kept; `f = 0` a no-op. In Python: the slope at the
      open within 0.05 of `(1 − f)` today's plus `f` the long run's, `n_eff`,
      the Kish size and the first re-solve equal with and without the
      blend, and `TestSessionShrinkBlend`'s numpy oracle at the normalised
      kernels. `tests/reference_paths.py` keeps a weight per target beside
      the Gram's, since each normalises over its own rows; seeded with the
      old blend its oracle misses by 3.5e-2 to 8.4e-2, with the Gram's
      normalisation for the targets by 5.0e-4 to 9.1e-4.

<a id="task-146"></a>
- [x] 146. **A clock parameter acts on the clock -- built 2026-10-02,
      but for `kalman`'s `coef_halflife`, which is task 150.** Size M.
      Every parameter in clock units acts on the policy clock, so its
      effect does not change when rows arrive more or less densely at the
      same clock; anything that counts rows says so in its docs, and a lag
      counts rows only between gaps. Measured on 0.13.0 where marked; the
      rest found by reading the code, and measured as each is fixed.
      - **Drift detection** (measured): `emit_drift`'s Page-Hinkley test
        added one excess per scored row to an undecayed mean, so the same
        30-unit burst flagged at 4 rows a unit and not at 1. It now
        accumulates `d · (e − mean − δ)` against a mean decayed at the
        model's halflife, so `drift_threshold` is in σ times clock units; a
        row with no residual ages the mean. At rows one unit apart with no
        decay it is the classic test, so its unit tests run unchanged with
        `d = 1`, `λ = 1`. WARMUP-AND-CONVERGENCE §7.1's coefficient CUSUM
        is to be built the same way from the start.
      - **`resid_quantiles` and `mahal_quantiles`** (measured): the P²
        estimator never forgot; at halflife 10 rows, 3,000 rows after the
        noise fell tenfold, `absresid_q0.9` read 1.55 where the recent
        quantile is 0.166. Replaced by `EwQuantile`
        (`crates/online-core/src/stats.rs`): DDSketch (Masson, Rim & Lee
        2019) with its bucket weights decayed on the clock, each value at
        its row's weight. A bucket is `64 e + ⌊64 m⌋` of `v = m 2^e`, exact
        on the float's bits, so no libm result enters the state; the
        representative is within `tanh(1/128)` (0.78%) of every value in
        its bucket, so of the exponentially weighted quantile. The decay is
        one factor folded into the buckets every 64 halvings, when an end
        bucket under `1e-12` of the total is dropped; each level keeps its
        bucket and the weight below it, so a row costs a few comparisons.
        One sketch per slot answers every level, from the first residual.
      - **`sgd` under `schedule="constant"`** (measured): the coefficients
        never decay, and `halflife` reaches only `n_eff`, the scaler and
        the AdaGrad sum, so halflife 10 and 10,000 gave identical slopes.
        Its memory is in rows, about `1 / (lr · E[z²])`, and the docs say
        so. `halflife` stays required: `n_eff`, and with it
        `min_periods`, always reads it.
      - **`deco`'s linear dynamics**: `alpha` and `beta` step once per row,
        so a capped weekend moves `ρ` as much as a millisecond; the docs
        say "on the model's own clock". They are per row, as a DCC model's
        are, and the docs say so.
      - **`hmm` transitions** apply once per row, a weekend one step, and
        the learned counts decay on the clock but grow per row, so the
        staying probability rises with row density. The docs say a
        transition is per row and what density does to it.
      - **`resid_autocorr_lag`** paired scored residuals, not rows, and was
        not cleared at a capped gap or a session change, so Friday's last
        residual met Monday's first. Its buffer is now cleared where the
        model's lag rings are, the docs say "scored residuals back", and
        the cross moment keeps a weight of its own, the pairs', so a value
        with no partner -- the first `lag` of a stream or of a run -- adds
        nothing to it rather than a zero product. Found while fixing it: a
        row with no residual, or of weight 0, skipped the tracker and its
        decay with it; it now ages it, and the drift detector's mean and
        the quantile sketch likewise.
      - **`conformal_rate`** steps the radius once per scored row, beside a
        coverage that decays on the clock; the docs say the step is per row.
      - **State schema 21.** The bank file keeps the stream's diagnostics,
        so it refuses one older than 21 (`MIN_BANK_SCHEMA_VERSION`). Of the
        models' own states only `ew_cov` with `mahal_quantiles` changed; it
        refuses one older than 21 by name, and the core minimum stays 14
        (the user, 2026-09-28).

      Tests: a burst found at the same clock at one and four rows a unit,
      in Rust and from Python; the detector against its recursion written
      out on irregular steps; the sketch within `tanh(1/128)` of the
      exponentially weighted quantile's definition row by row, on
      irregular steps and weights with zeros and a tenfold fall in scale,
      through rescaling and pruning, at the extreme floats, and saved and
      resumed; `absresid_q<p>` and `mahal_q<p>` against the same definition
      from the emitted residuals and distances; an autocorrelation null
      throughout when every row follows a break; a step with no value
      ageing the detector and the tracker as one step of the product's
      decay; an old `ew_cov` state with P² markers, written in msgpack,
      refused by name.

<a id="task-150"></a>
- [x] 150. **`kalman`'s `coef_half_life` on the clock -- built 2026-10-03,
      the recommended option.** Size S. Split from task 146. The process
      noise `q_i = σ² (ln 2 / h)²` was added per clock unit, but a Kalman
      gain grows with the square root of the noise, so the coefficients'
      memory was about `h · sqrt(d)` for rows `d` apart: adapting to a
      slope step took 74, 38 and 13 clock units at rows 1, 0.25 and 0.04
      apart with `coef_half_life=50`, where `ewridge` at half-life 50 took
      44 to 59 (measured on 0.13.0). The documented promise, EW-RLS's
      steady-state gain, holds at any spacing when the noise added per row
      is `σ² (ln 2 · d / h)²`: EW-RLS forgets `2^(-d/h)` over the step, a
      per-row half-life of `h/d` rows, and the matching is done at that
      half-life. **Built**: `P += Q · d²` per row (`kalman.rs`, the numpy
      reference, the README's equations, the builder's docstring); an
      explicit `q` is added as `q_i d²` too, and a reverting slot's
      stationary variance is `q_i d² / (1 − φ²)` at rows `d` apart. The
      match is first order in `d/h` (the exact per-row noise is `σ² g² /
      (1 − g)`, `g = 1 − 2^(−d/h)`: within 1% for rows closer than half a
      half-life, 4% at one). Bit-identical at unit spacing (the goldens'
      rows: the two values there that moved by 1e-15 are the standardizer's
      drift since the pins of `93ef404`, inside the 1e-12 check, not this
      task's), and the golden signatures move at the
      stream's gap row, re-pinned against the numpy reference -- the core's
      (`golden.rs`), the pipeline's (`tests/test_golden_pipeline.py`:
      `kalman`, `kalman_revert`, and the `seqtest` that compares the
      Kalman with the ridge) and the `filterpy` second opinion, whose
      `predict` now takes `Q d²` too. Tests:
      `crates/online-core/tests/kalman_clock.rs` -- the clock time to learn
      a slope step at rows 1, 0.25 and 0.04 apart (165, 176 and 170 clock
      units at `coef_half_life` 50, within 7% of each other, against
      `ewridge`'s 120, 113 and 116 at half-life 50; another stream and
      criterion than the 74/38/13 above, which are not comparable to these
      -- at unit spacing the two forms are identical) where the old form
      parted by a factor of five; the Kalman's 1.4 over `ewridge` is `R =
      σ²` inflating at the step before `P` catches up through `q ∝ σ²`,
      spacing-free; and a gap row adds exactly `q d²` to `P`, a unit row
      `q`. Not taken: keeping the random walk and documenting the
      `sqrt(d)`.

<a id="task-147"></a>
- [x] 147. **A row weight scales evidence, not counts -- built 2026-10-02,
      but for `ftrl` and the density half of `micro`, which are task 151.**
      Size M. Multiplying every row's weight by one constant changes no
      output -- the EW models already hold to it, since their sums are means
      -- unless the docs say otherwise for that model. Where a weight entered
      a count, it now enters as `w / w̄`, `w̄` the EW mean weight of the rows
      the quantity takes (this row's included, so the first row reads 1, and
      every row of a constant weight reads 1, which leaves those streams bit
      for bit as they were).

      - **`conformal_rate`** (measured): the step was `rate · σ · w · (miss
        − α)`, so the same rows at weight 100 swung the band: width sd 3.10
        against 0.21 at weight 1, mean 6.0 against 3.3. It is `rate · σ ·
        (w / w̄)`, `w̄` over the scored rows at the model's decay.
      - **`bocpd`** added the row weight to the posterior's counts (`κ`,
        `ν`) but not to the likelihood, so a heavy row made the predictive
        confident without counting as evidence. The weight enters both, as
        `w / w̄` over the rows learned from (nothing in `bocpd` decays):
        the run's statistics, and the message `πᵣ^{w/w̄}`, under `robust`
        times its β-power weight. `κ₀` and `ν₀` are in rows of the mean
        weight. What the row reports is read as a row of the mean weight,
        which is what `predict`, never told a weight, can say:
        `tests/model_contract.rs` held the first version, which reported
        at `w/w̄`, to `predict == step` and refused it.
      - **`quantile`**: the warm-up gate and the band floor compared the
        raw decayed weight with a row count, so small weights never left
        warm-up, and -- found by the property test -- rows at weight 100
        left it on their first row, where the fit, leaning on a one-row
        Gram, reached 1e51. Each target keeps its present rows counted one
        each (`nobs`), and the gate, the floor and the band's weight are
        read in rows.
      - **`micro`**: `beta_mu` and the pruning threshold `ξ` were absolute
        weights, and a summary admitted a unit row. All three take `w̄`, the
        EW mean weight of the rows learned from, so a heavier stream keeps
        no more summaries. The docs' `ξ` is corrected to what the code
        computes: the weight of a summary that took one row of the mean
        weight every `Tp` clock units since it opened, `Σ_{i ≤ a/Tp}
        2^(−i Tp/h)`. Its counts stay in rows, as DenStream's do, and the
        docs say a denser stream fills its summaries faster.
      - **`ftrl`**: built and reverted. Its weight is Vowpal Wabbit's
        importance weight, and `tests/test_second_opinion.py` holds the fit
        to VW's with weights, so penalties scaled by the mean weight
        parted from VW by design. The docs say a weight is an importance
        weight; the choice is task 151.

      Tests: `tests/test_weight_scale.py` fits every spec of the release
      probe's workload, every kind, at the stream's weights and at a
      hundred times them, `min_periods` scaled with them, and holds the
      specs whose outputs move to exactly the documented exceptions, each
      with its reason: `rls` (a sum-scale prior), `kalman` (a weight is an
      observation's precision), `sgd` (a step size), `ftrl` (an importance
      weight), `pa` (a weight above 1 counts as 1), `hmm` (counts against a
      Dirichlet pseudo-count) and `seqtest` (it reads `kalman`). In Rust,
      each fixed model at a hundred times the weights, and the conformal,
      `micro` and quantile oracles in Python with the same rule. The
      quantile test fails at row 2 when the gate reads raw weight.

<a id="task-151"></a>
- [x] 151. **`ftrl` and `micro` keep their units: a sum-scale prior and a
      point density -- decided and the docs written 2026-10-02.** Size S.
      Split from task 147. Nothing in the code changes.

      - **`ftrl`'s penalties stay in absolute weight.** FTRL-Proximal
        (McMahan et al. 2013) minimizes the cumulative loss plus a fixed
        regularizer, `g_{1:t}·w + λ₁‖w‖₁ + ½λ₂‖w‖²`: the penalty is a prior
        of fixed mass that the evidence outweighs, which is what its regret
        bound rests on, and an importance weight adds evidence, as Vowpal
        Wabbit's does. Under a halflife the evidence is bounded at `W∞`,
        so the effective regularization `λ₁/W∞` depends on the weights'
        scale and the rows' density: the sum-scale prior family, beside
        `rls`'s ridge, `ridge_decay` and `kalman`'s observation precision,
        which the library documents as the exceptions. The mean-scale
        sparse fit, invariant to both, is `lasso`. Penalties times the
        taught rows' mean weight, built on 2026-10-02, were `lasso`'s
        semantics under `ftrl`'s name and parted from VW's weighted fit;
        reverted. Docs: `ftrl` says its `l1`, `l2` and `beta` are a prior
        of fixed mass against evidence that grows with weight and
        density, and points to `lasso` for a penalty on the mean scale.
      - **`micro`'s thresholds stay in points.** DenStream (Cao et al.
        2006) is DBSCAN on a stream: a cluster is a region with at least
        `MinPts` points, the count is the definition of density, and the
        paper takes the arrival rate `v` as the stream's given -- the
        total weight is `W = v/(1 − 2^{−λ})`, and `ξ` is the weight of a
        cluster fed one point per time unit. A threshold as a share of the
        running weight would be density-free and a different algorithm: a
        cluster holding 5% of a stream that splits into twenty falls below
        a share while still dense. Task 147's `w/w̄` keeps points as the
        unit. Docs: with halflife `h` and `v` rows per clock unit the
        stream's steady-state weight is about `1.44·v·h`, so a summary
        meant to hold a share `s` of it needs `beta_mu ≈ 1.44·s·v·h`.
      - **A clock-rate normaliser is not built.** `max_dclock` bounds its
        dip at a capped gap to about `G/(1.44h)` and never removes it;
        skipping the rate's update on a `capped` or `session_changed` row
        would, with no new plumbing. Neither is wanted given the two
        points above.

      Tests: `tests/test_weight_scale.py` keeps `ftrl` among the named
      exceptions; a docstring test holds `ftrl`'s and `micro`'s docstrings
      to the two sentences above.

<a id="task-152"></a>
- [x] 152. **The clock a row was scored at, and the clock of the last row
      learned -- built 2026-10-02** (the user, 2026-10-02: "see the clock
      of the row that was scored and the clock of the last row learned at
      each scoring so we can see the embargo in action"). Size S–M. Works
      with today's `label_delay`; task 104's tests read it.

      - **`emit_clocks=True`** on any spec adds two fields to every scored
        row: `scored_clock`, the row's own clock, and `learned_clock`, the
        clock of the newest row the models had learned from when the row was
        scored, before the row's own update. Without a delay that is the
        previous row learned; under `label_delay` (task 104's `embargo`) it
        is the newest row whose delay had elapsed.
      - **The clock column's own values and type**: a numeric clock as a
        float, a temporal one as a `Datetime` in its own unit and zone,
        exact in integer nanoseconds (never an epoch read into a double, the
        task 88 rule); with no clock column, the row's index in its group,
        the unit the bank counts, as an integer. The first temporal field in
        the output struct.
      - **Learned** means released into the models at a positive weight:
        a zero-weight row advances the clock and teaches nothing, so it is
        never the learned row, and the doubled stream's scoring copies
        (`po.stream.embargo`) never show as learned. A row with a null target
        still entered the state (`n_eff`), so it counts.
      - **Null** on a row the spec skips; `learned_clock` is null before the
        stream's first learned row, and after a reset until the next. One
        pair per spec, no `@h` suffix: every instance of a grid learns the
        same rows at the same time (a drift reset restarts a model, not the
        release).
      - **The delay counts elapsed time on the clock column** (task 153),
        so on every row `scored_clock` less `learned_clock` is at least the
        delay, and the next held row is less than one delay back. Where a
        session change restarts the clock, the two clocks are in different
        sessions' units and the session's gap counts between them; the docs
        say so.
      - **`predict`** scores against the saved state, so `learned_clock` is
        the fit's last learned row on every row it scores.
      - **The state** keeps the last learned row's clock, and each held row
        its own, as a `ClockValue` (`#[serde(default)]`; schema 21 is not
        released yet, so no bump). Chunk-invariant like every field.

      **Measured 2026-10-02** (`ewridge`, k=20, 400k rows, machine load
      5–6): 3.02M rows/s with `emit_clocks` against 3.01M without, +28
      bytes of state; with `label_delay=10` 2.00M against 2.06M, and 0.12.0's
      1.92M on the same rows, so task 153's elapsed-time counting cost
      nothing either. The clock type is recorded only once a chunk is
      taken, so a refused chunk changes nothing (`chunk_plan.rs` held it).

      Tests: without a delay, `learned_clock` is the previous accepted row
      of positive weight; under `label_delay` the newest row whose delay
      had passed in elapsed time, row by row from the frame; a
      `Datetime` clock exact to the nanosecond in each unit; a reset and a
      group's first row null; a zero-weight row never shown as learned;
      chunk invariance, save and load, `predict`.

<a id="task-153"></a>
- [x] 153. **`label_delay` learned labels early at a break -- built
      2026-10-02** (the user: "Why not simply consider the max dclock or
      session gap a time delta and deliver rows as defined by it?", then
      "Yes" to fixing it now). A break -- a gap past `max_dclock`, or a
      session change -- released every held row at once, because its events
      (the lag rings' clear, `corrchange`'s span, `session_shrink`'s blend)
      ran when the break's row arrived, and a held row learned after them
      would have been paired across the break or blended (review L2). So a
      forward label was learned before it was known wherever a break was
      shorter than the delay: measured with a 10-unit delay and a 5-unit
      cap, an 8-unit gap learned two labels early, and a session change with
      no gap learned nine. Capping the delay at `max_dclock`, as task 104
      proposed, closed the gaps and not the sessions.

      - **The delay counts elapsed time**, `ClockAdvance::elapsed`: the
        clock column's own steps, uncapped, skipped rows' time included, and
        `session_gap` where a session change restarts the clock (a clock may
        restart at a session; `max_dclock` and `session_gap` say how much a
        model forgets across a break, not how long it lasted -- counted as
        time, a session gap longer than its real step released rows early).
      - **A break's events wait in the buffer with the row after it** -- or,
        raised on skipped rows, with the next accepted row, carried across a
        chunk boundary in the stream's state (`HeldBreak`) -- and run when
        that row is learned, after every row before the break. The models
        run one delay behind, events included. A reset still acts on arrival
        and drops the buffer.
      - The data summary counts a break where it arrived. A spec without a
        delay is unchanged, bit for bit; where a break is longer than the
        delay, as overnight, the release is where it was. New fields are
        `#[serde(default)]` within the unreleased schema 21.

      Tests: at every row, the rows learned are the rows whose delay has
      passed in elapsed time, counted from the frame, across a capped gap
      shorter than the delay, a session change with no gap and a clock that
      restarts at a session; the delayed bank ends as a plain bank fed the
      matured rows, with breaks inside the last delay; `ew_cov`'s lagged
      co-moments are the plain run's over the matured rows with breaks
      shorter than the delay (L2 kept); a break on a skipped row survives any
      chunking, the carry dropped is caught at chunk sizes 1, 3 and 81;
      `ClockAdvance::elapsed` against hand-worked steps. The four count
      tests failed on the old build.

<a id="task-148"></a>
- [x] 148. **The docs say what the code does: the 2026-10-01 findings --
      built 2026-10-02.** Size S. Each was a wrong or missing sentence,
      corrected to the code, with the test that pins it.

      - **`ridge_decay` penalizes the intercept** (measured: `y = 5` read an
        intercept of 3.50 at row 20, and 4.95 at row 199, against 5.0
        without it, at `ridge=10`, halflife 50), where `ridge`'s docs said
        "never on the intercept"; and `coef_prior`'s intercept slot is read
        under it, and only there. The docs say so, in Python and Rust.
      - **`po.gram.solve`** cannot reproduce a `ridge_decay` or `coef_prior`
        fit, though its docs promise "the model's own algebra"; the docs
        name the exception.
      - **`n_eff`**: it settles at `1 / (1 − λ^d)` for rows `d` clock units
        apart, not the documented `1 / (1 − λ)` (8,657 at halflife 600 with
        rows 0.1 apart, where the docs said 866), and the Kish size
        likewise; it is a weight, not a sample size, so "the effective
        number of observations" becomes "the weight behind the state", and
        `po.corr.signal_share` takes a Kish size: its argument is
        `n_kish_blocks`, was `n_eff_blocks`.
      - **`emit_selected` and `emit_averaged`** compare a halflife grid's
        errors over windows of different lengths; the docs say so, and
        point to `lasso`'s `select_halflife` for a like-for-like choice.
      - **The README's claim** that three diagnostics take no row weight is
        stale: zero-weight rows are skipped, and every residual diagnostic
        parts from the doubled stream.
      - **`label_delay`**'s "one unit per accepted row" is every row of the
        group, a skipped one included, and a gap capped by `max_dclock`
        releases the held rows as a session change does; the Rust `hmm`
        docs' account of where the transition prior sits matches the code.

      Tests: a constant target's intercept under `ridge_decay`, with and
      without a `coef_prior` intercept; `po.gram.solve` off a `ridge_decay`
      and a `coef_prior` fit, on beside a plain one; `n_eff` and `n_kish` at
      `1 / (1 − λ^d)` and `(1 + λ^d) / (1 − λ^d)` at rows 0.1 apart; a
      halflife grid's `sigma` against the EW recursion at each slot's own
      halflife, and `selected` its argmin; a skipped row counted by a
      clockless `label_delay`; the spread of hand-computed EW correlations
      of noise explained by `1 / (n_kish − 3)` and twice the floor at
      `n_eff`. The docstring pass under `docs/WRITING.md` is task 149.

<a id="task-149"></a>
- [x] 149. **A docstring pass under `docs/WRITING.md` -- done
      (2026-10-03).** Size M. Task 148 put the facts right; the pass rewrites
      the docstrings to the README's rules: map first, tables over lists, a
      mechanism named rather than alluded to, sentences near 20 words,
      every fact checked against the code. Each page is drafted beside its
      old text, accounted by a diff of the backticked names, the numbers and
      the link targets (nothing lost; a stale fact corrected against the
      code is noted here), measured before and after, and checked by the
      doc tests and Sphinx `-W`.

      ### Task 149 -- the docstring pass under docs/WRITING.md (map, before any prose)
      The map, the baseline and the pages done follow this entry, at the
      document's own level.

**Task 149, step 1, measured 2026-10-03 before the pass** (`measure_docstrings.py`, the
public docstrings the reference renders: the spec module and its 20 builders,
`ModelBank`, the two `online` namespaces, `stream`, `ops`, `eval`, `po.target`):
93 docstrings, 1,540 sentences, a mean of 23.3 words, 116 sentences of 45 words
or more, 104 bullet lines. The worst by long sentences: `spec.marginal` (13 of
87), the `spec` module (11 of 148, 48 bullet lines), `spec.ewridge` (7),
`stream.with_windows` (5, mean 34.6), `spec.micro` (5), `stream.embargo` (4,
mean 35.3), the `ops` module (4, mean 33.2), the `stream` module (3, mean
41.6); `ModelBank.last_row` (mean 40.6), `lf.online.unnest` (35.3),
`ModelBank.fit` (32.2), `fit_predict` (33.2, 7 bullet lines),
`closed_groups` (31.2, 5 bullets).

**The reader and the altitude** (WRITING §0, §1): a docstring is a deep doc,
reached from the Sphinx reference or `help()`, by a reader who has the
README's words; it carries mechanisms, equations, units, defaults and
refusals, and the README keeps the consequence. Nothing is cut: a sentence
that leaves a docstring lands in the README, `docs/OUTPUTS.md`,
`docs/PERFORMANCE.md` or `CHANGELOG.md`, and the cross-reference stays.

**Task 149, the map, per page** (`←` marks what moves; the order is the order of work):

| page | docstrings | the map each keeps | what changes |
|---|---|---|---|
| `spec` module | 1 | what a spec is → the stream parameters (one entry each, why / what / units) → clock units (the table stays) → what a spec writes (fields, the grid block, the diagnostics table) → errors | long entries split at the idea (`gap_cap`, `restart_after_step_back`, `embargo`, `min_weight`); the diagnostics table's long cells split into two sentences each; no history |
| builders, linear family: `ewridge`, `rls`, `kalman`, `lasso`, `huber`, `quantile`, `sgd`, `pa`, `ftrl` | 9 | one paragraph on what the model is → the update equations as a code block → parameters (why / what / units, default stated) → output fields → example as code with comments → raises | sweeps become tables (`ewridge`'s solvers and gates, `lasso`'s path, `kalman`'s clock forms); measurements → one number with its PERFORMANCE section |
| builders, classification and tests: `ew_class`, `seqtest` | 2 | as above | as above |
| builders, unsupervised: `ew_cov`, `kmeans`, `micro`, `deco`, `bocpd`, `corrchange`, `hmm`, `rcov`, `marginal` | 9 | as above | `marginal`: `serial_rule`'s three rules → a table (rule, the bracket, null when); the lag and window sizes → a table; `shards` and `feature_moments` → one number each ← PERFORMANCE §25, §27; `bins` split into what it reports, how edges are learned, memory (a table) |
| `holt` | 1 | as above | -- |
| `ModelBank` | 1 class + its methods | the class: what a bank is → the three ways in → state; each method: what it does → arguments → returns → raises | `fit_predict`/`fit`/`last_row`/`closed_groups`: bullets of parallel things → tables; sentences split |
| `lf.online`, `df.online` | 2 namespaces | one line per method, the plan form first (WRITING §2) | `unnest`'s sentences split |
| `stream` module + `embargo`, `with_windows`, `refresh_time`, `sample` | 5 | what the transform is → the clock paragraph → resumes → raises | the clock paragraph's sweep of policies → a table (keyword, what it does, units); `with_windows`'s resume paragraph stays prose (an argument) |
| `ops` module + 7 operators | 8 | the semantics paragraph (membership rules) → the ratio paragraph → the operators | the membership rules → a table (closed, backward near/far, forward near/far); the exactness claim stays one sentence |
| `eval` | the metrics functions | what each computes → arguments → returns | sentences split |
| `po.target` | 1 | -- | sentences split |

**Steps 3 to 8** (WRITING §6): draft a page, run its checks, account by a
diff of the backticked names, numbers and link targets (a script in the
scratchpad, `account.py`, prints what the new text lost), measure again, and
the gate's doc tests (`tests/test_examples.py`, `tests/test_api_links.py`,
`tests/test_doc_structure.py`, Sphinx `-W`).

**Task 149, pages done.** The counts are sentences, the mean in words, and
the number of sentences of 45 words or more, before and after, by
`measure_pages.py` (the scratchpad's): headings, code blocks and list-tables
dropped, then each paragraph split on its own, as WRITING §6 says (a first
count that merged table cells overstated the after-counts). Nothing moved
out of a page unless the row says so; a stale fact corrected against the
code is named.

| page | before | after | moved or corrected |
|---|---|---|---|
| `spec` module | 124, 19.5, 4 | 166, 14.5, 0 | `gap_cap`'s "``embargo`` releases the rows it holds" at a break was stale since task 153 (a break releases nothing early, `stream.rs`): corrected; the two refusal enumerations became lists |
| `spec.marginal` | 90, 24.6, 12 | 110, 18.0, 1 | `serial_rule`'s three rules, the bins' two memory sizes and `window_lags`' snapshot sizes became tables; the raises a list |
| `spec.ewridge` | 75, 23.4, 5 | 81, 18.7, 1 | `ridge_scale`'s two settings, `target_gaps`' two settings and `window_budget`'s three cases became tables |
| `stream` module | 14, 21.4, 0 | 16, 17.8, 0 | the history paragraph (`polars_online.prep`, `time=`/`by=`) left for the CHANGELOG, which records the rename (0.12.0) |
| `stream.embargo` | 17, 31.1, 3 | 24, 20.2, 0 | the two row kinds became a table; the raises a list; "cheaper" named as what is spent |
| `stream.with_windows` | 25, 33.3, 6 | 36, 21.3, 0 | the clock keywords became a table |
| `spec.micro` | 47, 20.2, 4 | 52, 17.3, 0 | the two wrong values of `eps` became the failure-mode table (WRITING §7's own example) |
| `ops` module | 29, 25.7, 3 | 21, 19.7, 0 | the operators' definitions, the `closed` membership rules and `partial`'s three values became tables |
| `stream.refresh_time` | 31, 25.8, 2 | 44, 18.1, 0 | the raises became a list |
| `spec.ew_cov` | 49, 23.2, 2 | 54, 18.2, 0 | the four windowed caveats and the PCA's four fields became tables; "the cost" named as the work |
| all ten (batch one) | 501, 23.3, 41 (106 of 35 or more) | 604, 17.5, 2 (24) | |
| `spec.corrchange` | 72, 21.1, 3 | 85, 17.4, 1 | `boundary_gamma`'s three settings became a table; the raises a list |
| `spec.kmeans` | 37, 19.6, 3 | 40, 17.6, 0 | the four seeding rules became a table |
| `spec.bocpd` | 45, 22.4, 2 | 45, 18.2, 0 | the three emissions, the four prior parameters and the three breaks' `p_change` became tables; "costs" named as the work |
| `spec.ew_class` | 29, 22.0, 2 | 29, 19.8, 0 | the three covariance shapes became a table |
| `spec.deco` | 37, 20.9, 2 | 46, 16.7, 0 | the raises became a list |
| `spec.ftrl` | 31, 20.5, 2 | 37, 17.3, 0 | the wrap of "a prior of fixed mass" kept as `test_weight_scale.py` pins it |
| `spec.holt` | 27, 19.7, 1 | 31, 17.5, 0 | -- |
| `spec.hmm` | 45, 19.4, 1 | 53, 16.2, 0 | the raises became a list |
| `spec.seqtest` | 34, 19.7, 1 | 38, 17.7, 0 | -- |
| `ModelBank.fit` | 13, 22.5, 2 | 16, 19.6, 1 (a bold lead merged) | -- |
| `ModelBank.summary` | 20, 17.9, 2 | 21, 17.1, 1 (a header merged) | -- |
| `ModelBank.last_row` | 7, 27.0, 1 | 9, 19.1, 0 | -- |
| `ModelBank.skip_learned` | 10, 23.1, 1 | 13, 16.8, 0 | the raises became a list |
| `ModelBank.fit_predict` | 20, 19.0, 1 | 21, 18.2, 1 (a lead-in merged) | -- |
| `ModelBank.closed_groups` | 24, 23.2, 1 | 18, 20.5, 0 | the blocks per kind became a table |
| `lf.online.unnest` | 8, 25.0, 2 | 14, 14.0, 0 | the errors became a list |
| `lf.online.predict` | 7, 26.9, 1 | 12, 15.8, 0 | the errors became a list |
| `eval.rolling_metrics` | 3, 30.7, 1 | 7, 14.6, 0 | the raises became a list |
| `eval.seqtest` | 13, 21.5, 1 | 18, 15.5, 0 | the raises became a list |
| `ops.increment` | 3, 37.3, 1 | 5, 22.2, 0 | -- |
| all twenty (batch two) | 485, 21.1, 31 (87 of 35 or more) | 558, 17.5, 4 (28) | the four are a bold lead sentence and definition-list headers the splitter merges with their bodies |
| `spec.kalman` | 35, 18.3, 1 | 36, 17.8, 0 | "costs nothing" named as what is added |
| `spec.lasso` | 36, 17.8, 1 | 38, 17.0, 0 | "costs nothing" named as the solve it saves |
| `spec.coef_fields` | 12, 16.3, 1 | 13, 15.3, 0 | -- |
| `eval.metrics` | 13, 15.8, 1 | 16, 12.8, 0 | -- |
| `eval.unpack` | 9, 15.6, 1 | 12, 12.0, 0 | the raises became a list |
| `eval.sums` | 16, 14.8, 1 | 18, 13.6, 0 | "pays for" named as the term it takes |
| all six (batch three) | 121, 17.0, 6 (13 of 35 or more) | 133, 15.6, 0 (6) | |

**Task 149, steps 6 to 8 (2026-10-03).** Measured again over all 93 public
docstrings, side by side, with `measure_docstrings.py` dropping headings,
code blocks and list-tables and splitting each paragraph on its own (the
first count of 116 sentences of 45 words or more merged text across table
cells; this count is the fair one at both ends):

| | sentences | mean words | 45 or more | 35 or more |
|---|---:|---:|---:|---:|
| before the pass (`aa30f00`) | 1,666 | 20.7 | 78 | 261 |
| after it | 1,854 | 17.7 | 6 | 113 |

The remaining sentences of 45 or more are the splitter's: a bold lead
sentence (`**With one exception...**`) and definition-list headers joined
to their bodies. Thirty-six pages were rewritten (every page with such a
sentence); the other 57 already measured within the rules and were left as
they were, each a reference entry near 20 words a sentence. Step 7 (render)
is Sphinx `-W` over the reference, which the gate runs; step 8 is the gate's
doc tests (`test_examples`, `test_api_links`, `test_doc_structure`,
`test_api_surface`), the docstring-pin tests (`test_weight_scale`,
`test_temporal_clock`) and Sphinx. Done.

<a id="task-154"></a>
- [x] 154. **The reader-facing documents rewritten after tasks 139 to 153,
      104 and the review rounds -- requested 2026-10-03.** Size L. The
      user's words: "Rewrite each user facing non requirements Md document
      ensuring that it has all the necessary sections after the recent
      additions. Review the organization of the sections to keep
      similarities together and ensure that important information is
      presented first in each section and that all phrasing and
      documentation rules are followed." Task 138 rewrote the README on
      2026-09-29, at `5e96018`; since then came tasks 139 to 153 and 104
      and nine review rounds of the window operators (§14), each adding
      text where it was built. This pass runs `docs/WRITING.md` §6 over
      every guide, against the code at `df555b2`.

      ### Task 154 -- the documentation pass (scope, baseline and maps, before any prose)
      The scope, the baseline and each document's map follow this entry, at
      the document's own level.
