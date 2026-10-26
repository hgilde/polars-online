//! Frozen state fixtures: one state of every [`ModelState`] variant, saved
//! mid-stream with what it keeps, embedded as bytes beside the rows that
//! follow it and the numbers they reported (docs/PLAN.md task 198; review
//! round 4, D1 and CF1). Each is held three ways:
//!
//! 1. **it loads**: the bytes decode and the model's `restore` takes them;
//! 2. **it goes on to the bit**: the rows after the save report what they
//!    reported when the fixture was written -- to the bit on the platform
//!    that wrote it, and elsewhere to the golden files' tolerance, since a
//!    decay or a log-likelihood runs through libm, whose last bit differs by
//!    platform (CLAUDE.md, no libm in the state); a NaN is a NaN on both;
//! 3. **it saves its bytes again**: the model restored from them writes the
//!    same bytes.
//!
//! The bank's own fixtures -- a bank file, a `with_windows` state and a
//! `refresh_time` state -- are `online-polars`' `tests/state_fixtures.rs`,
//! on the same three checks.
//!
//! **The fixtures are the schema's.** Every schema from `MIN_SCHEMA_VERSION`
//! to `SCHEMA_VERSION` has a set, and `the_fixtures_cover_the_schemas_this_build_loads`
//! refuses a range one does not cover: a layout change cannot land without
//! its fixtures. Before 1.0 a layout change regenerates the set and raises
//! the minimum with the version (hard rule 5's waiver):
//!
//! ```text
//! PRINT_STATE_FIXTURES=1 cargo test -p online-core --test state_fixtures
//! ```
//!
//! rewrites every file under `tests/state_fixtures/` from the cases in
//! `tests/cases/mod.rs`, and a run without the variable checks them. A
//! change that moves a number moves a continuation too: regenerate after
//! confirming the move is intended, as for `tests/golden.rs`.
//!
//! **From 1.0** the minimum stays at 1.0's schema, and a layout change
//! keeps the fixtures of the schema before it (docs/PLAN.md §18 D1; review
//! round 5, D1 and D2). The first 1.x layout change, from schema 44 to 45,
//! goes like this:
//!
//! 1. copy the current set, every `<case>.rs` and `index.rs`, to
//!    `tests/state_fixtures/v44/` as it is -- the files are the schema's,
//!    and `v44/index.rs` lists them as the current index does;
//! 2. raise `SCHEMA_VERSION` to 45 and write the loader the change needs,
//!    in the model's `Deserialize` (`check_schema`'s doc);
//! 3. list the kept schema here: `previous![44]`, which includes
//!    `v44/index.rs` -- a schema listed without its files does not compile;
//! 4. regenerate the current set, which is then schema 45's.
//!
//! Each kept set is held by `every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes`:
//! every fixture loads through the loader, goes on to the bit against its
//! own frozen continuation, and -- loaded, then saved -- writes the current
//! schema's fixture bytes of the same case, since the two are the same case
//! over the same rows (the exact-conversion check). A later change repeats
//! the steps with `v45/`, and `previous![44, 45]`: the loaders of 1.x stay.
//!
//! The variant names are the files' tags (the named encoding writes
//! `{"EwRidge": {...}}`), so they are frozen here too: a renamed variant
//! breaks every saved file.

mod cases;

#[path = "state_fixtures/index.rs"]
mod frozen;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use cases::{Case, Fixture, Out, agree, all, hex, load, named, saved, writer};
use online_core::{MIN_SCHEMA_VERSION, ModelState, SCHEMA_VERSION, State};

/// The schemas before the current one whose fixtures are kept, each under
/// `tests/state_fixtures/v<N>/` and loaded by this build: `PREVIOUS` lists
/// them, and `previous_sets` is each set as its `v<N>/index.rs` lists it.
/// A schema listed without its files does not compile. None before 1.0,
/// where a layout change regenerates the set instead (the module doc has
/// the steps from 1.0). 61, 1.0's, kept since 62 moved a blocked
/// `ewridge`'s layout (docs/PLAN.md task 257): its states load through the
/// field's default, a blocked one's targets current through its held rows.
/// No kept case is blocked, so each converts to the current bytes exactly;
/// a blocked state saved mid-block at 61 goes on from its held rows within
/// the fold's bound of the definition, not to the bits a 62 build reaches
/// from the same rows (`gaps::fold_tests`). 62 kept since 63 gave `huber`
/// and `sgd(loss="huber")` a robust scale (docs/PLAN.md task 271): a 62
/// state has no `huber_scale` and was fitted under the RMS, so it loads as
/// `huber_scale = "rms"` (`HuberScale::of_a_state_before_63`), with no
/// sketch, and goes on to the bit; the `huber` and `sgd_huber` cases are
/// the RMS's, so each converts to the current bytes exactly, and the MAD's
/// form is the new cases `huber_mad` and `sgd_huber_mad`.
///
/// 63 kept since 64 gave `huber` and `sgd(loss = "huber")` the refit of
/// each target's first rows (docs/PLAN.md task 277): a 63 state has no
/// `refit_rows` and no buffers and was fitted without a refit, so it loads
/// with `refit_rows = 0` and none, and goes on to the bit, never refitting
/// rows its stream has long passed; every kept case is the refit's off
/// form, so each converts to the current bytes exactly, and the refit's
/// forms are the new cases `huber_refit` and `sgd_huber_refit`, saved
/// with their buffers half full, and `huber_refit_done` and
/// `sgd_huber_refit_done`, saved after the refit. 64 gave `kalman` its
/// covariance in units of the noise and a diffuse intercept too (task
/// 278): a 63 state has no `noise_units` and loads as the form it was
/// fitted in, `P` absolute and the noise estimate in the gain, and goes on
/// to the bit; the kept `kalman` and `kalman_gap` cases are that form, so
/// each converts to the current bytes exactly, and the new form is the
/// case `kalman_noise_units`, saved with its intercepts still diffuse.
///
/// The windowed cases convert otherwise since the running products
/// (schema 63 before its release, docs/PLAN.md task 275): a ring saved
/// without them loads with none, so the first read after loading is the
/// saved state's to the bit, and its snapshots go on decaying by the summed
/// clock until they leave. Once a snapshot taken after the load is the
/// boundary, its products decay it and the reads part from the old build's
/// in the last bits (from row 12 of 16 in `ewridge_window`), as a fresh
/// run's do, within the golden tolerance; and the state it saves is not
/// the current fixture's, whose reads took the products from the first
/// row. So a windowed case is held to its first row exactly, the rest to
/// the golden tolerance, and to saving bytes that load back to the same.
///
/// The `kmeans` cases convert otherwise since task 278 (before 63's
/// release; rule 5's second exception, the user's, 2026-10-25): `init =
/// "lloyd"` starts from greedy k-means++ and runs each start to its fixed
/// point, where it took ten Lloyd passes from plain k-means++. A `kmeans`
/// state of schema 61-63 saved before it seeded seeds by the new rule after
/// loading: the same centres and distances to the bit, the cluster labels
/// possibly permuted, held to one consistent relabelling of the frozen
/// labels on every row (`kmeans_warming`, which seeds four rows after the
/// save; its bytes are the current set's). A state saved after seeding
/// continues to the bit (`kmeans`); its centres are in the old rule's
/// order, so its bytes are not the current set's, which the new rule
/// seeded, and it is held to saving bytes that load back to the same.
///
/// The `lasso` cases convert otherwise since task 278 too (rule 5's third
/// exception, the user's, 2026-10-25): the descent's stopping rule reads a
/// sweep's move against `tol` times the target's spread, where it read it
/// against `tol` itself, so where the spread is not 1 a descent stops on
/// another sweep. A `lasso` state of schema 61-63 continues within 1e-9
/// relative, not to the bit, and re-saves the same bytes; [`MOVED`] lists
/// the cases and their bound, and holds an unwindowed one's converted
/// state to the current fixture's field for field, its floats within the
/// bound.
macro_rules! previous {
    ($($schema:literal),* $(,)?) => {
        const PREVIOUS: &[u32] = &[$($schema),*];

        fn previous_sets() -> Vec<(u32, &'static [&'static Fixture])> {
            vec![$({
                mod set {
                    include!(concat!("state_fixtures/v", $schema, "/index.rs"));
                }
                // The index is the set as it was generated, its schema with it.
                assert_eq!(set::SCHEMA, $schema, "v{}/index.rs is another schema's", $schema);
                ($schema, set::ALL)
            }),*]
        }
    };
}

previous![61, 62, 63];

/// The cases whose numbers a later fix moved without touching the layout,
/// each with how far its continuation may part from a previous set's
/// frozen one. A previous fixture of such a case was fitted under the old
/// arithmetic, so neither its continuation nor its converted state is the
/// current build's to the bit: it is held to loading, to saving bytes that
/// load back to the same, and to its continuation within the bound, by
/// [`agree`]'s relative measure. The current set is regenerated as for any
/// change that moves a number, and holds them to the bit.
///
/// Its converted state is held to the current fixture's field for field:
/// the same keys, lengths and every number but a float the same, each
/// float within the bound ([`same_but_floats`]), so the loader's work is
/// still checked whole -- but under a window, whose converted ring differs
/// by design (the doc above `previous!`).
///
/// - `lasso`, `lasso_window`: the descent's stopping rule compares a
///   sweep's move with `tol` times the target's spread, not with `tol`
///   itself (task 278, D3), so a descent stops on another sweep where the
///   spread is not 1. On these rows that moved the predictions by at most
///   5.8e-11 (`lasso`) and 1.7e-11 (`lasso_window`) of `1 + |pred|`,
///   measured 2026-10-25 on both kept sets; the bound is 17 times the
///   larger.
const MOVED: &[(&str, f64)] = &[("lasso", 1e-9), ("lasso_window", 1e-9)];

/// How far a continuation may part from the one frozen, on a platform other
/// than the writer's: the golden files' tolerance (`tests/golden.rs`).
const TOL: f64 = 1e-12;

const REGENERATE: &str = "PRINT_STATE_FIXTURES=1 cargo test -p online-core --test state_fixtures";

fn regenerating() -> bool {
    std::env::var("PRINT_STATE_FIXTURES").is_ok_and(|v| v == "1")
}

fn case_of(f: &Fixture) -> Case {
    all()
        .into_iter()
        .find(|c| c.name == f.name)
        .unwrap_or_else(|| {
            panic!(
                "{}: no case of that name; regenerate with `{REGENERATE}`",
                f.name
            )
        })
}

fn same_out(what: &str, got: &Out, want: &Out, tol: f64) {
    let pairs = got
        .pred
        .iter()
        .zip(&want.pred)
        .chain([(&got.n_eff, &want.n_eff)])
        .chain(got.extra.iter().zip(&want.extra));
    assert_eq!(
        (got.pred.len(), got.extra.len()),
        (want.pred.len(), want.extra.len()),
        "{what}: the row reports another number of slots"
    );
    for (i, (g, w)) in pairs.enumerate() {
        assert!(
            agree(*g, *w, tol),
            "{what}, number {i}: got {g:e} ({:#018x}), frozen {w:e} ({:#018x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

#[test]
fn every_fixture_loads() {
    if regenerating() {
        return;
    }
    for f in frozen::ALL {
        let case = case_of(f);
        let s: State = rmp_serde::from_slice(&f.bytes())
            .unwrap_or_else(|e| panic!("{}: the bytes do not decode: {e}", f.name));
        assert_eq!(s.schema_version, f.schema, "{}", f.name);
        let variant = format!("{:?}", s.model);
        assert!(
            variant.starts_with(&format!("{}(", f.variant)),
            "{}: a state of {}",
            f.name,
            variant.split('(').next().unwrap()
        );
        if let Err(e) = (case.restore)(&s) {
            panic!("{}: the state does not load: {e}", f.name);
        }
    }
}

#[test]
fn every_fixture_goes_on_to_the_bit() {
    if regenerating() {
        return;
    }
    for f in frozen::ALL {
        let case = case_of(f);
        let mut m = load(&case, &f.bytes()).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        let tol = if f.writer == writer() { 0.0 } else { TOL };
        for (i, (row, want)) in f.rows().iter().zip(f.outs()).enumerate() {
            let got = m.step(row);
            same_out(
                &format!("{}, row {i} after the save", f.name),
                &got,
                &want,
                tol,
            );
        }
    }
}

#[test]
fn every_fixture_saves_its_bytes_again() {
    if regenerating() {
        return;
    }
    for f in frozen::ALL {
        let case = case_of(f);
        let bytes = f.bytes();
        let m = load(&case, &bytes).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        assert!(
            named(&m.state()) == bytes,
            "{}: the state restored from the fixture saves other bytes",
            f.name
        );
    }
}

/// The set a kept schema has on disk: `dir/v<schema>/index.rs` and the
/// fixtures beside it, counted; or why there is none.
fn set_on_disk(dir: &Path, schema: u32) -> Result<usize, String> {
    let set = dir.join(format!("v{schema}"));
    if !set.join("index.rs").is_file() {
        return Err(format!(
            "schema {schema} is listed in PREVIOUS and has no set under {}: a kept schema keeps \
             its fixtures there, with an index.rs listing them",
            set.display()
        ));
    }
    let fixtures = std::fs::read_dir(&set)
        .map_err(|e| format!("{}: {e}", set.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs") && !p.ends_with("index.rs"))
        .count();
    if fixtures == 0 {
        return Err(format!(
            "schema {schema}: {} holds an index and no fixture",
            set.display()
        ));
    }
    Ok(fixtures)
}

/// Every schema this build claims to load has its fixtures, and every
/// fixture is of a schema it loads: so `SCHEMA_VERSION` cannot move without
/// a set of its own, and `MIN_SCHEMA_VERSION` cannot sit below the oldest
/// set kept. A kept schema's set is on disk, under `v<N>/`, with as many
/// fixtures as its index lists (review round 5, D2).
#[test]
fn the_fixtures_cover_the_schemas_this_build_loads() {
    if regenerating() {
        return;
    }
    assert!(
        frozen::ALL.iter().all(|f| f.schema == frozen::SCHEMA),
        "one schema a set"
    );
    assert_eq!(
        frozen::SCHEMA,
        SCHEMA_VERSION,
        "SCHEMA_VERSION is {SCHEMA_VERSION} and the frozen state fixtures are of {}: a layout \
         change ships its fixtures. Before 1.0, regenerate them with `{REGENERATE}` and raise \
         MIN_SCHEMA_VERSION (and the bank's minimum) to the new schema; from 1.0, keep this set \
         as the previous schema's (the module doc), write its loader, then regenerate",
        frozen::SCHEMA
    );
    let covered: Vec<u32> = PREVIOUS.iter().copied().chain([frozen::SCHEMA]).collect();
    let claimed: Vec<u32> = (MIN_SCHEMA_VERSION..=SCHEMA_VERSION).collect();
    assert_eq!(
        claimed, covered,
        "this build loads schemas {MIN_SCHEMA_VERSION}..={SCHEMA_VERSION} and fixtures cover \
         {covered:?}: each schema it loads needs a frozen set, held to its loader"
    );
    let sets = previous_sets();
    assert_eq!(
        sets.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        PREVIOUS,
        "the kept sets are the schemas listed"
    );
    for (schema, set) in &sets {
        let on_disk = set_on_disk(&dir(), *schema).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            on_disk,
            set.len(),
            "schema {schema}: v{schema}/ holds {on_disk} fixtures and its index lists {}",
            set.len()
        );
    }
}

/// The on-disk check behind the coverage test, proven on a directory of
/// its own while `PREVIOUS` is empty (review round 5, D2): a listed schema
/// with no set, or an index alone, is refused; a set with fixtures passes
/// and is counted.
#[test]
fn a_kept_schema_without_its_set_on_disk_is_refused() {
    let root = std::env::temp_dir().join(format!(
        "polars-online-state-fixture-sets-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let missing = set_on_disk(&root, 44).unwrap_err();
    assert!(
        missing.contains("schema 44 is listed") && missing.contains("v44"),
        "{missing}"
    );
    let set = root.join("v44");
    std::fs::create_dir_all(&set).unwrap();
    std::fs::write(set.join("index.rs"), "// an index\n").unwrap();
    let empty = set_on_disk(&root, 44).unwrap_err();
    assert!(empty.contains("no fixture"), "{empty}");
    std::fs::write(set.join("ewridge.rs"), "// a fixture\n").unwrap();
    std::fs::write(set.join("notes.txt"), "not a fixture\n").unwrap();
    assert_eq!(set_on_disk(&root, 44), Ok(1));
    std::fs::write(set.join("sgd.rs"), "// a fixture\n").unwrap();
    assert_eq!(set_on_disk(&root, 44), Ok(2));
    assert!(set_on_disk(&root, 43).is_err(), "another schema has no set");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Each kept schema's fixtures, held to the loader the change after it
/// shipped (the module doc; review round 5, D1 and D2): every one loads,
/// goes on to the bit against its own frozen continuation -- as it did
/// when written -- and, loaded then saved, writes the current schema's
/// fixture bytes of the same case: the same case over the same rows, so
/// the converted state is the one this build writes, field for field (the
/// exact-conversion check). The sets must share a writer, since a libm
/// bit in a decayed sum differs by platform. Nothing to run before 1.0,
/// where `PREVIOUS` is empty.
#[test]
fn every_previous_fixture_loads_goes_on_and_converts_to_the_current_bytes() {
    if regenerating() {
        return;
    }
    for (schema, set) in previous_sets() {
        assert!(!set.is_empty(), "schema {schema}: an empty set");
        for f in set {
            let what = format!("v{schema}/{}", f.name);
            assert_eq!(f.schema, schema, "{what}: a fixture of schema {}", f.schema);
            let case = case_of(f);
            let mut m = load(&case, &f.bytes()).unwrap_or_else(|e| {
                panic!("{what}: the state does not load through this build's loader: {e}")
            });
            let current = frozen::ALL
                .iter()
                .find(|c| c.name == f.name)
                .unwrap_or_else(|| {
                    panic!("{what}: the current set has no fixture of that case; a kept case stays")
                });
            assert_eq!(
                current.writer, f.writer,
                "{what}: written on {}, the current set on {}: the sets are written on one \
                 platform, or their bytes differ in libm's last bit",
                f.writer, current.writer
            );
            let windowed = holds_a_window(&f.bytes());
            let moved = MOVED.iter().find(|(n, _)| *n == f.name).map(|&(_, b)| b);
            // Seeded by the `lloyd` rule before task 278 (plain k-means++
            // starts, ten Lloyd passes): the rule since finds the same
            // centres in another order (the doc above `previous!`).
            let reseeded = f.name == "kmeans_warming";
            if f.name == "kmeans" {
                let saved = named(&m.state());
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
            } else if windowed || moved.is_some() {
                // A window's snapshots from before task 275 carry no running
                // products (the doc above `previous!`): converted, they say
                // so, and the state saves bytes that load back to the same.
                // A moved case's (`MOVED`) state is the old arithmetic's,
                // and saves bytes that load back to the same.
                let saved = named(&m.state());
                assert_eq!(holds_a_window(&saved), windowed, "{what}: the window");
                let again = load(&case, &saved).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    named(&again.state()) == saved,
                    "{what}: saved, loaded, saved"
                );
                // A window's converted ring lacks the running products, as
                // above, so a windowed case is not held field for field.
                if let Some(bound) = moved.filter(|_| !windowed) {
                    let decode = |b: &[u8]| rmpv::decode::read_value(&mut &b[..]).unwrap();
                    same_but_floats(&decode(&saved), &decode(&current.bytes()), bound, &what);
                }
            } else {
                assert!(
                    named(&m.state()) == current.bytes(),
                    "{what}: loaded, the state does not save the current schema's fixture bytes \
                     of the same case: the loader converts it to another state than this build \
                     reaches from the same rows"
                );
            }
            let tol = if f.writer == writer() { 0.0 } else { TOL };
            let mut relabel: Vec<(u64, u64)> = Vec::new();
            for (i, (row, want)) in f.rows().iter().zip(f.outs()).enumerate() {
                if reseeded {
                    let got = m.step(row);
                    let (g, w) = (got.pred[0], want.pred[0]);
                    if !(g.is_nan() && w.is_nan()) {
                        let pair = (w.to_bits(), g.to_bits());
                        assert!(
                            relabel.iter().all(|&(a, b)| (a == pair.0) == (b == pair.1)),
                            "{what}, row {i} after the save: label {w} read as {g}, not one \
                             relabelling of the frozen labels ({relabel:?})"
                        );
                        relabel.push(pair);
                    }
                    let mut want = want.clone();
                    want.pred[0] = g;
                    same_out(&format!("{what}, row {i} after the save"), &got, &want, tol);
                    continue;
                }
                // Under a window, the first read is the saved state's to the
                // bit; once a snapshot taken after the load is the boundary,
                // its products decay it, and the reads part from the old
                // build's in their last bits, as a fresh run's do.
                let tol = if windowed && i > 0 { tol.max(TOL) } else { tol };
                let tol = moved.map_or(tol, |b| tol.max(b));
                let got = m.step(row);
                same_out(&format!("{what}, row {i} after the save"), &got, &want, tol);
            }
        }
    }
}

/// `got` is `want` field for field -- the same map keys in the same order,
/// the same array lengths, every value but a float equal -- and each float
/// within `bound` of its twin by [`agree`]'s measure: a [`MOVED`] case's
/// converted state beside the current fixture's.
fn same_but_floats(got: &rmpv::Value, want: &rmpv::Value, bound: f64, what: &str) {
    use rmpv::Value as V;
    match (got, want) {
        (V::Map(g), V::Map(w)) => {
            assert_eq!(g.len(), w.len(), "{what}: a map of another size");
            for ((gk, gv), (wk, wv)) in g.iter().zip(w) {
                assert_eq!(gk, wk, "{what}: another key");
                same_but_floats(gv, wv, bound, &format!("{what}.{gk}"));
            }
        }
        (V::Array(g), V::Array(w)) => {
            assert_eq!(g.len(), w.len(), "{what}: an array of another length");
            for (i, (gv, wv)) in g.iter().zip(w).enumerate() {
                same_but_floats(gv, wv, bound, &format!("{what}[{i}]"));
            }
        }
        (V::F64(g), V::F64(w)) => {
            assert!(agree(*g, *w, bound), "{what}: {g:e} against {w:e}");
        }
        _ => assert_eq!(got, want, "{what}"),
    }
}

/// Whether a saved state holds a window's ring of snapshots: a model's
/// `win.snaps.ring`, non-empty.
fn holds_a_window(bytes: &[u8]) -> bool {
    let v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let Some(model) = at(&v, &["model"]).and_then(rmpv::Value::as_map) else {
        return false;
    };
    model.iter().any(|(_, m)| {
        at(m, &["win", "snaps", "ring"])
            .and_then(rmpv::Value::as_array)
            .is_some_and(|r| !r.is_empty())
    })
}

/// The variant names of [`ModelState`], as serde lists them: the one place
/// the list exists outside the enum.
fn variants() -> Vec<String> {
    let err = serde_json::from_str::<ModelState>(r#"{"Nope": null}"#)
        .unwrap_err()
        .to_string();
    let quoted: Vec<&str> = err.split('`').skip(1).step_by(2).collect();
    assert_eq!(quoted[0], "Nope", "{err}");
    quoted[1..].iter().map(|s| s.to_string()).collect()
}

/// The variant names are the tags every state file carries, so a rename
/// breaks every file written before it: they are frozen. A new model adds a
/// name here, and a fixture.
#[test]
fn the_variant_names_are_frozen() {
    const FROZEN: [&str; 22] = [
        "EwCov",
        "EwRidge",
        "Rls",
        "Lasso",
        "Kalman",
        "Robust",
        "Ftrl",
        "EwCovModel",
        "Sgd",
        "Pa",
        "Holt",
        "KMeans",
        "Micro",
        "EwClass",
        "SeqTest",
        "Marginal",
        "Deco",
        "Rcov",
        "Hmm",
        "CorrChange",
        "Bocpd",
        "Audit",
    ];
    assert_eq!(
        variants(),
        FROZEN,
        "a ModelState variant was renamed, removed or added: its name is the tag in every \
         state file, so a rename needs a loader for the old tag; a new one adds its name here \
         and a case in tests/cases/mod.rs"
    );
}

/// Every variant has a fixture, and every case is frozen: a case added or
/// renamed in `tests/cases/mod.rs` needs the fixtures regenerated.
#[test]
fn every_variant_and_every_case_has_a_fixture() {
    if regenerating() {
        return;
    }
    let names: Vec<&str> = frozen::ALL.iter().map(|f| f.name).collect();
    let cases: Vec<&str> = all().iter().map(|c| c.name).collect();
    assert_eq!(
        names, cases,
        "the frozen fixtures are not the cases: regenerate with `{REGENERATE}`"
    );
    for v in variants() {
        assert!(
            frozen::ALL.iter().any(|f| f.variant == v),
            "{v}: no frozen state; add a case to tests/cases/mod.rs and regenerate"
        );
    }
    for c in all() {
        assert!(
            c.before > 0 && c.rows().len() == c.before + cases::CONTINUATION,
            "{}",
            c.name
        );
    }
}

/// `v` at a path of map keys, or `None` where a key is absent.
fn at<'a>(v: &'a rmpv::Value, path: &[&str]) -> Option<&'a rmpv::Value> {
    path.iter().try_fold(v, |v, key| {
        v.as_map()?
            .iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, v)| v)
    })
}

/// How much of a layout `v` holds: its finite, non-zero numbers, through
/// any nesting. An empty layout writes an empty vector, a zero (`rls`'s
/// `s2` before a row) or a NaN (a share no solve has fit).
fn held(v: &rmpv::Value) -> usize {
    match v {
        rmpv::Value::Array(a) => a.iter().map(held).sum(),
        rmpv::Value::Map(m) => m.iter().map(|(_, v)| held(v)).sum(),
        v => usize::from(v.as_f64().is_some_and(|f| f.is_finite() && f != 0.0)),
    }
}

/// A layout a schema bump moved is written by a fixture that holds it, so
/// a change to its tags or fields cannot pass the harness unseen (review
/// round 5, D3): `sgd`'s per-loss state -- the residual scale under the
/// Huber loss (schema 40), the target's spread under the epsilon-insensitive
/// one (44) -- `ewridge`'s kept systems (38, under `set_keep_factor`),
/// `bocpd`'s warm-up rows (40), `rls`'s second weight sum `s₂` and the
/// robust models' data shares (45, docs/PLAN.md task 116). Each was empty
/// in every fixture before. The standardizing models' warm-up (47,
/// docs/PLAN.md task 206) is written twice for each of `sgd` and `pa`:
/// saved inside it, its count short of 22 and not switched, and saved past
/// it, switched, where the coefficients mean another thing. `kalman`'s
/// anchor, the squared innovations its prior is sized from and its clock
/// since an observation (49, docs/PLAN.md task 211): the last written by
/// `kalman_gap`, saved after a row whose targets are null, where `kalman`'s
/// clocks are 0, its last row having observed both targets. The clusters'
/// centres as pairs (51, docs/PLAN.md task 215, D2): each summary in a list
/// keeps its own low parts, so the check runs through every element, and
/// some of `kmeans`' centres and of `micro`'s must hold one. The row counts
/// of `ewridge`'s and `lasso`'s cross-moments (52, docs/PLAN.md task 217),
/// per target and over every row, live and in every snapshot a window
/// holds. `audit`'s state (54, docs/PLAN.md task 223 (b)): its pairs, its
/// clock's regular steps, and per column its lag pairs and a digest with
/// both centroids and values waiting; its counters' two forms, and its
/// moments over a power of two (`audit_scaled`), are
/// `an_audits_counters_are_written_in_both_forms`'. Its smallest change per
/// column and the changes at it (59, task 233). A blocked `ewridge`'s held
/// targets (62, task 257), `acc.held`, written only while a block holds a
/// target's values: `ewridge_blocked` is saved mid-block. The Huber
/// scale's two forms (63, task 271): `huber_mad` and `sgd_huber_mad` keep
/// a sketch of `|e|` per target, `huber_mad` its own scale memory, and each
/// cfg names its scale, `"mad"` there and `"rms"` in `huber` and
/// `sgd_huber`; each MAD sketch names its band (task 271 (b)). `huber`'s
/// feature sets (63, task 272 (a)): `huber_sets` names them in its cfg and
/// holds a fit per target and set; `huber_sum` asks for the sum-scale
/// prior and the emitted scale. `marginal`'s Huber location per bin (63,
/// task 272 (c)): `marginal_bins_huber` keeps each cell's robust weight and
/// each target's location and scale, and its cfg names its statistic and
/// cut.
#[test]
fn every_layout_a_schema_moved_is_written_by_a_fixture() {
    if regenerating() {
        return;
    }
    const FORMS: [(&str, &[&str]); 37] = [
        ("huber_sets", &["model", "Robust", "cfg", "feature_sets"]),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "bins", "hist", "om"],
        ),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "bins", "hist", "scales", "w"],
        ),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "bins", "hist", "scales", "mad"],
        ),
        (
            "marginal_bins_huber",
            &["model", "Marginal", "cfg", "bins", "stat", "huber", "delta"],
        ),
        ("huber_mad", &["model", "Robust", "mad"]),
        (
            "huber_mad",
            &["model", "Robust", "cfg", "huber_scale_decay"],
        ),
        ("sgd_huber_mad", &["model", "Sgd", "mad"]),
        ("ewridge_blocked", &["model", "EwRidge", "acc", "held", "y"]),
        (
            "ewridge_blocked",
            &["model", "EwRidge", "acc", "held", "w_end"],
        ),
        ("ewridge", &["model", "EwRidge", "acc", "cross", "nj"]),
        ("ewridge", &["model", "EwRidge", "acc", "cross", "n"]),
        (
            "ewridge_window",
            &["model", "EwRidge", "acc", "cross", "nj"],
        ),
        ("ewridge_window", &["model", "EwRidge", "acc", "cross", "n"]),
        ("lasso", &["model", "Lasso", "acc", "cross", "nj"]),
        ("lasso", &["model", "Lasso", "acc", "cross", "n"]),
        ("lasso_window", &["model", "Lasso", "acc", "cross", "nj"]),
        ("lasso_window", &["model", "Lasso", "acc", "cross", "n"]),
        ("sgd", &["model", "Sgd", "warmup"]),
        ("sgd_warming", &["model", "Sgd", "warmup"]),
        ("pa", &["model", "Pa", "warmup"]),
        ("pa_warming", &["model", "Pa", "warmup"]),
        ("kalman", &["model", "Kalman", "anchor_hi"]),
        ("kalman", &["model", "Kalman", "anchor_lo"]),
        ("kalman", &["model", "Kalman", "anchor_scale"]),
        ("kalman", &["model", "Kalman", "basis"]),
        ("kalman_gap", &["model", "Kalman", "elapsed"]),
        ("sgd_huber", &["model", "Sgd", "sig2"]),
        ("sgd_huber", &["model", "Sgd", "wsig"]),
        ("sgd_eps", &["model", "Sgd", "spread"]),
        (
            "ewridge_leverage",
            &["model", "EwRidge", "ready", "systems"],
        ),
        ("bocpd_warming", &["model", "Bocpd", "warm"]),
        ("rls", &["model", "Rls", "s2"]),
        ("huber", &["model", "Robust", "support"]),
        ("quantile", &["model", "Robust", "support"]),
        ("audit", &["model", "Audit", "pairs"]),
        ("audit", &["model", "Audit", "clock", "regular"]),
    ];
    for (name, path) in FORMS {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let Some(field) = at(&v, path) else {
            panic!("{name}: the state has no {}", path.join("."));
        };
        assert!(
            held(field) > 0,
            "{name}: {} is empty, so the fixture holds nothing of the layout it is there for",
            path.join(".")
        );
        if path.last() == Some(&"warmup") {
            let switched = at(field, &["switched"]).and_then(rmpv::Value::as_bool);
            assert_eq!(
                switched,
                Some(!name.ends_with("_warming")),
                "{name}: saved on the wrong side of the switch"
            );
        }
    }
    // `huber_sum` asks for the sum-scale prior and the emitted scale, and
    // the rest do not (63, task 272 (a)).
    for (name, want) in [("huber_sum", true), ("huber", false), ("huber_sets", false)] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        for field in ["ridge_scale", "emit_scale"] {
            let got = at(&v, &["model", "Robust", "cfg", field]).and_then(rmpv::Value::as_bool);
            assert_eq!(got, Some(want), "{name}: cfg.{field}");
        }
    }
    // Each Huber form names its scale, and the RMS's keeps no sketch.
    for (name, variant, scale) in [
        ("huber", "Robust", "rms"),
        ("huber_mad", "Robust", "mad"),
        ("sgd_huber", "Sgd", "rms"),
        ("sgd_huber_mad", "Sgd", "mad"),
    ] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let got = at(&v, &["model", variant, "cfg", "huber_scale"]).and_then(rmpv::Value::as_str);
        assert_eq!(got, Some(scale), "{name}: cfg.huber_scale");
        let sketches = at(&v, &["model", variant, "mad"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the state has no mad"));
        assert_eq!(sketches.is_empty(), scale == "rms", "{name}: mad");
        // A MAD's sketch is bounded and names its band (docs/PLAN.md task
        // 271 (b)); a sketch without one loads unbounded.
        for q in sketches {
            let band = at(q, &["band"]).and_then(rmpv::Value::as_u64);
            let want = u64::from(online_core::MAD_BAND_OCTAVES);
            assert_eq!(band, Some(want), "{name}: mad.band");
        }
    }
    // Every snapshot a window holds keeps the counts as they stood before
    // its row; past the first rows, a target's are above 0.
    for (name, variant) in [("ewridge_window", "EwRidge"), ("lasso_window", "Lasso")] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let ring = at(&v, &["model", variant, "win", "snaps", "ring"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the state has no window ring"));
        assert!(!ring.is_empty(), "{name}: the window holds no snapshot");
        for (i, item) in ring.iter().enumerate() {
            let moments = item.as_array().and_then(|t| t.get(2)).unwrap_or_else(|| {
                panic!("{name}: snapshot {i} is no (clock, stamp, moments, products)")
            });
            for field in ["nj", "n"] {
                let Some(count) = at(moments, &["acc", "cross", field]) else {
                    panic!("{name}: snapshot {i} has no acc.cross.{field}");
                };
                assert!(
                    held(count) > 0,
                    "{name}: snapshot {i}'s acc.cross.{field} is empty"
                );
            }
        }
    }
    // Every windowed model keeps the running products of its decays, one
    // per decay its window is read with -- two for `lasso`, its selection's
    // beside its moments' -- and every snapshot the products it was taken
    // at (docs/PLAN.md task 275).
    for (name, variant, decays) in [
        ("ewridge_window", "EwRidge", 1),
        ("lasso_window", "Lasso", 2),
        ("ew_cov_window", "EwCovModel", 1),
        ("ew_class_window", "EwClass", 1),
        ("marginal_window", "Marginal", 1),
    ] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let snaps = at(&v, &["model", variant, "win", "snaps"])
            .unwrap_or_else(|| panic!("{name}: the state has no window"));
        let live = at(snaps, &["decay"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the window has no running products"));
        assert_eq!(live.len(), decays, "{name}: the running products");
        assert!(held(&rmpv::Value::Array(live.clone())) > 0, "{name}: decay");
        let ring = at(snaps, &["ring"])
            .and_then(rmpv::Value::as_array)
            .unwrap();
        assert!(!ring.is_empty(), "{name}: the window holds no snapshot");
        for (i, item) in ring.iter().enumerate() {
            let products = item
                .as_array()
                .and_then(|t| t.get(3))
                .and_then(rmpv::Value::as_array)
                .unwrap_or_else(|| panic!("{name}: snapshot {i} keeps no products"));
            assert_eq!(products.len(), decays, "{name}: snapshot {i}'s products");
            assert!(
                held(item.as_array().unwrap().get(3).unwrap()) > 0,
                "{name}: snapshot {i}"
            );
        }
    }
    type Listed<'a> = (&'a str, &'a [&'a str], &'a [&'a str]);
    const LISTED: [Listed; 8] = [
        ("kmeans", &["model", "KMeans", "clusters"], &["c_lo"]),
        // Schema 59 (task 233, C-1): each column's smallest change and the
        // changes at it, continuous values and a few values both.
        (
            "audit",
            &["model", "Audit", "columns"],
            &["min_change_count"],
        ),
        (
            "audit_exact",
            &["model", "Audit", "columns"],
            &["min_change_count"],
        ),
        ("micro", &["model", "Micro", "mc"], &["s", "c_lo"]),
        (
            "audit",
            &["model", "Audit", "columns"],
            &["digest", "centroids"],
        ),
        (
            "audit",
            &["model", "Audit", "columns"],
            &["digest", "buffer"],
        ),
        ("audit", &["model", "Audit", "columns"], &["lag"]),
        (
            "audit_exact",
            &["model", "Audit", "columns"],
            &["digest", "buffer"],
        ),
    ];
    for (name, list, inner) in LISTED {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let Some(items) = at(&v, list).and_then(rmpv::Value::as_array) else {
            panic!("{name}: the state has no list {}", list.join("."));
        };
        let mut lows = 0;
        for item in items {
            let Some(field) = at(item, inner) else {
                panic!(
                    "{name}: an element of {} has no {}",
                    list.join("."),
                    inner.join(".")
                );
            };
            lows += held(field);
        }
        assert!(
            lows > 0,
            "{name}: no {} in {} holds a number, so the fixture holds nothing of the layout",
            inner.join("."),
            list.join(".")
        );
    }
}

/// The refit's forms (schema 64, docs/PLAN.md task 277): `huber_refit` and
/// `sgd_huber_refit` are saved with every target's buffer part full -- rows
/// held, fewer learned than the configuration's count -- and their
/// continuations fill them, so the refit runs on a row the fixture holds;
/// `huber_refit_done` and `sgd_huber_refit_done` are saved after it, every
/// buffer done and empty. The cases with the refit off write no buffer.
#[test]
fn the_refit_is_written_part_full_and_done() {
    if regenerating() {
        return;
    }
    let buffers = |bytes: &[u8], variant: &str| -> (u64, Vec<(bool, u64, usize)>) {
        let v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let cap = at(&v, &["model", variant, "cfg", "refit_rows"])
            .and_then(rmpv::Value::as_u64)
            .unwrap_or_else(|| panic!("{variant}: cfg.refit_rows"));
        let list = at(&v, &["model", variant, "refit"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{variant}: refit"))
            .iter()
            .map(|b| {
                let done = at(b, &["done"]).and_then(rmpv::Value::as_bool).unwrap();
                let learned = at(b, &["learned"]).and_then(rmpv::Value::as_u64).unwrap();
                (done, learned, held(at(b, &["rows"]).unwrap()))
            })
            .collect();
        (cap, list)
    };
    for (name, variant) in [("huber_refit", "Robust"), ("sgd_huber_refit", "Sgd")] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let (cap, list) = buffers(&f.bytes(), variant);
        assert_eq!(list.len(), 2, "{name}: a buffer a target");
        for (done, learned, rows) in &list {
            assert!(
                !done && *learned > 0 && *learned < cap && *rows > 0,
                "{name}: {list:?}"
            );
        }
        let mut m = load(&case_of(f), &f.bytes()).unwrap();
        for row in f.rows() {
            m.step(&row);
        }
        let (_, after) = buffers(&named(&m.state()), variant);
        assert!(
            after.iter().all(|(done, _, rows)| *done && *rows == 0),
            "{name}: the continuation does not reach the refit: {after:?}"
        );
    }
    for (name, variant) in [
        ("huber_refit_done", "Robust"),
        ("sgd_huber_refit_done", "Sgd"),
    ] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let (cap, list) = buffers(&f.bytes(), variant);
        assert_eq!(list.len(), 2, "{name}: a buffer a target");
        for (done, learned, rows) in &list {
            assert!(*done && *learned == cap && *rows == 0, "{name}: {list:?}");
        }
    }
    for (name, variant) in [("huber", "Robust"), ("sgd_huber", "Sgd")] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        assert_eq!(buffers(&f.bytes(), variant), (0, vec![]), "{name}");
    }
}

/// `sgd`'s target-spread forms (schema 64, docs/PLAN.md task 278: D1, D2,
/// D8, D-S): `sgd_spread` keeps each target's spread under the squared loss,
/// its step gains and its level, every one holding a number; `sgd_unstable`
/// is saved with its gate withholding, so its continuation reads the gains
/// from the file and withholds every row; `sgd_spread_adagrad` keeps the
/// spread and the level and no gains, AdaGrad's step being bounded. The
/// cases without the flag write none of them.
#[test]
fn the_target_spread_forms_are_written() {
    if regenerating() {
        return;
    }
    let decoded = |name: &str| -> rmpv::Value {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap()
    };
    let field = |name: &str, key: &str| -> rmpv::Value {
        at(&decoded(name), &["model", "Sgd", key])
            .cloned()
            .unwrap_or_else(|| panic!("{name}: no {key}"))
    };
    let len = |v: &rmpv::Value| v.as_array().map_or(0, Vec::len);
    for name in ["sgd_spread", "sgd_unstable", "sgd_spread_adagrad"] {
        let on = at(&decoded(name), &["model", "Sgd", "cfg", "target_spread"])
            .and_then(rmpv::Value::as_bool);
        assert_eq!(on, Some(true), "{name}: cfg.target_spread");
        for key in ["spread", "level"] {
            assert_eq!(len(&field(name, key)), 2, "{name}: {key}, one a target");
            assert!(held(&field(name, key)) > 0, "{name}: {key} holds nothing");
        }
    }
    for name in ["sgd_spread", "sgd_unstable"] {
        assert_eq!(len(&field(name, "gain")), 2, "{name}: gains, one a target");
        assert!(
            held(&field(name, "gain")) > 0,
            "{name}: the gains hold nothing"
        );
    }
    assert_eq!(len(&field("sgd_spread_adagrad", "gain")), 0);
    for name in ["sgd", "sgd_huber", "sgd_logistic"] {
        for key in ["gain", "level"] {
            assert_eq!(len(&field(name, key)), 0, "{name}: {key} without the flag");
        }
    }
    // The gate: the stable case predicts after its save, the unstable one
    // withholds every row, read from the gains in the file.
    for (name, withheld) in [("sgd_spread", false), ("sgd_unstable", true)] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let mut m = load(&case_of(f), &f.bytes()).unwrap();
        let preds: Vec<f64> = f.rows().iter().flat_map(|r| m.step(r).pred).collect();
        assert_eq!(
            preds.iter().all(|p| p.is_nan()),
            withheld,
            "{name}: {preds:?}"
        );
    }
}

/// `kalman`'s covariance in units of the noise (schema 64, docs/PLAN.md task
/// 278): `kalman_noise_units` names the form in its cfg and is saved with
/// every covariance's intercept still diffuse -- its row and column 0 --
/// after rows that moved the moments and the clock since an observation,
/// and its continuation takes the diffuse step. `kalman` and `kalman_gap`
/// are the form before, which writes no `noise_units`, so a state saved
/// before it loads as that form.
#[test]
fn kalmans_noise_units_form_is_written_with_its_intercept_open() {
    if regenerating() {
        return;
    }
    let intercepts = |v: &rmpv::Value| -> Vec<f64> {
        at(v, &["model", "Kalman", "p"])
            .and_then(rmpv::Value::as_array)
            .expect("p")
            .iter()
            .map(|p| p.as_array().unwrap()[0].as_f64().unwrap())
            .collect()
    };
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "kalman_noise_units")
        .expect("kalman_noise_units: no fixture; add the case and regenerate");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let units = at(&v, &["model", "Kalman", "cfg", "noise_units"]).and_then(rmpv::Value::as_bool);
    assert_eq!(units, Some(true), "cfg.noise_units");
    let open = intercepts(&v);
    assert!(
        open.len() == 2 && open.iter().all(|p| *p == 0.0),
        "{open:?}"
    );
    let elapsed = at(&v, &["model", "Kalman", "elapsed"]).expect("elapsed");
    assert!(held(elapsed) > 0, "the clock since an observation moved");
    let mut m = load(&case_of(f), &f.bytes()).unwrap();
    for row in f.rows() {
        m.step(&row);
    }
    let after = rmpv::decode::read_value(&mut named(&m.state()).as_slice()).unwrap();
    let closed = intercepts(&after);
    assert!(
        closed.iter().all(|p| *p > 0.0),
        "the continuation takes the step: {closed:?}"
    );
    for name in ["kalman", "kalman_gap"] {
        let f = frozen::ALL.iter().find(|f| f.name == name).unwrap();
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        assert!(
            at(&v, &["model", "Kalman", "cfg", "noise_units"]).is_none(),
            "{name}"
        );
    }
}

/// `audit`'s counters in both of their forms (schema 54, docs/PLAN.md task
/// 223 (b)): a summary past the cap, with decrements, and exact counts below
/// it. Integers, which [`held`] does not count, so read here.
#[test]
fn an_audits_counters_are_written_in_both_forms() {
    if regenerating() {
        return;
    }
    for (name, over) in [("audit", true), ("audit_exact", false)] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let columns = at(&v, &["model", "Audit", "columns"])
            .and_then(rmpv::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: the state has no columns"));
        for c in columns {
            let counts = at(c, &["counts"]).unwrap();
            assert_eq!(
                at(counts, &["over"]).and_then(rmpv::Value::as_bool),
                Some(over),
                "{name}"
            );
            let held_counts = at(counts, &["counts"])
                .and_then(rmpv::Value::as_array)
                .map_or(0, Vec::len);
            assert!(held_counts > 0, "{name}: no counter is held");
            let decrements = at(counts, &["decrements"]).and_then(rmpv::Value::as_u64);
            assert_eq!(decrements.is_some_and(|d| d > 0), over, "{name}");
        }
    }
    // And its moments over a power of two above 1, for values past 2^65.
    let f = frozen::ALL
        .iter()
        .find(|f| f.name == "audit_scaled")
        .expect("audit_scaled: no fixture; add the case and regenerate");
    let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
    let columns = at(&v, &["model", "Audit", "columns"])
        .and_then(rmpv::Value::as_array)
        .expect("audit_scaled: the state has no columns");
    for c in columns {
        let scale = at(c, &["moments", "scale"]).and_then(rmpv::Value::as_f64);
        assert!(
            scale.is_some_and(|s| s > 1.0),
            "audit_scaled: scale {scale:?}"
        );
    }
}

/// `ew_cov`'s `bias` in both of its values (schema 61, docs/PLAN.md task
/// 244): `false`, the default, in `ew_cov`, whose statistics and components
/// the factor scales, and `true` in `ew_cov_window`. A bool, which [`held`]
/// does not count, so read here.
#[test]
fn ew_covs_bias_is_written_in_both_values() {
    if regenerating() {
        return;
    }
    for (name, bias) in [("ew_cov", false), ("ew_cov_window", true)] {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: no fixture; add the case and regenerate"));
        let v = rmpv::decode::read_value(&mut f.bytes().as_slice()).unwrap();
        let got = at(&v, &["model", "EwCovModel", "cfg", "bias"]).and_then(rmpv::Value::as_bool);
        assert_eq!(got, Some(bias), "{name}");
    }
}

// --- regeneration -----------------------------------------------------------

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/state_fixtures")
}

/// `bits` as a Rust array of hex literals, eight a line.
fn words(out: &mut String, name: &str, bits: &[u64]) {
    writeln!(out, "    {name}: &[").unwrap();
    for chunk in bits.chunks(4) {
        let line: Vec<String> = chunk.iter().map(|b| format!("{b:#018x}")).collect();
        writeln!(out, "        {},", line.join(", ")).unwrap();
    }
    writeln!(out, "    ],").unwrap();
}

fn file_of(case: &Case) -> String {
    let (mut m, rows) = saved(case);
    let bytes = named(&m.state());
    let after = &rows[case.before..];
    let outs: Vec<Out> = after.iter().map(|r| m.step(r)).collect();
    // The state saved before the continuation, loaded, goes on as the model
    // that never stopped, and saves its bytes again: a fixture that fails
    // its own checks is not written.
    let mut back = load(case, &bytes).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    assert!(
        named(&back.state()) == bytes,
        "{}: re-saves other bytes",
        case.name
    );
    for (r, want) in after.iter().zip(&outs) {
        same_out(case.name, &back.step(r), want, 0.0);
    }
    let n_pred = outs[0].pred.len();
    let n_extra = outs[0].extra.len();
    let n_targets = after[0].y.len();
    let s: State = rmp_serde::from_slice(&bytes).unwrap();
    let mut out = String::new();
    writeln!(
        out,
        "// @generated by `{REGENERATE}`: do not edit by hand.\n\
         // The `{}` case of tests/cases/mod.rs after {} rows, and its next {} rows.\n\
         pub const FIXTURE: crate::cases::Fixture = crate::cases::Fixture {{",
        case.name,
        case.before,
        after.len()
    )
    .unwrap();
    writeln!(out, "    name: {:?},", case.name).unwrap();
    writeln!(out, "    variant: {:?},", case.variant).unwrap();
    writeln!(out, "    schema: {},", s.schema_version).unwrap();
    writeln!(out, "    writer: {:?},", writer()).unwrap();
    writeln!(out, "    state: \"\\").unwrap();
    let h = hex(&bytes);
    for (i, chunk) in h.as_bytes().chunks(96).enumerate() {
        let tail = if (i + 1) * 96 >= h.len() { "\"," } else { "\\" };
        writeln!(out, "        {}{tail}", std::str::from_utf8(chunk).unwrap()).unwrap();
    }
    let x: Vec<u64> = after
        .iter()
        .flat_map(|r| r.x.iter().map(|v| v.to_bits()))
        .collect();
    words(&mut out, "x", &x);
    writeln!(out, "    y: &[").unwrap();
    for r in after.iter().filter(|r| !r.y.is_empty()) {
        let ys: Vec<String> =
            r.y.iter()
                .map(|v| match v {
                    Some(v) => format!("Some({:#018x})", v.to_bits()),
                    None => "None".into(),
                })
                .collect();
        writeln!(out, "        {},", ys.join(", ")).unwrap();
    }
    writeln!(out, "    ],").unwrap();
    let d: Vec<u64> = after.iter().map(|r| r.d.to_bits()).collect();
    let w: Vec<u64> = after.iter().map(|r| r.w.to_bits()).collect();
    words(&mut out, "d", &d);
    words(&mut out, "w", &w);
    let pred: Vec<u64> = outs
        .iter()
        .flat_map(|o| o.pred.iter().map(|v| v.to_bits()))
        .collect();
    let n_eff: Vec<u64> = outs.iter().map(|o| o.n_eff.to_bits()).collect();
    let extra: Vec<u64> = outs
        .iter()
        .flat_map(|o| o.extra.iter().map(|v| v.to_bits()))
        .collect();
    words(&mut out, "pred", &pred);
    words(&mut out, "n_eff", &n_eff);
    words(&mut out, "extra", &extra);
    writeln!(
        out,
        "    n_features: {},\n    n_targets: {n_targets},\n    n_pred: {n_pred},\n    n_extra: {n_extra},\n}};",
        case.n_features
    )
    .unwrap();
    out
}

/// Rewrites every fixture, and the index, when `PRINT_STATE_FIXTURES=1`.
#[test]
fn regenerate_when_asked() {
    if !regenerating() {
        return;
    }
    let dir = dir();
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rs") {
            std::fs::remove_file(path).unwrap();
        }
    }
    let mut index = format!(
        "// @generated by `{REGENERATE}`: do not edit by hand.\n\
         // The frozen state fixtures, one file a case of tests/cases/mod.rs.\n\n\
         /// The schema every fixture here was written at.\n\
         pub const SCHEMA: u32 = {SCHEMA_VERSION};\n\n"
    );
    let cases = all();
    for case in &cases {
        let text = file_of(case);
        std::fs::write(dir.join(format!("{}.rs", case.name)), &text).unwrap();
        writeln!(
            index,
            "pub mod {} {{\n    include!(\"{}.rs\");\n}}",
            case.name, case.name
        )
        .unwrap();
        println!("wrote {}.rs ({} bytes)", case.name, text.len());
    }
    writeln!(
        index,
        "\n/// Every fixture, in the cases' order.\npub const ALL: &[&crate::cases::Fixture] = &["
    )
    .unwrap();
    for case in &cases {
        writeln!(index, "    &{}::FIXTURE,", case.name).unwrap();
    }
    writeln!(index, "];").unwrap();
    std::fs::write(dir.join("index.rs"), index).unwrap();
}
