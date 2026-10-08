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
/// the steps from 1.0).
macro_rules! previous {
    ($($schema:literal),* $(,)?) => {
        const PREVIOUS: &[u32] = &[$($schema),*];

        fn previous_sets() -> Vec<(u32, &'static [&'static Fixture])> {
            vec![$({
                mod set {
                    include!(concat!("state_fixtures/v", $schema, "/index.rs"));
                }
                ($schema, set::ALL)
            }),*]
        }
    };
}

previous![];

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
            assert!(
                named(&m.state()) == current.bytes(),
                "{what}: loaded, the state does not save the current schema's fixture bytes of \
                 the same case: the loader converts it to another state than this build reaches \
                 from the same rows"
            );
            let tol = if f.writer == writer() { 0.0 } else { TOL };
            for (i, (row, want)) in f.rows().iter().zip(f.outs()).enumerate() {
                same_out(
                    &format!("{what}, row {i} after the save"),
                    &m.step(row),
                    &want,
                    tol,
                );
            }
        }
    }
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
    const FROZEN: [&str; 21] = [
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

/// A layout a schema bump moved is written by a fixture that holds it, so
/// a change to its tags or fields cannot pass the harness unseen (review
/// round 5, D3): `sgd`'s per-loss state -- the residual scale under the
/// Huber loss (schema 40), the target's spread under the epsilon-insensitive
/// one (44) -- `ewridge`'s kept systems (38, under `set_keep_factor`) and
/// `bocpd`'s warm-up rows (40). Each was empty in every fixture before.
#[test]
fn every_layout_a_schema_moved_is_written_by_a_fixture() {
    if regenerating() {
        return;
    }
    const FORMS: [(&str, &[&str]); 5] = [
        ("sgd_huber", &["model", "Sgd", "sig2"]),
        ("sgd_huber", &["model", "Sgd", "wsig"]),
        ("sgd_eps", &["model", "Sgd", "spread"]),
        (
            "ewridge_leverage",
            &["model", "EwRidge", "ready", "systems"],
        ),
        ("bocpd_warming", &["model", "Bocpd", "warm"]),
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
        let held = field
            .as_array()
            .map_or(0, |a| a.iter().filter(|e| !e.is_nil()).count());
        assert!(
            held > 0,
            "{name}: {} is empty, so the fixture holds nothing of the layout it is there for",
            path.join(".")
        );
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
