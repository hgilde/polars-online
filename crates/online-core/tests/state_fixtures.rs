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
//! `tests/cases/mod.rs`, and a run without the variable checks them. From
//! 1.0 the minimum stays at 1.0's schema: a layout change keeps the
//! fixtures of the schema before it (moved to `state_fixtures/v<N>/` and
//! listed in [`PREVIOUS`]), whose states must load through the loader the
//! change ships and go on as they did, and regenerates the current set. A
//! change that moves a number moves a continuation too: regenerate after
//! confirming the move is intended, as for `tests/golden.rs`.
//!
//! The variant names are the files' tags (the named encoding writes
//! `{"EwRidge": {...}}`), so they are frozen here too: a renamed variant
//! breaks every saved file.

mod cases;

#[path = "state_fixtures/index.rs"]
mod frozen;

use std::fmt::Write as _;
use std::path::PathBuf;

use cases::{Case, Fixture, Out, agree, all, hex, load, named, saved, writer};
use online_core::{MIN_SCHEMA_VERSION, ModelState, SCHEMA_VERSION, State};

/// The schemas before the current one whose fixtures are kept, each with a
/// loader of its own: none before 1.0, where a layout change regenerates
/// the set instead.
const PREVIOUS: &[u32] = &[];

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

/// Every schema this build claims to load has its fixtures, and every
/// fixture is of a schema it loads: so `SCHEMA_VERSION` cannot move without
/// a set of its own, and `MIN_SCHEMA_VERSION` cannot sit below the oldest
/// set kept.
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
         as the previous schema's (PREVIOUS), write its loader, then regenerate",
        frozen::SCHEMA
    );
    let covered: Vec<u32> = PREVIOUS.iter().copied().chain([frozen::SCHEMA]).collect();
    let claimed: Vec<u32> = (MIN_SCHEMA_VERSION..=SCHEMA_VERSION).collect();
    assert_eq!(
        claimed, covered,
        "this build loads schemas {MIN_SCHEMA_VERSION}..={SCHEMA_VERSION} and fixtures cover \
         {covered:?}: each schema it loads needs a frozen set, held to its loader"
    );
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
