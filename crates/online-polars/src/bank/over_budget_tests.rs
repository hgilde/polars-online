//! The refusal past a `window_budget` says how to raise it, spelled as a
//! spec writes it. A file of its own, keeping bank.rs under
//! `tests/test_repo_hygiene.py`'s 250 KB cap for a source file.
use super::{GroupKey, over_budget_message};
use crate::Spec;
use online_core::Cadence;

fn spec(budget: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ew_ridge", "window_size": 50{budget}}},
                "targets": ["y"], "features": ["x"], "half_life": 20}}"#
    ))
    .unwrap()
}

/// A temporal spec with a clock spacing written `"90s"`.
fn temporal(every: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ew_ridge", "window_size": "30m"{every}}},
                "targets": ["y"], "features": ["x"], "half_life": "10m",
                "clock": "ts", "gap_cap": "10m"}}"#
    ))
    .unwrap()
}

fn clock(spacing: f64) -> Cadence {
    Cadence {
        spacing,
        rows: usize::MAX,
    }
}

/// The refusal says how to raise the budget, spelled as a spec writes
/// it, and under the default says it was the default (the user,
/// 2026-09-28).
#[test]
fn the_refusal_says_how_to_raise_the_budget() {
    let key = GroupKey(Some("a".into()));
    let default = over_budget_message(&spec(""), &key, 300 << 20, clock(4.0));
    for want in [
        "the default window_budget of 256 MiB",
        "window_budget = {\"refuse\": MiB} with a larger number of MiB",
        "{\"refuse\": inf} for no bound",
        "float(\"inf\")",
        "raise window_every (4 now)",
        "window_budget = {\"thin\": MiB}",
        "group \"a\"",
        "300.000 MiB",
    ] {
        assert!(default.contains(want), "{want:?} not in {default}");
    }
    let named = over_budget_message(
        &spec(r#", "window_budget": {"refuse": 8}"#),
        &GroupKey(None),
        9 << 20,
        Cadence::EVERY_ROW,
    );
    assert!(
        named.contains("past window_budget = {\"refuse\": 8}"),
        "{named}"
    );
    assert!(!named.contains("default"), "{named}");
}

/// The refusal names the cadence the ring ran at, whichever is in force
/// -- the clock spacing, the row cap, both, or every row -- and the clock
/// spacing as the spec wrote it, or in its form where thinning moved it
/// (docs/PLAN.md task 162).
#[test]
fn the_refusal_names_the_cadence_in_force() {
    let key = GroupKey(None);
    let say = |s: &Spec, c: Cadence| over_budget_message(s, &key, 9 << 20, c);
    let both = Cadence {
        spacing: 90.0,
        rows: 25,
    };
    let rows = Cadence {
        spacing: f64::INFINITY,
        rows: 25,
    };
    for (s, cadence, want) in [
        (
            temporal(r#", "window_every": "90s""#),
            clock(90.0),
            "raise window_every (90s now);",
        ),
        (
            temporal(r#", "window_every": "90s""#),
            clock(180.0),
            "raise window_every (3m now);",
        ),
        (
            temporal(r#", "window_every": "90s", "max_rows_between_snapshots": 25"#),
            both,
            "raise window_every (90s now) and max_rows_between_snapshots (25 now)",
        ),
        (
            spec(r#", "max_rows_between_snapshots": 25"#),
            rows,
            "raise max_rows_between_snapshots (25 now), or give window_every",
        ),
        (
            spec(""),
            Cadence::EVERY_ROW,
            "set window_every to the clock between snapshots, or max_rows_between_snapshots \
             to the rows between them (every row now)",
        ),
        (
            spec(r#", "window_every": 2.5"#),
            clock(2.5),
            "raise window_every (2.5 now)",
        ),
    ] {
        let got = say(&s, cadence);
        assert!(got.contains(want), "{want:?} not in {got}");
    }
}
