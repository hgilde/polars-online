//! The refusal past a `window_budget` says how to raise it, spelled as a
//! spec writes it. A file of its own, keeping bank.rs under
//! `tests/test_repo_hygiene.py`'s 250 KB cap for a source file.
use super::{GroupKey, over_budget_message};
use crate::Spec;

fn spec(budget: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ew_ridge", "window_size": 50{budget}}},
                "targets": ["y"], "features": ["x"], "half_life": 20}}"#
    ))
    .unwrap()
}

/// The refusal says how to raise the budget, spelled as a spec writes
/// it, and under the default says it was the default (the user,
/// 2026-09-28).
#[test]
fn the_refusal_says_how_to_raise_the_budget() {
    let key = GroupKey(Some("a".into()));
    let default = over_budget_message(&spec(""), &key, 300 << 20, 4);
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
        1,
    );
    assert!(
        named.contains("past window_budget = {\"refuse\": 8}"),
        "{named}"
    );
    assert!(!named.contains("default"), "{named}");
}
