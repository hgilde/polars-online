//! Which streams can hold a readiness notice not yet taken when a call
//! ends, and that each one's notice is said once (task 332).
//!
//! A stream's notices are raised only as it learns (`Stream::process_chunk`,
//! called for the chunk's groups alone), and `Bank::fit_predict` takes them
//! at the end of the call that raised them, so a call need walk only the
//! groups in its chunk. These tests hold each path a notice can take: a
//! group raising its notice among many it has seen, in a later call than
//! the bank's first; a group restored from a saved state, whose notices
//! were already said or are still to come; a group whose session closes
//! within the call; and a chunk refused before any row is learned.
//!
//! The notice used is the support notice: under a ridge of 0.5, a feature
//! duplicated reads a support below 0.5 from its first solve, and at a
//! half-life of 4 rows on a row clock the decay time before a group's row
//! `i` (0-based) is `i − 1`, so it is 95% settled from row 19, where the
//! notice is raised, once for the life of the stream.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// The row a group's notice is raised on (0-based): its 20th.
const RAISED_AT: usize = 19;

fn spec(name: &str, extra: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "{name}", "model": {{"type": "ewridge", "ridge": 0.5}},
            "targets": ["y"], "features": ["x0", "x1", "x2"],
            "half_life": 4.0, "coef_every": 1, "group": "g"{extra}}}"#
    ))
    .unwrap()
}

/// Rows `from..from + n` of group `g`'s stream.
fn rows(g: i64, from: usize, n: usize) -> DataFrame {
    let x0: Vec<f64> = (from..from + n)
        .map(|i| ((i * 7) % 11) as f64 - 5.0)
        .collect();
    let x1: Vec<f64> = (from..from + n)
        .map(|i| ((i * 5) % 9) as f64 - 4.0)
        .collect();
    let y: Vec<f64> = (from..from + n)
        .map(|i| x0[i - from] - x1[i - from] + ((i * 3) % 7) as f64 * 0.1)
        .collect();
    df!("g" => vec![g; n], "x0" => &x0, "x1" => &x1, "x2" => &x0, "y" => &y).unwrap()
}

fn concat(parts: &[DataFrame]) -> DataFrame {
    let mut out = parts[0].clone();
    for p in &parts[1..] {
        out.vstack_mut(p).unwrap();
    }
    out
}

fn named(notices: &[String], spec: &str, g: i64) -> usize {
    let tag = format!("spec {spec:?} group \"{g}\":");
    notices.iter().filter(|n| n.starts_with(&tag)).count()
}

/// Among 300 groups seen, a call to a few of them says the notices of
/// those that reach row 19 in it, once each, and of no other; and in a
/// fixed order: by spec, then by the group's first row in the chunk.
#[test]
fn a_call_says_the_notices_of_its_own_groups_once_in_chunk_order() {
    let mut bank = Bank::new(vec![spec("a", ""), spec("b", "")]).unwrap();
    // Every group 10 rows in: no notice yet.
    let warm: Vec<DataFrame> = (0..300).map(|g| rows(g, 0, 10)).collect();
    bank.fit_predict(&concat(&warm)).unwrap();
    assert!(bank.take_notices().is_empty());
    // Groups 7, 250 and 3 to row 19, in that order of first rows, and
    // group 5 to row 14.
    let call = concat(&[
        rows(7, 10, 4),
        rows(250, 10, 10),
        rows(5, 10, 5),
        rows(3, 10, 10),
        rows(7, 14, 6),
    ]);
    bank.fit_predict(&call).unwrap();
    let said = bank.take_notices();
    let expect: Vec<(&str, i64)> = vec![
        ("a", 7),
        ("a", 250),
        ("a", 3),
        ("b", 7),
        ("b", 250),
        ("b", 3),
    ];
    assert_eq!(said.len(), expect.len(), "{said:#?}");
    for (n, (s, g)) in said.iter().zip(&expect) {
        assert!(
            n.starts_with(&format!("spec {s:?} group \"{g}\":"))
                && n.contains("support_coef < 0.5"),
            "{said:#?}"
        );
    }
    // Once for the life of the stream: their next rows say nothing, and
    // group 5's row 19 says its own.
    bank.fit_predict(&concat(&[rows(7, 20, 3), rows(5, 15, 5)]))
        .unwrap();
    let said = bank.take_notices();
    assert_eq!(said.len(), 2, "{said:#?}");
    assert_eq!(
        (named(&said, "a", 5), named(&said, "b", 5)),
        (1, 1),
        "{said:#?}"
    );
}

/// A group restored from a saved state: one saved before its notice is
/// raised says it once on the row that raises it, and one saved after says
/// nothing again (the stream keeps which notices it has said).
#[test]
fn a_restored_group_says_its_notice_once() {
    let mut bank = Bank::new(vec![spec("a", "")]).unwrap();
    bank.fit_predict(&concat(&[rows(1, 0, RAISED_AT), rows(2, 0, RAISED_AT + 1)]))
        .unwrap();
    let said = bank.take_notices();
    assert_eq!((said.len(), named(&said, "a", 2)), (1, 1), "{said:#?}");
    let bytes = bank.save_bytes().unwrap();
    let mut back = Bank::load_bytes(&bytes, None).unwrap();
    assert!(back.take_notices().is_empty(), "a load says nothing");
    back.fit_predict(&concat(&[rows(2, RAISED_AT + 1, 3), rows(1, RAISED_AT, 3)]))
        .unwrap();
    let said = back.take_notices();
    assert_eq!((said.len(), named(&said, "a", 1)), (1, 1), "{said:#?}");
}

/// A chunk refused before any row is learned -- a group's clock stepping
/// back -- raises nothing, and the corrected chunk says the notice once.
#[test]
fn a_refused_chunk_then_its_correction_says_the_notice_once() {
    let clocked = |extra: &str| spec("a", &format!(r#", "clock": "t", "gap_cap": 100.0{extra}"#));
    let mut bank = Bank::new(vec![clocked("")]).unwrap();
    let with_clock = |df: DataFrame, from: usize| {
        let n = df.height();
        let mut df = df;
        df.with_column(Column::new(
            "t".into(),
            (from..from + n).map(|i| i as f64).collect::<Vec<_>>(),
        ))
        .unwrap();
        df
    };
    bank.fit_predict(&with_clock(rows(1, 0, RAISED_AT), 0))
        .unwrap();
    assert!(bank.take_notices().is_empty());
    // Group 1's crossing row with a clock behind its last, after a new group.
    let bad = concat(&[
        with_clock(rows(9, 0, 2), 0),
        with_clock(rows(1, RAISED_AT, 2), 0),
    ]);
    assert!(bank.fit_predict(&bad).is_err());
    assert!(bank.take_notices().is_empty());
    let good = concat(&[
        with_clock(rows(9, 0, 2), 0),
        with_clock(rows(1, RAISED_AT, 2), RAISED_AT),
    ]);
    bank.fit_predict(&good).unwrap();
    let said = bank.take_notices();
    assert_eq!((said.len(), named(&said, "a", 1)), (1, 1), "{said:#?}");
}

/// A group whose session closes within the call: the span that goes on
/// after the close is a fresh stream, which says its own notice on its own
/// row 19, in the call that reaches it.
#[test]
fn a_session_closed_in_the_call_says_the_new_spans_notice() {
    let mut bank = Bank::new(vec![spec(
        "a",
        r#", "session": "s", "group_close": "session""#,
    )])
    .unwrap();
    let session = |df: DataFrame, s: &str| {
        let n = df.height();
        let mut df = df;
        df.with_column(Column::new("s".into(), vec![s; n])).unwrap();
        df
    };
    // A first span of 5 rows, closed by the second, which runs to its row 19.
    bank.fit_predict(&concat(&[
        session(rows(1, 0, 5), "one"),
        session(rows(1, 0, RAISED_AT + 1), "two"),
    ]))
    .unwrap();
    let said = bank.take_notices();
    assert_eq!((said.len(), named(&said, "a", 1)), (1, 1), "{said:#?}");
    assert_eq!(bank.closed_groups(None, true).unwrap().height(), 1);
}
