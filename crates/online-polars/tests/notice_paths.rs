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
//! A group or session span that closes in a call says the notices it
//! raised in that call, as one closed in a later call does, so which are
//! said, and in what order, does not depend on where a chunk ends (task
//! 333): under `group_close = "monotone"` the closed stream was removed
//! before the notices were taken, under `"session"` the span's stream was
//! replaced within the call, and either lost them.
//!
//! The notice used is the support notice: under a ridge of 0.5, a feature
//! duplicated reads a support below 0.5 from its first solve, and at a
//! half-life of 4 rows on a row clock the decay time before a group's row
//! `i` (0-based) is `i − 1`, so it is 95% settled from row 19, where the
//! notice is raised, once for the life of the stream.

use online_polars::{Bank, Spec};
use polars::prelude::*;
use proptest::prelude::*;

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

/// `df` with the session column `s` set to `name` on every row.
fn session(df: DataFrame, name: &str) -> DataFrame {
    let n = df.height();
    let mut df = df;
    df.with_column(Column::new("s".into(), vec![name; n]))
        .unwrap();
    df
}

/// The notices a bank of `specs` says over `df` fed as calls ending at
/// `cuts` (row positions, in order; the last call ends at the frame's end),
/// in the order they are taken, and the closed rows it queues.
fn said_over(specs: &[Spec], df: &DataFrame, cuts: &[usize]) -> (Vec<String>, usize) {
    let mut bank = Bank::new(specs.to_vec()).unwrap();
    let mut said = Vec::new();
    let mut from = 0;
    for &to in cuts.iter().chain(std::iter::once(&df.height())) {
        if to > from {
            bank.fit_predict(&df.slice(from as i64, to - from)).unwrap();
            said.extend(bank.take_notices());
        }
        from = from.max(to);
    }
    let closed = bank.closed_groups(None, true).unwrap().height();
    (said, closed)
}

/// Every way to cut `df` into two calls, and into calls of one row: each
/// says the notices one call says, in the same order, and queues the same
/// closed rows.
fn same_at_every_cut(specs: &[Spec], df: &DataFrame) -> Vec<String> {
    let (whole, closed) = said_over(specs, df, &[]);
    for cut in 1..df.height() {
        assert_eq!(
            said_over(specs, df, &[cut]),
            (whole.clone(), closed),
            "cut at row {cut}"
        );
    }
    let rows: Vec<usize> = (1..df.height()).collect();
    assert_eq!(
        said_over(specs, df, &rows),
        (whole.clone(), closed),
        "a row a call"
    );
    whole
}

/// Under `group_close = "monotone"`, a group that reaches its row 19 in the
/// call whose next group closes it says its notice there, as it does when
/// the call ends before the next group's first row.
#[test]
fn a_group_closed_in_the_call_says_its_notice_at_any_chunking() {
    let specs = [spec("a", r#", "group_close": "monotone""#)];
    let df = concat(&[rows(1, 0, RAISED_AT + 1), rows(2, 0, 3)]);
    let said = same_at_every_cut(&specs, &df);
    assert_eq!((said.len(), named(&said, "a", 1)), (1, 1), "{said:#?}");
    // Two groups closed in one call, each past its row 19, and a third that
    // closes nothing: both notices, in key order.
    let df = concat(&[
        rows(1, 0, RAISED_AT + 2),
        rows(2, 0, RAISED_AT + 1),
        rows(3, 0, 2),
    ]);
    let said = same_at_every_cut(&specs, &df);
    assert_eq!(said.len(), 2, "{said:#?}");
    assert!(said[0].starts_with("spec \"a\" group \"1\":"), "{said:#?}");
    assert!(said[1].starts_with("spec \"a\" group \"2\":"), "{said:#?}");
}

/// Under `group_close = "session"`, a span that reaches its row 19 in the
/// call that closes it says its notice there, before the next span's own:
/// each span is a fresh stream and says its notice once. Three spans in one
/// call carry the first's notice through the second's stream.
#[test]
fn a_session_span_closed_in_the_call_says_its_notice_at_any_chunking() {
    let specs = [spec("a", r#", "session": "s", "group_close": "session""#)];
    let df = concat(&[
        session(rows(1, 0, RAISED_AT + 1), "one"),
        session(rows(1, 0, 3), "two"),
    ]);
    let said = same_at_every_cut(&specs, &df);
    assert_eq!((said.len(), named(&said, "a", 1)), (1, 1), "{said:#?}");
    let df = concat(&[
        session(rows(1, 0, RAISED_AT + 1), "one"),
        session(rows(1, 0, 4), "two"),
        session(rows(1, 0, RAISED_AT + 2), "three"),
        session(rows(1, 0, 1), "four"),
    ]);
    let said = same_at_every_cut(&specs, &df);
    assert_eq!((said.len(), named(&said, "a", 1)), (2, 2), "{said:#?}");
}

/// The rows of a stream of groups, each group's rows contiguous and the
/// groups in ascending key order (so `"monotone"` closes each at the next
/// one's first row), each group cut into session spans: `groups[g]` holds
/// group `g`'s span lengths. A group's rows are numbered along the group,
/// so a monotone stream's row 19 is the group's.
fn stream_of(groups: &[Vec<usize>]) -> DataFrame {
    let mut parts = Vec::new();
    for (g, spans) in groups.iter().enumerate() {
        let mut from = 0;
        for (k, &len) in spans.iter().enumerate() {
            parts.push(session(rows(g as i64, from, len), &format!("{g}-{k}")));
            from += len;
        }
    }
    concat(&parts)
}

proptest! {
    /// Over random streams of groups and spans and random chunkings, a
    /// monotone spec and a session spec beside it say the same notices, in
    /// the same order within each spec, as one call does, and the count is
    /// the streams' that reach row 19: each group's under `"monotone"`, each
    /// span's under `"session"`.
    #[test]
    fn notices_are_the_same_at_any_chunking(
        groups in prop::collection::vec(prop::collection::vec(1usize..=26, 1..=3), 1..=4),
        cuts in prop::collection::vec(0usize..1000, 0..=6),
    ) {
        let specs = [
            spec("mono", r#", "group_close": "monotone""#),
            spec("sess", r#", "session": "s", "group_close": "session""#),
        ];
        let df = stream_of(&groups);
        let n = df.height();
        let mut cuts: Vec<usize> = cuts.iter().map(|c| c * n / 1000).collect();
        cuts.sort_unstable();
        let (whole, closed) = said_over(&specs, &df, &[]);
        let (chunked, closed_chunked) = said_over(&specs, &df, &cuts);
        let of = |said: &[String], s: &str| -> Vec<String> {
            let tag = format!("spec {s:?} ");
            said.iter().filter(|n| n.starts_with(&tag)).cloned().collect()
        };
        prop_assert_eq!(of(&chunked, "mono"), of(&whole, "mono"));
        prop_assert_eq!(of(&chunked, "sess"), of(&whole, "sess"));
        prop_assert_eq!(chunked.len(), whole.len());
        prop_assert_eq!(closed_chunked, closed);
        let groups_past = groups
            .iter()
            .filter(|spans| spans.iter().sum::<usize>() > RAISED_AT)
            .count();
        let spans_past = groups.iter().flatten().filter(|&&l| l > RAISED_AT).count();
        prop_assert_eq!(of(&whole, "mono").len(), groups_past, "{:#?}", whole);
        prop_assert_eq!(of(&whole, "sess").len(), spans_past, "{:#?}", whole);
    }
}
