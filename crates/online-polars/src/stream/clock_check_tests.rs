//! [`Stream::check_clock`] against the check it replaced (task 250): the
//! whole clock step, [`ClockState::advance`], run on a copy of the stream's
//! clock over every row of the chunk, its first refusal turned into the
//! [`ClockRefusal`] the bank names. The check now moves only the previous
//! clock and session ([`ClockState::refusal`]); each case here holds the
//! two to the same refusal -- the row, the step's bits, the late-row
//! minimum, the step back in nanoseconds or in the column's units -- or to
//! none, over float, temporal and integer clocks at the ages and sizes
//! where only an exact read gets them right, sessions, both policies, and a
//! clock saved and loaded mid-stream.

use online_core::{ClockCfg, ClockState, ClockValue, OnClockReset, SessionGap};

use super::{ClockRefusal, Stream};
use crate::arrow::ClockCol;
use crate::spec::Spec;

const DECADE_NS: i64 = 315_576_000 * 1_000_000_000;
/// 2024-01-01T00:00:00Z, in nanoseconds since the Unix epoch.
const T0_NS: i64 = 1_704_067_200 * 1_000_000_000;

fn stream() -> Stream {
    let spec: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "ewridge"}, "features": ["x"],
            "targets": ["y"], "clock": "t", "half_life": 10, "gap_cap": 100}"#,
    )
    .unwrap();
    Stream::new(&spec).unwrap()
}

/// The check as it was before task 250, verbatim but for its result: the
/// whole step on a copy of the clock, row by row.
fn advanced(
    clock_state: &ClockState,
    cfg: &ClockCfg,
    clock: &ClockCol,
    session: Option<&[u64]>,
    rows: &[usize],
    base: usize,
) -> Result<(), ClockRefusal> {
    let mut state = clock_state.clone();
    for (ri, &row) in rows.iter().enumerate() {
        let at = base + ri;
        let prev = state.last_clock();
        let adv = state.advance(cfg, Some(clock.at(at)), session.map(|s| s[at]), true);
        if let Some(raw) = adv.backwards {
            return Err(ClockRefusal::new(
                raw,
                row,
                adv.disorder,
                Some(clock.at(at)),
                prev,
            ));
        }
    }
    Ok(())
}

/// Both checks over one chunk: the same refusal, by its `Debug` text (every
/// double to the bit), or none. Whether it refused.
fn both(
    s: &Stream,
    cfg: &ClockCfg,
    clock: &ClockCol,
    session: Option<&[u64]>,
    rows: &[usize],
    base: usize,
) -> bool {
    let got = s.check_clock(cfg, Some(clock), session, rows, base);
    let want = advanced(&s.persisted.clock, cfg, clock, session, rows, base);
    assert_eq!(
        format!("{got:?}"),
        format!("{want:?}"),
        "{clock:?} {session:?} under {cfg:?}"
    );
    got.is_err()
}

fn policies() -> Vec<ClockCfg> {
    let mut out = Vec::new();
    for (on_clock_reset, min_backwards_jump) in [
        (OnClockReset::Error, 0.0),
        (OnClockReset::ResetState, 1e-9),
        (OnClockReset::ResetState, 0.5),
        (OnClockReset::ResetState, 9_007_199_254_740_992.0),
    ] {
        for session_gap in [None, Some(SessionGap::Gap(2.0)), Some(SessionGap::Reset)] {
            out.push(ClockCfg {
                gap_cap: 100.0,
                on_clock_reset,
                session_gap,
                min_backwards_jump,
            });
        }
    }
    out
}

/// A seeded sweep of chunks in each clock form, stepping forward, standing
/// still and stepping back by small and large amounts, with session changes,
/// starting from a fresh clock and from one saved and loaded mid-stream: the
/// check refuses what the whole step refused, at the same row with the same
/// numbers, and both outcomes occur often.
#[test]
fn the_check_refuses_what_the_whole_step_refused() {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let (mut refused, mut passed) = (0, 0);
    for case in 0..600 {
        let n = 2 + (next() % 40) as usize;
        let base = (next() % 3) as usize;
        let mut v: i64 = match case % 3 {
            0 => 0,
            1 => i64::MAX - 1_000_000,
            _ => T0_NS + DECADE_NS,
        };
        let mut steps = Vec::with_capacity(n + base);
        for _ in 0..n + base {
            let r = next();
            v = v.saturating_add(match r % 9 {
                0 => -1,
                1 => -(1 << 20),
                2 => 0,
                _ => (r >> 40) as i64 % 1000,
            });
            steps.push(v);
        }
        let clock = match case % 3 {
            0 => ClockCol::F64(steps.iter().map(|&i| i as f64 / 7.0).collect()),
            1 => ClockCol::I64(steps.clone()),
            _ => ClockCol::Ns(steps.clone()),
        };
        let sessions: Vec<u64> = (0..n + base).map(|_| u64::from(next() % 13 == 0)).collect();
        let rows: Vec<usize> = (0..n).map(|i| 1000 + 3 * i).collect();
        let mut s = stream();
        if case % 2 == 1 {
            // Mid-stream: the clock has seen a row before this chunk, and
            // has been through a save and a load.
            let cfg = policies()[0];
            s.persisted
                .clock
                .advance(&cfg, Some(clock.at(base)), Some(sessions[base]), true);
            let bytes = rmp_serde::to_vec_named(&s.persisted.clock).unwrap();
            s.persisted.clock = rmp_serde::from_slice(&bytes).unwrap();
        }
        for cfg in policies() {
            for session in [None, Some(&sessions[..])] {
                if both(&s, &cfg, &clock, session, &rows, base) {
                    refused += 1;
                } else {
                    passed += 1;
                }
            }
        }
    }
    assert!(refused > 2_000 && passed > 2_000, "{refused} {passed}");
}

/// The cases only an exact read gets right, each refused at the row the
/// whole step refused, with its numbers: a step back of one nanosecond a
/// decade into a temporal clock (a double of seconds resolves about 60 ns
/// there), and a step back of `2^53 + 1` on an integer clock at `i64::MAX`,
/// past a minimum of `2^53` (a restart, no refusal) though its double ties
/// the minimum, where `2^53` is refused.
#[test]
fn the_exact_cases_are_refused_as_the_whole_step_refused_them() {
    let s = stream();
    let error = ClockCfg {
        gap_cap: 100.0,
        ..ClockCfg::default()
    };
    let late = |min: f64| ClockCfg {
        on_clock_reset: OnClockReset::ResetState,
        min_backwards_jump: min,
        ..error
    };
    let t = T0_NS + DECADE_NS;
    let ns = ClockCol::Ns(vec![t, t + 1, t + 2, t + 1, t + 3]);
    for cfg in [error, late(1e-9)] {
        assert!(both(&s, &cfg, &ns, None, &[0, 1, 2, 3, 4], 0));
        let err = s.check_clock(&cfg, Some(&ns), None, &[0, 1, 2, 3, 4], 0);
        let r = err.unwrap_err();
        assert_eq!((r.row, r.back_ns, r.raw), (3, Some(1), -1e-9));
    }
    assert!(!both(&s, &late(0.0), &ns, None, &[0, 1, 2, 3, 4], 0));

    let two53 = 1i64 << 53;
    for (back, refused) in [(two53, true), (two53 + 1, false)] {
        let int = ClockCol::I64(vec![i64::MAX - two53 - 10, i64::MAX, i64::MAX - back]);
        assert_eq!(
            both(&s, &late(two53 as f64), &int, None, &[0, 1, 2], 0),
            refused
        );
        assert!(both(&s, &error, &int, None, &[0, 1, 2], 0));
        let r = s
            .check_clock(&error, Some(&int), None, &[0, 1, 2], 0)
            .unwrap_err();
        assert_eq!((r.row, r.back_int), (2, Some(i128::from(back))));
    }

    // A float clock's NaN row is no step back; the row after it is
    // measured from the NaN, and is none either.
    let nan = ClockCol::F64(vec![5.0, f64::NAN, 4.0, 3.0]);
    assert!(both(&s, &error, &nan, None, &[0, 1, 2, 3], 0));
    let r = s
        .check_clock(&error, Some(&nan), None, &[0, 1, 2, 3], 0)
        .unwrap_err();
    assert_eq!((r.row, r.raw), (3, -1.0));

    // A step back across a session change is no refusal under any policy.
    let across = ClockCol::F64(vec![5.0, 4.0]);
    for cfg in policies() {
        assert!(!both(&s, &cfg, &across, Some(&[0, 1]), &[0, 1], 0));
    }

    // A first row from a clock loaded mid-stream is measured from the row
    // before the save.
    let mut loaded = stream();
    loaded
        .persisted
        .clock
        .advance(&error, Some(ClockValue::F64(10.0)), None, true);
    let bytes = rmp_serde::to_vec_named(&loaded.persisted.clock).unwrap();
    loaded.persisted.clock = rmp_serde::from_slice(&bytes).unwrap();
    let r = loaded
        .check_clock(&error, Some(&ClockCol::F64(vec![9.5])), None, &[7], 0)
        .unwrap_err();
    assert_eq!((r.row, r.raw), (7, -0.5));
    assert!(both(
        &loaded,
        &error,
        &ClockCol::F64(vec![9.5]),
        None,
        &[7],
        0
    ));
}
