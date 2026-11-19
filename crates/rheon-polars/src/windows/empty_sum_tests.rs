//! A window with no row with a value, looking back and looking ahead: a sum
//! asking no minimum (`min_samples` 0, Polars' default for its
//! `rolling_sum_by`) adds up nothing, 0.0, and with 1 it is null as before; a
//! window still open when the input ends stays null either way.
use rheon_core::{ClockCfg, OnClockReset};

use super::*;

/// The sum's output at each row of the stream `(stamp, value)`, NaN for null.
fn sums(direction: Direction, closed: Closed, min_samples: u32, rows: &[(f64, f64)]) -> Vec<f64> {
    let kernel = KernelDef {
        direction,
        half_life: f64::INFINITY,
        window_size: Some(2.0),
        window_ns: Some(2_000_000_000),
        closed,
    };
    let op = OpDef {
        kernel: 0,
        stat: Stat::Sum,
        input: 0,
        min_samples,
        partial: Partial::Keep,
        bias: false,
        quantile: None,
    };
    let cfg = ClockCfg {
        gap_cap: 100.0,
        on_clock_reset: OnClockReset::Error,
        session_gap: None,
        min_backwards_jump: 0.0,
    };
    let mut core = Windows::new(vec![kernel], vec![op], cfg).unwrap();
    let mut out = Vec::new();
    for &(t, x) in rows {
        let group = core.group(None);
        core.push(&RowIn {
            group,
            clock: Some(ClockValue::F64(t)),
            session: None,
            values: &[x],
            accept: true,
        })
        .unwrap();
        out.extend(core.drain().values.remove(0));
    }
    out.extend(core.finish().values.remove(0));
    out
}

#[test]
fn an_empty_windows_sum_is_zero_with_no_minimum() {
    let nan = f64::NAN;
    let rows: Vec<(f64, f64)> = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 30.0]
        .into_iter()
        .zip([1.0, nan, nan, nan, 5.0, nan, nan])
        .collect();
    // Looking back over `[t - 2, t)`: row 0 has no row before it, rows 3 and
    // 4 see only nulls, row 6 sees nothing. Looking ahead over `(t, t + 2]`:
    // rows 0 and 1 see only nulls, row 2 none, rows 4 and 5 only nulls, and
    // row 6's window is still open.
    for (direction, closed, none, one) in [
        (
            Direction::Backward,
            Closed::Left,
            [0.0, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0],
            [nan, 1.0, 1.0, nan, nan, 5.0, nan],
        ),
        (
            Direction::Forward,
            Closed::Right,
            [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, nan],
            [nan, nan, nan, 5.0, nan, nan, nan],
        ),
    ] {
        for (min_samples, want) in [(0, none), (1, one)] {
            let got = sums(direction, closed, min_samples, &rows);
            assert!(
                got.len() == want.len()
                    && got
                        .iter()
                        .zip(want)
                        .all(|(g, w)| g.to_bits() == w.to_bits()),
                "{direction:?} {closed:?} min_samples {min_samples}: {got:?} vs {want:?}"
            );
        }
    }
}
