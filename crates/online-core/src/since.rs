//! Where a model's own schedule stands on the decayed clock, held exactly
//! (docs/PLAN.md task 180).
//!
//! Three cadences are counted on the clock a model is stepped on: the
//! regressions' `solve_every` (`ewridge`, `lasso`, `huber`, `quantile`),
//! `ew_cov`'s `pca_every` and `micro`'s `prune_every`. Each summed the
//! `d_clock`s since its last event in doubles, and the rounded steps
//! drifted: two thousand steps of 1 ms sum to 1.9999999999998905 s, so a
//! cadence of `"2s"` fired a row late, every time (CB1's class of bug, tasks
//! 175 and 176). A [`Since`] keeps the stamp of the last event instead
//! ([`Stamp`], the decayed clock held exactly) and decides the next one by
//! the stamps' exact difference ([`Stamp::cmp_span`]), as a window spaces
//! its snapshots (`window_every`, task 175) and the stream its `coef` rows
//! (`coef_every`, task 178). On a temporal clock that is integer
//! nanoseconds; on a number clock, two rows with no capped gap or session
//! step between them differ by one subtraction of their raw values.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::{Stamp, ns_of_seconds, seconds_of_ns};

/// How far the decayed clock has moved since a model's last scheduled event
/// -- a solve, a refresh of the components, a checkpoint -- with the stamps
/// that measure it exactly (the module docs).
///
/// A caller that stamps its rows hands each one's stamp before the step
/// ([`Self::stamp_next`]): the stream does, through
/// `crate::OnlineModel::stamp_next`. A row stepped with no stamp is
/// measured as every cadence was before task 180, by the clock summed from
/// the `d_clock`s since the last event, so a direct caller of the core is
/// unchanged to the bit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Since {
    /// The stamp the clock is measured from: the row of the last event, or,
    /// before the first, the model's start -- its first stamped row's stamp
    /// less that row's own step ([`back_by`]), which in a stream is the
    /// row's stamp whenever the stream's first row is learned, its step
    /// being 0. `None` until a stamped row.
    from: Option<Stamp>,
    /// The stamp of the row last stepped, `None` where it had none: what an
    /// event at that row starts the clock from, and an event between two
    /// rows too (`ewridge`'s solve after a blend).
    last: Option<Stamp>,
    /// The clock summed from the `d_clock`s since the last event: what a row
    /// stepped with no stamp is measured by.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    summed: f64,
    /// The next row's stamp, which [`Self::step`] takes. Never state:
    /// nothing is pending between two rows.
    #[serde(skip)]
    next: Option<Stamp>,
}

impl Since {
    /// The next row's place on the decayed clock, held exactly, for
    /// [`Self::step`] to take.
    pub(crate) fn stamp_next(&mut self, stamp: Stamp) {
        self.next = Some(stamp);
    }

    /// One row stepped, `d_clock` after the last, at the stamp
    /// [`Self::stamp_next`] handed, if any.
    pub(crate) fn step(&mut self, d_clock: f64) {
        self.summed += d_clock;
        self.last = self.next.take();
        if let (None, Some(now)) = (self.from, self.last) {
            self.from = Some(back_by(now, self.summed));
        }
    }

    /// Whether `every` clock units have passed since the last event, at the
    /// row last stepped, inclusive: by the stamps' exact difference for a
    /// stamped row ([`Stamp::cmp_span`], which reads a comparison with no
    /// answer as equal, as `window_every` and `coef_every` do), and by the
    /// summed clock for one with none.
    pub(crate) fn reached(&self, every: f64) -> bool {
        match (self.last, self.from) {
            (Some(now), Some(from)) => now.cmp_span(from, every) != Ordering::Less,
            _ => self.summed >= every,
        }
    }

    /// The event happened at the row last stepped, or, between two rows,
    /// where that row left the clock: the clock is measured from there on.
    pub(crate) fn restart(&mut self) {
        self.from = self.last;
        self.summed = 0.0;
    }

    /// The clock summed since the last event, for a test stepping a model
    /// with no stamps.
    #[cfg(test)]
    pub(crate) fn summed(&self) -> f64 {
        self.summed
    }
}

/// For a model's own test (task 180): rows `0..n`, 1 ms apart, through
/// `step(row, stamp, d_clock)`, which steps the model and says whether the
/// row was an event; the rows that were. Each row is handed its stamp in
/// integer nanoseconds when `stamped`, as the stream hands it, and none
/// otherwise, as a direct caller of the core does.
#[cfg(test)]
pub(crate) fn events_on_millisecond_rows(
    n: usize,
    stamped: bool,
    mut step: impl FnMut(usize, Option<Stamp>, f64) -> bool,
) -> Vec<usize> {
    let ms = seconds_of_ns(1_000_000);
    (0..n)
        .filter(|&i| {
            let stamp = stamped.then(|| Stamp::Ns(i as i128 * 1_000_000));
            step(i, stamp, if i == 0 { 0.0 } else { ms })
        })
        .collect()
}

/// `stamp` moved back by `d` clock units: where the clock stood `d` before
/// it. Exact at `d = 0`, the stamp itself, which is the case of a stream
/// whose first row is learned; past a skipped first row, the stamp of the
/// stream's start to the rounding of `d`, a sum of doubles. A temporal
/// stamp moves by `d`'s nanoseconds ([`ns_of_seconds`]), a float one by
/// adding `d` to the time it removes, an integer one by `d`'s whole units
/// and its fraction (task 200); a `d` with no nanoseconds (infinite) leaves
/// a number of the same clock.
fn back_by(stamp: Stamp, d: f64) -> Stamp {
    if d == 0.0 {
        return stamp;
    }
    match stamp {
        Stamp::Ns(n) => match ns_of_seconds(d) {
            Some(dn) => Stamp::Ns(n.saturating_sub(dn)),
            None => Stamp::Raw(seconds_of_ns(n) - d, 0.0),
        },
        Stamp::Raw(raw, removed) => Stamp::Raw(raw, removed + d),
        Stamp::Int(whole, frac) => stamp
            .int_back_by(d)
            .unwrap_or(Stamp::Raw(whole as f64 + frac - d, 0.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i128 = 1_000_000;

    /// The rows a cadence of `every` fires at over `n` rows `d` apart, the
    /// row `i` stamped by `stamp(i)` (none for `None`), each event starting
    /// the clock over as a model's does.
    fn fired(n: usize, d: f64, every: f64, stamp: impl Fn(usize) -> Option<Stamp>) -> Vec<usize> {
        let mut since = Since::default();
        let mut rows = Vec::new();
        for i in 0..n {
            if let Some(s) = stamp(i) {
                since.stamp_next(s);
            }
            since.step(if i == 0 { 0.0 } else { d });
            if since.reached(every) {
                rows.push(i);
                since.restart();
            }
        }
        rows
    }

    /// Task 180: `ew_cov`'s `pca_every` reads the stamps its caller hands,
    /// seen from outside, here because `ewcov.rs` is at the repository's
    /// size cap. A row is scored with the components in force before it,
    /// which a refresh changes for the next row: every 2,000th 1 ms row
    /// under 2 s after the first refresh (the second row, with
    /// `min_weight`); handed no stamps, a row late each time, as it was.
    #[test]
    fn pca_every_measures_the_stamps_it_is_handed() {
        use crate::{Decay, EwCovCfg, EwCovModel, OnlineModel};
        for (stamped, want) in [(true, [1, 2001, 4001]), (false, [1, 2002, 4003])] {
            let mut m = EwCovModel::new(EwCovCfg {
                n_features: 2,
                decay: Decay::Halflife(f64::INFINITY),
                stats: Vec::new(),
                min_weight: 2.0,
                precision_prior: None,
                mahal_quantiles: Vec::new(),
                pca: 1,
                pca_every: 2.0,
                max_rows_between_pca: u32::MAX,
                lags: Vec::new(),
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
            })
            .unwrap();
            // Per row, the component's variance, share and two loadings,
            // which only a refresh moves (its score is the row's own).
            let mut frozen: Vec<Vec<u64>> = Vec::new();
            events_on_millisecond_rows(4_100, stamped, |i, stamp, d| {
                if let Some(s) = stamp {
                    m.stamp_next(s);
                }
                let a = ((i * 37) % 101) as f64 / 101.0;
                let x = [a, 2.0 * a + ((i * 11) % 7) as f64 / 70.0];
                let pred = m.step(&x, &[], d, 1.0).pred;
                frozen.push(pred[..4].iter().map(|v| v.to_bits()).collect());
                false
            });
            let got: Vec<usize> = (0..frozen.len() - 1)
                .filter(|&t| frozen[t] != frozen[t + 1])
                .collect();
            assert_eq!(got, want, "stamped: {stamped}");
        }
    }

    /// Task 180: on 1 ms rows a cadence of 2 s fires every 2,000th row when
    /// the rows are stamped with their integer nanoseconds, where the
    /// summed doubles fire a row late each time. The oracle is the raw
    /// nanoseconds: row `i` is `i` ms from the first, which the clock runs
    /// from.
    #[test]
    fn stamped_rows_fire_on_the_exact_clock_where_summed_doubles_fire_a_row_late() {
        let step = seconds_of_ns(MS);
        let summed: f64 = (0..2000).map(|_| step).sum();
        assert_eq!(summed, 1.999_999_999_999_890_5, "the drift this fixes");
        let stamped = fired(6004, step, 2.0, |i| Some(Stamp::Ns(i as i128 * MS)));
        assert_eq!(stamped, [2000, 4000, 6000]);
        let unstamped = fired(6004, step, 2.0, |_| None);
        assert_eq!(unstamped, [2001, 4002, 6003], "the old rule, unchanged");
    }

    /// On a number clock of tenths a cadence of 0.3 is decided by one
    /// subtraction of the raw values, the oracle below: an event 3 rows
    /// after the last on some rows and 4 on others (`0.3 - 0.0` reaches 0.3,
    /// `1.2 - 0.9` does not), whatever steps the rows are handed with. Steps
    /// of a tenth summed fire every third row.
    #[test]
    fn a_number_clock_is_one_subtraction_of_its_raw_values() {
        let t = |i: usize| i as f64 / 10.0;
        let n = 200;
        let (mut want, mut last) = (Vec::new(), 0);
        for i in 1..n {
            if t(i) - t(last) >= 0.3 {
                want.push(i);
                last = i;
            }
        }
        let gaps: std::collections::BTreeSet<usize> =
            want.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(gaps, [3, 4].into());
        let got = fired(n, 0.1, 0.3, |i| Some(Stamp::Raw(t(i), 0.0)));
        assert_eq!(got, want);
        let by_sums = fired(n, 0.1, 0.3, |_| None);
        assert_eq!(by_sums, (3..n).step_by(3).collect::<Vec<_>>());
    }

    /// A row with no stamp is measured as before task 180, bit for bit: the
    /// `d_clock`s summed since the last event against `every`.
    #[test]
    fn an_unstamped_row_reads_the_summed_clock_as_every_cadence_did() {
        for (d, every) in [(0.1, 0.3), (0.001, 2.0), (7.0, 5.0), (0.25, 1.0)] {
            let mut since = Since::default();
            let mut clock = 0.0_f64;
            for i in 0..5000 {
                let step = if i == 0 { 0.0 } else { d };
                since.step(step);
                clock += step;
                assert_eq!(since.reached(every), clock >= every, "{d} {every} row {i}");
                if clock >= every {
                    since.restart();
                    clock = 0.0;
                }
            }
        }
    }

    /// Before the first event the clock runs from the model's start: the
    /// first stamped row's stamp less its own step, exactly the row's stamp
    /// at a step of 0, and the stream's start past skipped rows.
    #[test]
    fn the_first_event_counts_from_the_start() {
        let mut since = Since::default();
        since.stamp_next(Stamp::Ns(5 * MS));
        since.step(0.0);
        assert_eq!(since.from, Some(Stamp::Ns(5 * MS)), "a step of 0: the row");
        let mut since = Since::default();
        since.stamp_next(Stamp::Ns(3 * MS));
        since.step(seconds_of_ns(3 * MS));
        assert_eq!(
            since.from,
            Some(Stamp::Ns(0)),
            "past skipped rows: the start"
        );
        since.stamp_next(Stamp::Ns(2000 * MS));
        since.step(seconds_of_ns(1997 * MS));
        assert!(since.reached(2.0), "2 s from the start");
        let mut since = Since::default();
        since.stamp_next(Stamp::Raw(10.5, 1.0));
        since.step(0.25);
        assert_eq!(since.from, Some(Stamp::Raw(10.5, 1.25)));
        since.stamp_next(Stamp::Raw(12.25, 1.0));
        since.step(1.75);
        assert!(since.reached(2.0) && !since.reached(2.0 + 1e-9));
        // A step with no nanoseconds leaves the clock as a number.
        assert_eq!(
            back_by(Stamp::Ns(MS), f64::INFINITY),
            Stamp::Raw(f64::NEG_INFINITY, 0.0)
        );
    }

    /// An event between two rows -- `ewridge`'s solve after a blend --
    /// starts the clock where the last row left it, and the next row's step
    /// counts from there.
    #[test]
    fn an_event_between_rows_starts_the_clock_at_the_last_row() {
        let mut since = Since::default();
        for i in 0..=700 {
            since.stamp_next(Stamp::Ns(i * MS));
            since.step(if i == 0 { 0.0 } else { 0.001 });
        }
        since.restart();
        assert_eq!(since.from, Some(Stamp::Ns(700 * MS)));
        since.stamp_next(Stamp::Ns(2699 * MS));
        since.step(1.999);
        assert!(!since.reached(2.0));
        since.stamp_next(Stamp::Ns(2700 * MS));
        since.step(0.001);
        assert!(since.reached(2.0));
    }

    /// The stamps and the summed clock are state, and a resumed schedule
    /// fires where the unbroken one does; the stamp handed for the next row
    /// is not.
    #[test]
    fn a_saved_schedule_resumes_where_it_stood() {
        let mut since = Since::default();
        for i in 0..=2500 {
            since.stamp_next(Stamp::Ns(i * MS));
            since.step(if i == 0 { 0.0 } else { 0.001 });
            if since.reached(2.0) {
                since.restart();
            }
        }
        let bytes = rmp_serde::to_vec(&since).unwrap();
        let mut back: Since = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, since);
        let json = serde_json::to_string(&since).unwrap();
        assert_eq!(serde_json::from_str::<Since>(&json).unwrap(), since);
        since.stamp_next(Stamp::Ns(9 * MS));
        let pending: Since = rmp_serde::from_slice(&rmp_serde::to_vec(&since).unwrap()).unwrap();
        assert_eq!(pending.next, None, "a pending stamp is not state");
        since.next = None;
        for i in 2501..=4000 {
            for m in [&mut since, &mut back] {
                m.stamp_next(Stamp::Ns(i * MS));
                m.step(0.001);
            }
            assert_eq!(since.reached(2.0), back.reached(2.0), "row {i}");
            assert_eq!(back.reached(2.0), i == 4000, "row {i}");
        }
    }
}
