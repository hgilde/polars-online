//! Which slots have held one value, and since which of their learned rows
//! (docs/PLAN.md task 94; by row since the review of 2026-09-25).
//!
//! A window reads a slot's spread by subtraction ([`crate::truncated`]),
//! and a slot that held one value over every row inside the window has no
//! spread there, which a subtraction cannot say: it leaves a remainder that
//! grows with the level and with the rows since the window's edge. A run
//! can. Per slot it keeps the value the slot's learned rows have carried
//! since it last changed, and the index, among those rows, of the one that
//! started the run; the slot is held over the window when the run started
//! at or before the first learned row inside it. The rule used to compare
//! the run's decayed weight with the window's: two numbers equal in exact
//! arithmetic, which drift apart by about the rows in the window times a
//! rounding step, past the tolerance under a long window and a long
//! halflife (fifty thousand rows of each at a halflife of a million), where
//! the held feature then read as moving. Row indices do not drift. A row of
//! weight 0 learns nothing, so it neither starts nor ends a run, and the
//! caller does not offer it; nor does the caller offer a row whose weight is
//! nothing next to the accumulator's (`crate::window::EMPTY_FRACTION` of
//! it), the window's own notion of nothing.

use serde::{Deserialize, Serialize};

/// Per slot, the value its learned rows have carried since it last changed,
/// and the learned-row index that started the run, `u64::MAX` before its
/// first learned row (the module docs). Empty in a state written before the
/// runs, which knew nothing of them: the next learned row starts every one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Runs {
    #[serde(default)]
    x: Vec<f64>,
    #[serde(default)]
    start: Vec<u64>,
    /// No run is kept ([`Self::off`]). `false` in a state written before
    /// it, which so keeps tracking: the side that is always right. Last,
    /// and not skipped, so both encodings read a state without it.
    #[serde(default)]
    off: bool,
}

impl Runs {
    pub fn new(k: usize) -> Self {
        Self {
            x: vec![0.0; k],
            start: vec![u64::MAX; k],
            off: false,
        }
    }

    /// Runs that keep nothing, for an owner without a window: only a
    /// window reads them ([`crate::truncated`]), and a window is fixed when
    /// the owner is built. Tracking them anyway cost `ewridge` a fifth of
    /// its row at five features (docs/PLAN.md task 128).
    pub fn off() -> Self {
        Self {
            x: Vec::new(),
            start: Vec::new(),
            off: true,
        }
    }

    /// Whether these runs keep nothing ([`Self::off`]).
    pub fn is_off(&self) -> bool {
        self.off
    }

    /// Both halves of every one of `k` runs are here: not a state written
    /// before them, nor one that lost a half.
    pub fn is_known(&self, k: usize) -> bool {
        self.x.len() == k && self.start.len() == k
    }

    /// A learned row carrying every slot, the `row`-th learned row, counted
    /// from 1 by the caller: a value the run holds extends it; another
    /// value, or a slot's first learned row, starts a run at this row. Runs
    /// that are not known start here.
    pub fn track(&mut self, x: &[f64], row: u64) {
        if self.off {
            return;
        }
        if !self.is_known(x.len()) {
            *self = Self::new(x.len());
        }
        for ((rx, rs), &xi) in self.x.iter_mut().zip(self.start.iter_mut()).zip(x) {
            if *rs == u64::MAX || *rx != xi {
                (*rx, *rs) = (xi, row);
            }
        }
    }

    /// [`Self::track`] for slot `i` of `k` alone, at that slot's own count
    /// of learned rows: a stream whose slots are present on rows of their
    /// own, as each target of a regression is.
    pub fn track_one(&mut self, k: usize, i: usize, x: f64, row: u64) {
        if self.off {
            return;
        }
        if !self.is_known(k) {
            *self = Self::new(k);
        }
        if self.start[i] == u64::MAX || self.x[i] != x {
            (self.x[i], self.start[i]) = (x, row);
        }
    }

    /// No run stands any more, and each starts again from its next learned
    /// row: for moments that now stand for rows the runs never saw (a blend
    /// with the slow twin, say). It can leave a window reading a held slot
    /// from its subtraction, never the reverse.
    pub fn forget(&mut self) {
        self.x.clear();
        self.start.clear();
    }

    /// The value slot `i` of `k` has held on every learned row from the
    /// `row`-th on: `Some` when its run started at or before that row,
    /// `None` where it moved since, has no learned row yet, or the runs are
    /// not known.
    pub fn started_by(&self, k: usize, i: usize, row: u64) -> Option<f64> {
        (self.is_known(k) && self.start[i] != u64::MAX && self.start[i] <= row).then(|| self.x[i])
    }

    /// Slot `i`'s value and start row, where the runs are known.
    pub fn get(&self, i: usize) -> Option<(f64, u64)> {
        Some((*self.x.get(i)?, *self.start.get(i)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs extend on a repeated value and restart on another, at the row
    /// that changed it; a forgotten run starts again at the next learned
    /// row whatever its value.
    #[test]
    fn runs_follow_the_values_and_their_start_rows() {
        let mut r = Runs::new(2);
        assert_eq!(
            r.get(0),
            Some((0.0, u64::MAX)),
            "no run before a learned row"
        );
        r.track(&[1.0, 5.0], 1);
        assert_eq!((r.get(0), r.get(1)), (Some((1.0, 1)), Some((5.0, 1))));
        r.track(&[1.0, 6.0], 2);
        assert_eq!(r.get(0), Some((1.0, 1)), "the same value extends the run");
        assert_eq!(r.get(1), Some((6.0, 2)), "another value starts one");
        assert_eq!(r.started_by(2, 0, 1), Some(1.0));
        assert_eq!(
            r.started_by(2, 1, 1),
            None,
            "started after the row asked about"
        );
        assert_eq!(r.started_by(2, 1, 2), Some(6.0));
        r.track_one(2, 1, 6.0, 3);
        assert_eq!(r.get(1), Some((6.0, 2)));
        r.track_one(2, 1, 7.0, 4);
        assert_eq!(r.get(1), Some((7.0, 4)));
        r.forget();
        assert!(!r.is_known(2));
        assert_eq!(
            r.started_by(2, 0, u64::MAX),
            None,
            "forgotten runs cover nothing"
        );
        r.track(&[1.0, 7.0], 5);
        assert_eq!(
            (r.get(0), r.get(1)),
            (Some((1.0, 5)), Some((7.0, 5))),
            "started again, the values the same or not"
        );
    }

    /// Runs that are off keep nothing, whatever they are offered, and read
    /// as no run; forgetting leaves them off. A state written before the
    /// flag reads as on, and tracks (docs/PLAN.md task 128).
    #[test]
    fn runs_that_are_off_keep_nothing() {
        let mut r = Runs::off();
        r.track(&[1.0, 2.0], 1);
        r.track_one(2, 1, 3.0, 2);
        assert!(r.is_off() && !r.is_known(2));
        assert_eq!(r.started_by(2, 0, u64::MAX), None);
        r.forget();
        assert!(r.is_off());
        let old: Runs =
            serde_json::from_value(serde_json::json!({"x": [1.0], "start": [4]})).unwrap();
        assert!(!old.is_off());
        assert_eq!(old.started_by(1, 0, 4), Some(1.0));
    }

    /// Runs that are not known -- a state written before them, or one that
    /// lost a half -- start at the next learned row, and a slot no learned
    /// row has carried yet starts on its first, whatever the value.
    #[test]
    fn unknown_runs_start_at_the_next_row() {
        for mut r in [
            Runs::default(),
            Runs {
                x: vec![1.0],
                start: vec![],
                off: false,
            },
        ] {
            assert!(!r.is_known(2) && r.started_by(2, 0, u64::MAX).is_none());
            r.track(&[1.0, 2.0], 9);
            assert_eq!((r.get(0), r.get(1)), (Some((1.0, 9)), Some((2.0, 9))));
        }
        let mut r = Runs::default();
        r.track_one(3, 2, 4.0, 1);
        assert!(r.is_known(3));
        assert_eq!(r.get(2), Some((4.0, 1)));
        assert_eq!(r.get(0), Some((0.0, u64::MAX)));
        assert_eq!(r.started_by(3, 0, u64::MAX), None, "no learned row yet");
        r.track_one(3, 0, 0.0, 1);
        assert_eq!(
            r.get(0),
            Some((0.0, 1)),
            "its first row starts its run, whatever the value"
        );
    }
}
