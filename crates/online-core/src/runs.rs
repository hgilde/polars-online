//! Which slots have held one value, and over how much weight (docs/PLAN.md
//! task 94).
//!
//! A window reads a slot's spread by subtraction ([`crate::truncated`]),
//! and a slot that held one value over every row inside the window has no
//! spread there, which a subtraction cannot say: it leaves a remainder that
//! grows with the level and with the rows since the window's edge. A run
//! can. Per slot it keeps the value the slot's learned rows have carried
//! since it last changed, and the weight of those rows, decayed as the
//! accumulator's own weight is; when that weight is all of the window's,
//! the spread is exactly zero. A row of weight 0 learns nothing, so it ends
//! no run: it only ages them.

use serde::{Deserialize, Serialize};

/// Per slot, the value its learned rows have carried since it last changed,
/// and the weight of those rows (the module docs). Empty in a state written
/// before the runs, which knew nothing of them: the next learned row starts
/// every one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Runs {
    #[serde(default)]
    x: Vec<f64>,
    #[serde(default)]
    w: Vec<f64>,
}

impl Runs {
    pub fn new(k: usize) -> Self {
        Self {
            x: vec![0.0; k],
            w: vec![0.0; k],
        }
    }

    /// Both halves of every one of `k` runs are here: not a state written
    /// before them, nor one that lost a half.
    pub fn is_known(&self, k: usize) -> bool {
        self.x.len() == k && self.w.len() == k
    }

    /// A row carrying every slot, at weight `w` after a decay of `lam`: a
    /// value the run holds extends it, `W' = lam·W + w`; another value starts
    /// a run at the row's own weight; a row of weight 0 changes no value and
    /// ages every run. Runs that are not known start here.
    pub fn track(&mut self, x: &[f64], lam: f64, w: f64) {
        if !self.is_known(x.len()) {
            self.x = x.to_vec();
            self.w = vec![w; x.len()];
            return;
        }
        if w > 0.0 {
            for ((rx, rw), &xi) in self.x.iter_mut().zip(self.w.iter_mut()).zip(x) {
                if *rx == xi {
                    *rw = lam * *rw + w;
                } else {
                    (*rx, *rw) = (xi, w);
                }
            }
        } else {
            self.age(lam);
        }
    }

    /// [`Self::track`] for slot `i` of `k` alone: a stream whose slots are
    /// present on rows of their own, as each target of a regression is.
    pub fn track_one(&mut self, k: usize, i: usize, x: f64, lam: f64, w: f64) {
        if !self.is_known(k) {
            *self = Self::new(k);
        }
        if w > 0.0 {
            if self.x[i] == x {
                self.w[i] = lam * self.w[i] + w;
            } else {
                (self.x[i], self.w[i]) = (x, w);
            }
        } else {
            self.w[i] *= lam;
        }
    }

    /// Every run ages by `lam`, as the accumulator's weight does over a row
    /// it does not learn.
    pub fn age(&mut self, lam: f64) {
        for rw in &mut self.w {
            *rw *= lam;
        }
    }

    /// Slot `i`'s run ages by `lam`: a row that does not carry the slot.
    pub fn age_one(&mut self, i: usize, lam: f64) {
        if let Some(rw) = self.w.get_mut(i) {
            *rw *= lam;
        }
    }

    /// No run covers any weight any more, and each starts again from its
    /// next learned row: for moments that now stand for rows the runs never
    /// saw (a blend with the slow twin, say). It can leave a window reading
    /// a held slot from its subtraction, never the reverse.
    pub fn forget(&mut self) {
        self.w.iter_mut().for_each(|w| *w = 0.0);
    }

    /// The value slot `i` of `k` has held on learned rows carrying at least
    /// `weight`, the newest first; `None` where it moved inside that weight,
    /// or where the runs are not known.
    pub fn held_over(&self, k: usize, i: usize, weight: f64) -> Option<f64> {
        (self.is_known(k) && self.w[i] >= weight).then(|| self.x[i])
    }

    /// Slot `i`'s value and run weight, where the runs are known.
    pub fn get(&self, i: usize) -> Option<(f64, f64)> {
        Some((*self.x.get(i)?, *self.w.get(i)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs extend on a repeated value at weight, restart on another value,
    /// and only age on a row of weight 0, whatever it carries.
    #[test]
    fn runs_follow_the_values_and_their_weights() {
        let mut r = Runs::new(2);
        r.track(&[1.0, 5.0], 0.5, 2.0);
        assert_eq!(r.get(0), Some((1.0, 2.0)));
        r.track(&[1.0, 6.0], 0.5, 1.0);
        assert_eq!(r.get(0), Some((1.0, 2.0)), "0.5 * 2 + 1");
        assert_eq!(r.get(1), Some((6.0, 1.0)), "another value starts a run");
        r.track(&[9.0, 9.0], 0.5, 0.0);
        assert_eq!(r.get(0), Some((1.0, 1.0)), "a row of no weight only ages");
        assert_eq!(r.get(1), Some((6.0, 0.5)));
        r.age(0.5);
        r.age_one(1, 0.5);
        assert_eq!(r.get(1), Some((6.0, 0.125)));
        r.track_one(2, 1, 6.0, 1.0, 1.0);
        assert_eq!(r.get(1), Some((6.0, 1.125)));
        r.track_one(2, 1, 7.0, 1.0, 1.0);
        assert_eq!(r.get(1), Some((7.0, 1.0)));
        r.track_one(2, 0, 3.0, 0.5, 0.0);
        assert_eq!(r.get(0), Some((1.0, 0.25)), "weight 0 on one slot: it ages");
        assert_eq!(r.held_over(2, 0, 0.25), Some(1.0));
        assert_eq!(r.held_over(2, 0, 0.3), None);
        r.forget();
        assert!(r.held_over(2, 0, 1e-300).is_none() && r.held_over(2, 1, 1e-300).is_none());
        assert_eq!(
            r.held_over(2, 1, 0.0),
            Some(7.0),
            "a forgotten run covers nothing more"
        );
    }

    /// Runs that are not known -- a state written before them, or one that
    /// lost a half -- start at the next row, as a run of that row alone.
    #[test]
    fn unknown_runs_start_at_the_next_row() {
        for mut r in [
            Runs::default(),
            Runs {
                x: vec![1.0],
                w: vec![],
            },
        ] {
            assert!(!r.is_known(2) && r.held_over(2, 0, 0.0).is_none());
            r.track(&[1.0, 2.0], 0.9, 0.75);
            assert_eq!((r.get(0), r.get(1)), (Some((1.0, 0.75)), Some((2.0, 0.75))));
        }
        let mut r = Runs::default();
        r.track_one(3, 2, 4.0, 0.9, 1.5);
        assert!(r.is_known(3));
        assert_eq!(r.get(2), Some((4.0, 1.5)));
        assert_eq!(r.get(0), Some((0.0, 0.0)));
    }
}
