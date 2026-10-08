//! What a loaded bank has learned, as the rows of a resumed input to drop:
//! the command line's `--skip-learned` (docs/PLAN.md task 196, N26), the
//! rule `ModelBank.skip_learned` applies in Python.

use std::collections::HashMap;

use online_core::ClockValue;
use polars::prelude::*;

use super::{Bank, GroupKey, group_indices};
use crate::arrow::{ClockArray, chunk_from_frame};
use crate::spec::Spec;

/// Per spec that reads a clock, each group's last clock as the bank held it
/// when this was taken: the position a resumed input is measured against.
/// Taken once, after the load and before the first row, so every chunk is
/// filtered against the same positions and the chunking moves nothing (hard
/// rule 3).
#[derive(Debug, Clone)]
pub struct Learned {
    specs: Vec<Spec>,
    lasts: Vec<Option<HashMap<GroupKey, ClockValue>>>,
}

impl Bank {
    /// What this bank has learned ([`Learned`]).
    ///
    /// # Errors
    ///
    /// When no spec reads a clock: a row-count bank resumes at the next
    /// row, which only the caller knows.
    pub fn learned(&self) -> Result<Learned, String> {
        let lasts: Vec<Option<HashMap<GroupKey, ClockValue>>> = self
            .specs()
            .iter()
            .zip(self.last_clocks())
            .map(|(spec, groups)| {
                spec.clock.as_ref().map(|_| {
                    groups
                        .into_iter()
                        .filter_map(|(key, last)| last.map(|c| (key, c)))
                        .collect()
                })
            })
            .collect();
        if lasts.iter().all(Option::is_none) {
            return Err(
                "skip_learned: no spec reads a clock, so the bank has no position to resume from; \
                 a row-count bank resumes at the next row, which only the caller knows"
                    .into(),
            );
        }
        Ok(Learned {
            specs: self.specs().to_vec(),
            lasts,
        })
    }
}

impl Learned {
    /// True on the rows of `df` the bank has not learned: in every spec that
    /// reads a clock, a clock after its group's last one, a group the bank
    /// has not seen, or a null clock, which is kept for the bank to judge. A
    /// row at its group's last clock counts as learned. A temporal clock is
    /// compared in the integer nanoseconds the bank keeps, and an integer
    /// clock in integers (task 200).
    ///
    /// # Errors
    ///
    /// A column a spec reads is missing or has a dtype the bank refuses, as
    /// [`Bank::fit_predict`] says it; and a clock temporal in the frame and
    /// numeric in the bank, or the other way round.
    pub fn unlearned(&self, df: &DataFrame) -> PolarsResult<BooleanChunked> {
        let mut keep = vec![true; df.height()];
        let chunk = chunk_from_frame(df, &self.specs)?;
        for (spec, lasts) in self.specs.iter().zip(&self.lasts) {
            let (Some(name), Some(lasts)) = (&spec.clock, lasts) else {
                continue;
            };
            let clocks = chunk.clock(spec, name)?;
            for (key, rows) in group_indices(&chunk, spec)? {
                let Some(last) = lasts.get(&key) else {
                    continue;
                };
                for i in rows {
                    let now = match clocks {
                        ClockArray::F64(a) => a.get(i).map(ClockValue::F64),
                        ClockArray::Nanos(a) => a.get(i).map(ClockValue::Ns),
                        // An integer clock, compared in integers (task 200).
                        ClockArray::I64(a) => a.get(i).map(ClockValue::I64),
                        // Past `i64::MAX` is after every clock an integer
                        // bank holds: kept, for the bank to refuse by row,
                        // as `ModelBank.skip_learned` keeps it.
                        ClockArray::U64(a) => match a.get(i).map(i64::try_from) {
                            None => None,
                            Some(Ok(v)) => Some(ClockValue::I64(v)),
                            Some(Err(_)) => continue,
                        },
                    };
                    // At or before the last learned clock. A comparison
                    // that is undefined -- a NaN clock -- is neither, and
                    // keeps the row for the bank to refuse by row, as a
                    // null is kept and as `ModelBank.skip_learned` keeps
                    // it; read as "not after", it dropped the row in
                    // silence (review round 5, C3).
                    let not_after = |o: Option<std::cmp::Ordering>| {
                        matches!(
                            o,
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        )
                    };
                    let learned = match (now, last) {
                        (None, _) => false,
                        (Some(ClockValue::F64(now)), ClockValue::F64(last)) => {
                            not_after(now.partial_cmp(last))
                        }
                        (Some(ClockValue::Ns(now)), ClockValue::Ns(last)) => now <= *last,
                        (Some(ClockValue::I64(now)), ClockValue::I64(last)) => now <= *last,
                        // An integer and a float clock are both numbers,
                        // compared as doubles, as `ModelBank.skip_learned`
                        // compares them; the bank refuses the chunk by its
                        // dtype (task 200).
                        (Some(ClockValue::I64(now)), ClockValue::F64(last)) => {
                            not_after((now as f64).partial_cmp(last))
                        }
                        (Some(ClockValue::F64(now)), ClockValue::I64(last)) => {
                            not_after(now.partial_cmp(&(*last as f64)))
                        }
                        (Some(now), _) => {
                            let (is, was) = match now {
                                ClockValue::Ns(_) => ("temporal", "numeric"),
                                ClockValue::F64(_) | ClockValue::I64(_) => ("numeric", "temporal"),
                            };
                            polars_bail!(ComputeError:
                                "spec {:?}: clock column {:?} is {is} in the frame and was {was} \
                                 in the bank",
                                spec.name, name
                            );
                        }
                    };
                    if learned {
                        keep[i] = false;
                    }
                }
            }
        }
        Ok(BooleanChunked::from_slice("unlearned".into(), &keep))
    }
}
