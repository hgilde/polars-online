//! `label_clock`: each row's label time as a column (docs/PLAN.md task 341;
//! the design is docs/records/TREE-FEATURES.md §2.4). Split from `spec.rs`
//! to keep it under the repository's file-size cap.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::{ModelKind, Spec};

/// `label_clock`: one column, or one per target. Version 1 reads one time
/// per row, so a list must name the same column for every target
/// ([`Spec::check_label_clock`]); the list form is accepted now so that a
/// spec whose targets become known at different times needs no second
/// spelling later.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LabelClock {
    One(String),
    PerTarget(Vec<String>),
}

impl LabelClock {
    /// The column each row's label time is read from: the one column, or
    /// the first of a list ([`Spec::check_label_clock`] refuses a list
    /// whose entries differ).
    pub fn column(&self) -> &str {
        match self {
            LabelClock::One(c) => c,
            LabelClock::PerTarget(v) => v.first().map_or("", String::as_str),
        }
    }
}

impl<'de> Deserialize<'de> for LabelClock {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = LabelClock;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a column name or a list of column names, one per target")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<LabelClock, E> {
                Ok(LabelClock::One(v.to_owned()))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                seq: A,
            ) -> Result<LabelClock, A::Error> {
                Vec::<String>::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
                    .map(LabelClock::PerTarget)
            }
        }

        d.deserialize_any(V)
    }
}

/// What a full buffer of held rows does: `"refuse"` the chunk (the
/// default), or `"drop_oldest"`, which drops the held row that arrived
/// first and counts it in `labels_dropped`.
pub const HELD_OVERFLOW: [&str; 2] = ["refuse", "drop_oldest"];

impl Spec {
    /// Whether a full buffer drops its oldest row rather than refusing the
    /// chunk.
    pub fn drops_oldest_held(&self) -> bool {
        self.held_overflow.as_deref() == Some("drop_oldest")
    }

    /// `label_clock`'s own rules, and the settings it refuses beside it.
    pub(super) fn check_label_clock(&self) -> Result<(), String> {
        let name = &self.name;
        let Some(label) = &self.label_clock else {
            for (param, given) in [
                ("max_held_rows", self.max_held_rows.is_some()),
                ("held_overflow", self.held_overflow.is_some()),
            ] {
                if given {
                    return Err(format!(
                        "spec {name:?}: {param} is given without label_clock, and only rows held \
                         for their label time are counted against it. Give label_clock, or \
                         leave {param} out."
                    ));
                }
            }
            return Ok(());
        };
        if let LabelClock::PerTarget(cols) = label {
            if cols.len() != self.m() {
                return Err(format!(
                    "spec {name:?}: label_clock lists {} columns for {} targets. Give one \
                     column for every target, or a single column for all of them.",
                    cols.len(),
                    self.m()
                ));
            }
            if let Some(other) = cols.iter().find(|c| *c != &cols[0]) {
                return Err(format!(
                    "spec {name:?}: label_clock lists {:?} and {other:?}, and a spec reads one \
                     label time a row. Give one column for every target, or fit the targets \
                     whose labels arrive at other times in a spec of their own.",
                    cols[0]
                ));
            }
        }
        if label.column().is_empty() {
            return Err(format!(
                "spec {name:?}: label_clock names no column. Give the column that holds each \
                 row's label time."
            ));
        }
        if self.clock.is_none() {
            return Err(format!(
                "spec {name:?}: label_clock {:?} is given with no clock, and a label time is \
                 a place on the clock. Give the spec a clock column of the label column's type.",
                label.column()
            ));
        }
        if self.clock.as_deref() == Some(label.column()) {
            return Err(format!(
                "spec {name:?}: label_clock is the clock column {:?}, so every label is known \
                 at its own row. Leave label_clock out to learn each row where it sits.",
                label.column()
            ));
        }
        let beside = if self.embargo.is_some() {
            Some(
                "embargo, which label_clock generalizes. Give the label times as a column \
                 (clock + delay) and leave embargo out",
            )
        } else if self.targets.any_formula() {
            Some(
                "a target that is a window expression, which is learned when its window \
                 closes. Give the target as a column and its label time in label_clock",
            )
        } else if self.group_close.is_some() {
            Some(
                "group_close, since a closed group cannot learn the rows it still holds. \
                 Leave group_close out",
            )
        } else if self.session.is_some() {
            Some(
                "a session column, since a row learned at its label time can fall in a later \
                 session than the row it was scored at. Leave session out, or use embargo",
            )
        } else if self.drift_action.as_deref() == Some("reset") {
            Some(
                "drift_action=\"reset\", since a restarted model would learn the held rows \
                 from a clock it never reached. Use drift_action=\"flag\"",
            )
        } else {
            None
        };
        if let Some(why) = beside {
            return Err(format!(
                "spec {name:?}: label_clock does not work beside {why}."
            ));
        }
        // The kinds whose score reads the clock since the newest learned
        // row: under `label_clock` that row is learned at its label time,
        // so the step would no longer be the row's distance from it.
        let reads_step = match &self.model {
            ModelKind::Holt { .. } => Some("holt's forecast reaches"),
            ModelKind::Kalman { .. } => Some("kalman's drift and error_inflation reach"),
            ModelKind::Hmm { .. } => Some("hmm's transitions age"),
            ModelKind::Micro { .. } => Some("micro's summaries age"),
            ModelKind::Audit { .. } => {
                return Err(format!(
                    "spec {name:?}: label_clock does not apply to audit, which has no label \
                     to wait for. Leave label_clock out."
                ));
            }
            _ => None,
        };
        if let Some(what) = reads_step {
            return Err(format!(
                "spec {name:?}: label_clock does not work with {}, since {what} from the newest \
                 learned row, which label_clock moves to its label time. Use embargo for a \
                 delay that is the same for every row.",
                self.model.kind_name()
            ));
        }
        if self.max_held_rows == Some(0) {
            return Err(format!(
                "spec {name:?}: max_held_rows is 0, which holds no row. Give a positive \
                 number of rows, or leave it out for no bound."
            ));
        }
        if let Some(o) = &self.held_overflow
            && !HELD_OVERFLOW.contains(&o.as_str())
        {
            return Err(super::spec_words_refusal(
                name,
                "held_overflow",
                &HELD_OVERFLOW,
                o,
            ));
        }
        Ok(())
    }
}
