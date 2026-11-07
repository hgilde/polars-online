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
        // Every problem at once: none of these reads what another refuses.
        let mut no = online_core::Refusals::new();
        let Some(label) = &self.label_clock else {
            for (param, given) in [
                ("max_held_rows", self.max_held_rows.is_some()),
                ("held_overflow", self.held_overflow.is_some()),
            ] {
                if given {
                    no.refuse(format!(
                        "spec {name:?}: {param} applies only under label_clock, which holds \
                         each row until its target becomes known, but label_clock is not set. \
                         Give label_clock, or leave {param} out."
                    ));
                }
            }
            return no.finish();
        };
        if let LabelClock::PerTarget(cols) = label {
            // An empty list is the next check's: it names no column.
            if !cols.is_empty() && cols.len() != self.m() {
                no.refuse(format!(
                    "spec {name:?}: label_clock lists {} for {}, but a spec reads one time a \
                     row at which its targets become known. Give a single column for all the \
                     targets.",
                    counted(cols.len(), "column"),
                    counted(self.m(), "target"),
                ));
            } else if let Some(other) = cols.iter().find(|c| *c != &cols[0]) {
                no.refuse(format!(
                    "spec {name:?}: label_clock lists {:?} and {other:?}, but a spec reads one \
                     time a row at which its targets become known. Give a single column for all \
                     the targets, or fit the targets that become known at other times in a spec \
                     of their own.",
                    cols[0]
                ));
            }
        }
        if label.column().is_empty() {
            no.refuse(format!(
                "spec {name:?}: label_clock names no column. Give the column that holds the \
                 time each row's target becomes known."
            ));
        } else if self.clock.is_none() {
            no.refuse(format!(
                "spec {name:?}: label_clock {:?} needs clock, since the time a target becomes \
                 known is a place on the clock, but clock is not set. Set clock to a column of \
                 the same type as {:?}, with gap_cap.",
                label.column(),
                label.column()
            ));
        } else if self.clock.as_deref() == Some(label.column()) {
            no.refuse(format!(
                "spec {name:?}: label_clock is the clock column {:?}, so every target is known \
                 at its own row. Leave label_clock out to learn each row right after it is \
                 predicted.",
                label.column()
            ));
        }
        if self.embargo.is_some() {
            no.refuse(format!(
                "spec {name:?}: label_clock does not work beside embargo, which it generalizes. \
                 Give the times the targets become known as a column (the clock plus the \
                 delay), and leave embargo out."
            ));
        }
        if self.targets.any_formula() {
            no.refuse(format!(
                "spec {name:?}: label_clock does not work beside a formula target, which is \
                 learned when its window closes. Give the target as a column, and the time it \
                 becomes known in label_clock."
            ));
        }
        if self.group_close.is_some() {
            no.refuse(format!(
                "spec {name:?}: label_clock does not work beside group_close, since a closed \
                 group cannot learn the rows it still holds. Leave group_close out."
            ));
        }
        if self.session.is_some() {
            no.refuse(format!(
                "spec {name:?}: label_clock does not work beside a session column, since a row \
                 learned when its target becomes known can fall in a later session than the row \
                 it was predicted at. Leave session out, or give embargo in place of \
                 label_clock."
            ));
        }
        if self.drift_action.as_deref() == Some("reset") {
            no.refuse(format!(
                "spec {name:?}: label_clock does not work beside drift_action=\"reset\", since \
                 a model that resets would learn the held rows from a clock it never reached. \
                 Use drift_action=\"flag\"."
            ));
        }
        // The kinds whose prediction reads the clock since the newest learned
        // row: under `label_clock` that row is learned at its label time,
        // so the step would no longer be the row's distance from it.
        let reads_step = match &self.model {
            ModelKind::Holt { .. } => Some("holt extrapolates"),
            ModelKind::Kalman { .. } => Some("kalman's drift and error_inflation grow"),
            ModelKind::Hmm { .. } => Some("hmm's transitions age"),
            ModelKind::Micro { .. } => Some("micro's small clusters age"),
            ModelKind::Audit { .. } => {
                no.refuse(format!(
                    "spec {name:?}: label_clock does not apply to audit, which has no target \
                     to wait for. Leave label_clock out."
                ));
                None
            }
            _ => None,
        };
        if let Some(what) = reads_step {
            no.refuse(format!(
                "spec {name:?}: label_clock does not work with {}, since {what} from the clock \
                 of the newest row learned, and under label_clock that row can be far older than \
                 the row predicted. Give embargo in place of label_clock for a delay that is the \
                 same for every row.",
                self.model.kind_name()
            ));
        }
        if self.max_held_rows == Some(0) {
            no.refuse(format!(
                "spec {name:?}: max_held_rows must be at least 1, but it is 0, which holds no \
                 row. Set max_held_rows to a positive number of rows, or leave it out for no \
                 bound."
            ));
        }
        if let Some(o) = &self.held_overflow
            && !HELD_OVERFLOW.contains(&o.as_str())
        {
            no.refuse(super::spec_words_refusal(
                name,
                "held_overflow",
                &HELD_OVERFLOW,
                o,
            ));
        }
        no.finish()
    }
}

/// `1 column`, `2 columns`: a count and its noun.
fn counted(n: usize, noun: &str) -> String {
    format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
}
