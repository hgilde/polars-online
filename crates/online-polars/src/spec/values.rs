//! The value types a spec's parameters are written in -- a number that also
//! reads `"inf"`, a number or a list, a session gap, a count or `"auto"`, a
//! window's memory budget, the ridge's scale and a window's `closed` -- each
//! with the serde rules that name a wrong value as what it is. Split from
//! `spec.rs` to keep it under the repository's file-size cap.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

use super::words_body;
use crate::span::Span;

/// A float that also accepts the JSON strings `"inf"` / `"-inf"`, since JSON
/// has no infinity literal and `half_life = inf` is meaningful (it pins a
/// coefficient, docs/PLAN.md §4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num(pub f64);

/// A window's memory budget as a spec writes it: `{"thin": MiB}` or
/// `{"refuse": MiB}`, the MiB a [`Num`] so that `"inf"`, no bound, reads
/// from JSON (review 2026-09-12, P4; the user's decision of 2026-09-15).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowBudgetSpec {
    Thin(Num),
    Refuse(Num),
}

impl WindowBudgetSpec {
    pub fn mib(&self) -> f64 {
        match *self {
            WindowBudgetSpec::Thin(m) | WindowBudgetSpec::Refuse(m) => m.0,
        }
    }

    pub fn to_core(self) -> online_core::WindowBudget {
        match self {
            WindowBudgetSpec::Thin(m) => online_core::WindowBudget::Thin(m.0),
            WindowBudgetSpec::Refuse(m) => online_core::WindowBudget::Refuse(m.0),
        }
    }
}

impl Serialize for Num {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            s.serialize_f64(self.0)
        } else if self.0 == f64::INFINITY {
            s.serialize_str("inf")
        } else if self.0 == f64::NEG_INFINITY {
            s.serialize_str("-inf")
        } else {
            s.serialize_str("nan")
        }
    }
}

impl Num {
    /// The words accepted in place of a number for the non-finite values.
    pub(crate) fn from_word(w: &str) -> Option<Num> {
        match w.to_ascii_lowercase().as_str() {
            "inf" | "+inf" | "infinity" | "+infinity" => Some(Num(f64::INFINITY)),
            "-inf" | "-infinity" => Some(Num(f64::NEG_INFINITY)),
            "nan" => Some(Num(f64::NAN)),
            _ => None,
        }
    }
}

// Hand-written visitors rather than `#[serde(untagged)]`: an untagged enum
// that matches nothing reports "data did not match any variant of untagged
// enum FloatOrList", which names a Rust type and not what was expected. A
// visitor says `invalid type: string "10", expected a number or a list of
// numbers ("inf" allowed)`, and does so for JSON, TOML and the msgpack state
// file alike (all three are self-describing, so `deserialize_any` is exactly
// what the untagged form used underneath).

struct NumVisitor;

impl serde::de::Visitor<'_> for NumVisitor {
    type Value = Num;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number or \"inf\"/\"-inf\"")
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Num, E> {
        Ok(Num(v))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Num, E> {
        Ok(Num(v as f64))
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Num, E> {
        Ok(Num(v as f64))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Num, E> {
        Num::from_word(v).ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(v), &self))
    }
}

impl<'de> Deserialize<'de> for Num {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(NumVisitor)
    }
}

/// A float or a list of floats (grids; `half_life` lists mean one accumulator
/// per value, docs/PLAN.md §4.1).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FloatOrList {
    Float(Num),
    List(Vec<Num>),
}

impl<'de> Deserialize<'de> for FloatOrList {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = FloatOrList;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a number or a list of numbers (\"inf\" allowed)")
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v)))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v as f64)))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<FloatOrList, E> {
                Ok(FloatOrList::Float(Num(v as f64)))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<FloatOrList, E> {
                Num::from_word(v)
                    .map(FloatOrList::Float)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(v), &self))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                seq: A,
            ) -> Result<FloatOrList, A::Error> {
                Vec::<Num>::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
                    .map(FloatOrList::List)
            }
        }

        d.deserialize_any(V)
    }
}

impl FloatOrList {
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            FloatOrList::Float(f) => vec![f.0],
            FloatOrList::List(v) => v.iter().map(|n| n.0).collect(),
        }
    }
}

/// `session_gap`: clock units, a duration under a temporal clock, or
/// "reset". The gap is a [`Span`], so an infinite one ("never") survives
/// JSON, which has no infinity literal.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SessionGapSpec {
    Gap(Span),
    Word(String),
}

impl<'de> Deserialize<'de> for SessionGapSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl serde::de::Visitor<'_> for V {
            type Value = SessionGapSpec;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(
                    "a gap in clock units (\"inf\" for never), a duration such as \"10m\", \
                     or the word \"reset\"",
                )
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Span::Units(v)))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Span::Units(v as f64)))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<SessionGapSpec, E> {
                Ok(SessionGapSpec::Gap(Span::Units(v as f64)))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<SessionGapSpec, E> {
                if v == "reset" {
                    return Ok(SessionGapSpec::Word(v.to_string()));
                }
                if let Some(n) = Num::from_word(v) {
                    return Ok(SessionGapSpec::Gap(Span::Units(n.0)));
                }
                crate::span::Duration::parse(v)
                    .map(|d| SessionGapSpec::Gap(Span::Duration(d)))
                    .map_err(|e| E::custom(format!("{e}, or the word \"reset\"")))
            }
        }

        d.deserialize_any(V)
    }
}

/// A count of at least one, or `"auto"`, which the bank sizes from its own
/// pool: what `marginal`'s `shards` and `ewridge`'s and `ew_cov`'s
/// `gram_threads` take. A setting, not state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountOrAuto {
    Count(usize),
    Auto,
}

/// How many ranges of features a `marginal` splits its pair work into
/// (docs/PLAN.md task 126): a count, or `"auto"`, sized from the model's
/// width and the bank's pool.
pub type ShardSpec = CountOrAuto;

/// How many threads of the bank's pool a spec's `k x k` Gram update may run
/// on (docs/PLAN.md tasks 225 and 253): a count, capped at the pool's size,
/// or `"auto"`, the whole pool. Absent is one: a bank whose groups already
/// fill the pool ran 9-11% slower under `"auto"` (task 253's measurement).
pub type GramThreads = CountOrAuto;

impl Serialize for CountOrAuto {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ShardSpec::Count(n) => s.serialize_u64(*n as u64),
            ShardSpec::Auto => s.serialize_str("auto"),
        }
    }
}

/// A count or `"auto"`, each read by its own rule, so a wrong value is
/// named as what it is.
impl<'de> Deserialize<'de> for CountOrAuto {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = CountOrAuto;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a whole number of at least 1, or \"auto\"")
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<ShardSpec, E> {
                usize::try_from(v)
                    .ok()
                    .filter(|n| *n >= 1)
                    .map(ShardSpec::Count)
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Unsigned(v), &self))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<ShardSpec, E> {
                match u64::try_from(v) {
                    Ok(v) => self.visit_u64(v),
                    Err(_) => Err(E::invalid_value(serde::de::Unexpected::Signed(v), &self)),
                }
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<ShardSpec, E> {
                match v {
                    "auto" => Ok(ShardSpec::Auto),
                    // The one shape of a word's refusal (task 244, P-4).
                    _ => Err(E::custom(format!(
                        "must be an int or one of \"auto\"; got {v:?}"
                    ))),
                }
            }
        }
        d.deserialize_any(V)
    }
}

/// What `ewridge`'s ridge is scaled against (docs/PLAN.md task 151): a
/// penalty on the mean moments, permanent, or a prior on the decaying sums,
/// fading (`"sum"`, classic RLS regularization; task 144 named the
/// difference, where `ridge_decay` was a bool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RidgeScale {
    #[default]
    Mean,
    Sum,
}

impl RidgeScale {
    /// The default, which a `huber` spec does not write.
    pub fn is_mean(&self) -> bool {
        *self == Self::Mean
    }
}

/// A windowed model's `closed`: the window operators' name and two of its
/// values (`crate::windows::Closed`, Polars' `rolling_*_by(closed=)`;
/// docs/PLAN.md task 196, N17), which edge of the window holds the row
/// exactly `window_size` old. `Right`, the default, keeps the rows less
/// than `window_size` old -- a row exactly that old has left, as it has in
/// the window operators and in `rolling_*_by` -- and `Both` keeps it too.
/// Polars' `"left"` and `"none"` leave out the row the window ends at,
/// which a model cannot do: it reads its fit after it has learned that row.
/// The two are refused with that reason as the spec is read, and any other
/// word with the two this takes, so the words the parameter takes are the
/// words it accepts (review round 5, E2: the API snapshot pinned all four).
/// Written as `Closed` writes the same two, so a state file does not move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelClosed {
    #[default]
    Right,
    Both,
}

impl<'de> Deserialize<'de> for ModelClosed {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = ModelClosed;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("`right` or `both`")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<ModelClosed, E> {
                match v {
                    "right" => Ok(ModelClosed::Right),
                    "both" => Ok(ModelClosed::Both),
                    "left" | "none" => Err(E::custom(format!(
                        "closed = \"{v}\" leaves out the current row, the one the window ends \
                         at; a model reads its fit after it has learned that row, so its window \
                         always holds it: give \"right\" (the default) or \"both\""
                    ))),
                    _ => Err(E::custom(words_body(&["right", "both"], v))),
                }
            }
        }
        d.deserialize_str(V)
    }
}

/// The edge a windowed model's ring keeps for its `closed`.
pub(super) fn model_edge(closed: ModelClosed) -> online_core::WindowClosed {
    match closed {
        ModelClosed::Right => online_core::WindowClosed::Right,
        ModelClosed::Both => online_core::WindowClosed::Both,
    }
}
