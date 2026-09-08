//! Floats that survive a human-readable encoding.
//!
//! JSON has no literal for `NaN` or `±inf`, and `serde_json` writes all three
//! as `null` without saying so. That is not a corner: `halflife = inf` means
//! "no decay" and is a documented setting, so an ordinary state carries an
//! infinity in every stream's [`crate::Decay`].
//!
//! These helpers write the three as the strings `"inf"`, `"-inf"` and
//! `"nan"` -- the spelling `online-polars`' `Num` already uses for the same
//! reason, so a spec's `halflife` and a state's `decay` read alike -- and
//! read them back. They key on [`serde::Serializer::is_human_readable`],
//! which **msgpack reports as `false`**, so the state file's bytes are
//! exactly what they were: `crates/online-core/tests/state_encoding.rs`
//! holds an annotated field to the same bytes as a bare `f64`.
//!
//! Apply with `#[serde(with = "crate::humanfloat::f64_or_tag")]` to any
//! config float that a caller may legitimately set to infinity. Missing one
//! is caught rather than shipped: `Bank::save_json_string` re-reads its own
//! output and refuses if it does not match the state.

use serde::{Deserializer, Serialize, Serializer};

fn tag(v: f64) -> &'static str {
    if v.is_nan() {
        "nan"
    } else if v > 0.0 {
        "inf"
    } else {
        "-inf"
    }
}

fn untag<E: serde::de::Error>(s: &str) -> Result<f64, E> {
    match s {
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        "nan" => Ok(f64::NAN),
        _ => Err(E::custom(format!(
            "expected a number or \"inf\"/\"-inf\"/\"nan\", got {s:?}"
        ))),
    }
}

struct Visit;

impl serde::de::Visitor<'_> for Visit {
    type Value = f64;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a number, or \"inf\" / \"-inf\" / \"nan\"")
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<f64, E> {
        Ok(v)
    }
    fn visit_f32<E: serde::de::Error>(self, v: f32) -> Result<f64, E> {
        Ok(v as f64)
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<f64, E> {
        Ok(v as f64)
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<f64, E> {
        Ok(v as f64)
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<f64, E> {
        untag(v)
    }
}

/// One `f64`.
pub mod f64_or_tag {
    use super::*;

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() && !v.is_finite() {
            s.serialize_str(tag(*v))
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        d.deserialize_any(Visit)
    }
}

/// A `Vec<f64>`, element by element (a per-slot halflife, say).
pub mod vec_f64_or_tag {
    use super::*;

    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        if !s.is_human_readable() {
            return v.serialize(s);
        }
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for x in v {
            if x.is_finite() {
                seq.serialize_element(x)?;
            } else {
                seq.serialize_element(tag(*x))?;
            }
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        struct Seq;
        impl<'de> serde::de::Visitor<'de> for Seq {
            type Value = Vec<f64>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sequence of numbers or \"inf\" / \"-inf\" / \"nan\"")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Vec<f64>, A::Error> {
                let mut out = Vec::with_capacity(a.size_hint().unwrap_or(0));
                while let Some(x) = a.next_element_seed(One)? {
                    out.push(x);
                }
                Ok(out)
            }
        }
        struct One;
        impl<'de> serde::de::DeserializeSeed<'de> for One {
            type Value = f64;
            fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<f64, D::Error> {
                d.deserialize_any(Visit)
            }
        }
        d.deserialize_seq(Seq)
    }
}

/// An `Option<f64>`: `null` stays `null`, and a non-finite value is tagged.
pub mod opt_f64_or_tag {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(x) if s.is_human_readable() && !x.is_finite() => s.serialize_some(tag(*x)),
            Some(x) => s.serialize_some(x),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
        struct Opt;
        impl<'de> serde::de::Visitor<'de> for Opt {
            type Value = Option<f64>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("null, a number, or \"inf\" / \"-inf\" / \"nan\"")
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<Option<f64>, E> {
                Ok(None)
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Option<f64>, E> {
                Ok(None)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Option<f64>, D::Error> {
                d.deserialize_any(Visit).map(Some)
            }
        }
        d.deserialize_option(Opt)
    }
}
