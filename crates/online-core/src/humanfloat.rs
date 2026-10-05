//! Floats that survive a human-readable encoding.
//!
//! JSON has no literal for `NaN` or `±inf`, and `serde_json` writes all three
//! as `null` without saying so. That is not a corner: `half_life = inf` means
//! "no decay" and is a documented setting, so an ordinary state carries an
//! infinity in every stream's [`crate::Decay`].
//!
//! These helpers write the three as the strings `"inf"`, `"-inf"` and
//! `"nan"` -- the spelling `online-polars`' `Num` already uses for the same
//! reason, so a spec's `half_life` and a state's `decay` read alike -- and
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

/// A `Vec<f64>`, element by element (a per-slot half-life, say).
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

/// A `Vec<Vec<f64>>`, each inner vector as [`vec_f64_or_tag`] (a per-slot
/// share per coefficient, say, where the intercept's is NaN by definition).
pub mod vec_vec_f64_or_tag {
    use super::*;

    struct Inner<'a>(&'a [f64]);
    impl Serialize for Inner<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            vec_f64_or_tag::serialize(self.0, s)
        }
    }

    pub fn serialize<S: Serializer>(v: &[Vec<f64>], s: S) -> Result<S::Ok, S::Error> {
        if !s.is_human_readable() {
            return v.serialize(s);
        }
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for x in v {
            seq.serialize_element(&Inner(x))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Vec<f64>>, D::Error> {
        struct Seq;
        impl<'de> serde::de::Visitor<'de> for Seq {
            type Value = Vec<Vec<f64>>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sequence of sequences of numbers or tags")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Vec<Vec<f64>>, A::Error> {
                let mut out = Vec::with_capacity(a.size_hint().unwrap_or(0));
                while let Some(x) = a.next_element_seed(One)? {
                    out.push(x);
                }
                Ok(out)
            }
        }
        struct One;
        impl<'de> serde::de::DeserializeSeed<'de> for One {
            type Value = Vec<f64>;
            fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Vec<f64>, D::Error> {
                vec_f64_or_tag::deserialize(d)
            }
        }
        d.deserialize_seq(Seq)
    }
}

/// A `Vec<Option<Vec<f64>>>`, `null` staying `null` and each present vector
/// as [`vec_f64_or_tag`] (a row's `support_coef`, absent off the cadence).
pub mod vec_opt_vec_f64_or_tag {
    use super::*;

    struct Inner<'a>(&'a Option<Vec<f64>>);
    impl Serialize for Inner<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            match self.0 {
                Some(v) => vec_f64_or_tag::serialize(v, s),
                None => s.serialize_none(),
            }
        }
    }

    pub fn serialize<S: Serializer>(v: &[Option<Vec<f64>>], s: S) -> Result<S::Ok, S::Error> {
        if !s.is_human_readable() {
            return v.serialize(s);
        }
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for x in v {
            seq.serialize_element(&Inner(x))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Option<Vec<f64>>>, D::Error> {
        struct Seq;
        impl<'de> serde::de::Visitor<'de> for Seq {
            type Value = Vec<Option<Vec<f64>>>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sequence of nulls or sequences of numbers or tags")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Vec<Option<Vec<f64>>>, A::Error> {
                let mut out = Vec::with_capacity(a.size_hint().unwrap_or(0));
                while let Some(x) = a.next_element_seed(One)? {
                    out.push(x);
                }
                Ok(out)
            }
        }
        struct One;
        impl<'de> serde::de::DeserializeSeed<'de> for One {
            type Value = Option<Vec<f64>>;
            fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Option<Vec<f64>>, D::Error> {
                struct Opt;
                impl<'de> serde::de::Visitor<'de> for Opt {
                    type Value = Option<Vec<f64>>;
                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str("null or a sequence of numbers or tags")
                    }
                    fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                        Ok(None)
                    }
                    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                        Ok(None)
                    }
                    fn visit_some<D2: Deserializer<'de>>(
                        self,
                        d: D2,
                    ) -> Result<Self::Value, D2::Error> {
                        vec_f64_or_tag::deserialize(d).map(Some)
                    }
                }
                d.deserialize_option(Opt)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde::de::IntoDeserializer;

    type ValueError = serde::de::value::Error;

    #[derive(Debug, Serialize, Deserialize)]
    struct One {
        #[serde(with = "f64_or_tag")]
        a: f64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Lists {
        #[serde(with = "vec_opt_vec_f64_or_tag")]
        v: Vec<Option<Vec<f64>>>,
    }

    /// Equal as numbers, NaN equal to NaN.
    fn same(a: &[Option<Vec<f64>>], b: &[Option<Vec<f64>>]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| match (x, y) {
                (None, None) => true,
                (Some(x), Some(y)) => {
                    x.len() == y.len()
                        && x.iter()
                            .zip(y)
                            .all(|(p, q)| p == q || (p.is_nan() && q.is_nan()))
                }
                _ => false,
            })
    }

    /// A number arrives as whatever kind the format holds it as -- an `f32`
    /// in msgpack, a signed or an unsigned integer in JSON -- and reads as
    /// its value.
    #[test]
    fn a_number_of_any_kind_reads_as_its_value() {
        let f: Result<f64, ValueError> = f64_or_tag::deserialize(2.5f32.into_deserializer());
        assert_eq!(f.unwrap(), 2.5);
        let i: Result<f64, ValueError> = f64_or_tag::deserialize((-3i64).into_deserializer());
        assert_eq!(i.unwrap(), -3.0);
        let u: Result<f64, ValueError> = f64_or_tag::deserialize(7u64.into_deserializer());
        assert_eq!(u.unwrap(), 7.0);

        // The same through the two formats a state is written in.
        assert_eq!(serde_json::from_str::<One>(r#"{"a": 7}"#).unwrap().a, 7.0);
        assert_eq!(serde_json::from_str::<One>(r#"{"a": -3}"#).unwrap().a, -3.0);
        #[derive(Serialize)]
        struct Narrow {
            a: f32,
        }
        let bytes = rmp_serde::to_vec_named(&Narrow { a: 2.5 }).unwrap();
        assert_eq!(rmp_serde::from_slice::<One>(&bytes).unwrap().a, 2.5);
    }

    /// A value of the wrong kind is refused with what was expected, in each
    /// helper: the message is what a caller reading a hand-edited state
    /// sees.
    #[test]
    fn a_value_of_the_wrong_kind_names_what_was_expected() {
        fn says<T: std::fmt::Debug, E: std::fmt::Display>(got: Result<T, E>, want: &str) {
            let e = got.expect_err("refused").to_string();
            assert!(e.contains(want), "{e:?} does not say {want:?}");
        }
        let mut d = serde_json::Deserializer::from_str("true");
        says(
            f64_or_tag::deserialize(&mut d),
            "expected a number, or \"inf\" / \"-inf\" / \"nan\"",
        );
        let mut d = serde_json::Deserializer::from_str("5");
        says(
            vec_f64_or_tag::deserialize(&mut d),
            "expected a sequence of numbers or \"inf\" / \"-inf\" / \"nan\"",
        );
        let mut d = serde_json::Deserializer::from_str("5");
        says(
            vec_vec_f64_or_tag::deserialize(&mut d),
            "expected a sequence of sequences of numbers or tags",
        );
        let mut d = serde_json::Deserializer::from_str("5");
        says(
            vec_opt_vec_f64_or_tag::deserialize(&mut d),
            "expected a sequence of nulls or sequences of numbers or tags",
        );
        // JSON hands an option's visitor a value and never asks it what it
        // expected; a deserializer with no options of its own, as serde's
        // value deserializers are, does.
        let b: serde::de::value::BoolDeserializer<ValueError> = true.into_deserializer();
        says(
            opt_f64_or_tag::deserialize(b),
            "expected null, a number, or \"inf\" / \"-inf\" / \"nan\"",
        );
        let seq: serde::de::value::SeqDeserializer<_, ValueError> =
            serde::de::value::SeqDeserializer::new(vec![true].into_iter());
        says(
            vec_opt_vec_f64_or_tag::deserialize(seq),
            "expected null or a sequence of numbers or tags",
        );
    }

    /// A row's `support_coef` -- present vectors with infinities and a NaN
    /// in them, an absent one, an empty one -- reads back as it was written:
    /// in JSON with the non-finite values tagged and the absent one `null`,
    /// and in msgpack as the bytes of the bare type.
    #[test]
    fn a_list_of_optional_lists_round_trips_with_its_tags() {
        let v = vec![
            Some(vec![2.5, f64::INFINITY, f64::NEG_INFINITY]),
            None,
            Some(vec![]),
            Some(vec![f64::NAN, -0.75]),
        ];
        let lists = Lists { v: v.clone() };
        let json = serde_json::to_string(&lists).unwrap();
        assert_eq!(
            json, r#"{"v":[[2.5,"inf","-inf"],null,[],["nan",-0.75]]}"#,
            "the non-finite values are tagged"
        );
        let back: Lists = serde_json::from_str(&json).unwrap();
        assert!(same(&back.v, &v), "{:?} against {v:?}", back.v);

        #[derive(Serialize)]
        struct Bare {
            v: Vec<Option<Vec<f64>>>,
        }
        let bytes = rmp_serde::to_vec_named(&lists).unwrap();
        assert_eq!(
            bytes,
            rmp_serde::to_vec_named(&Bare { v: v.clone() }).unwrap()
        );
        let back: Lists = rmp_serde::from_slice(&bytes).unwrap();
        assert!(same(&back.v, &v), "{:?} against {v:?}", back.v);

        // And an empty list is an empty list.
        let none: Lists = serde_json::from_str(r#"{"v":[]}"#).unwrap();
        assert!(none.v.is_empty());
    }

    /// A format that buffers what it reads -- serde's own, under
    /// `#[serde(flatten)]` or an untagged enum -- hands an option a JSON
    /// `null` as a unit: it stays `null`.
    #[test]
    fn a_null_read_through_a_buffered_format_stays_null() {
        #[derive(Debug, Deserialize)]
        struct Inner {
            #[serde(with = "opt_f64_or_tag")]
            c: Option<f64>,
        }
        #[derive(Debug, Deserialize)]
        struct Outer {
            #[serde(flatten)]
            inner: Inner,
        }
        #[derive(Debug, Deserialize)]
        #[serde(untagged)]
        enum Either {
            It(Inner),
        }
        let flat: Outer = serde_json::from_str(r#"{"c": null}"#).unwrap();
        assert_eq!(flat.inner.c, None);
        let Either::It(un) = serde_json::from_str(r#"{"c": null}"#).unwrap();
        assert_eq!(un.c, None);
        // The values beside it read as they do anywhere.
        let flat: Outer = serde_json::from_str(r#"{"c": "inf"}"#).unwrap();
        assert_eq!(flat.inner.c, Some(f64::INFINITY));
        let flat: Outer = serde_json::from_str(r#"{"c": 1.5}"#).unwrap();
        assert_eq!(flat.inner.c, Some(1.5));
    }
}
