//! Quantities measured in clock units: a plain number, or a duration
//! (docs/PLAN.md task 88).
//!
//! A numeric clock column has no unit, so a parameter measured against it is
//! a plain number of the column's units: `halflife = 600` on a clock in
//! seconds is ten minutes. A temporal column (`Datetime`, `Date`,
//! `Duration`) does carry a unit, and a parameter measured against it is a
//! duration: `halflife = "10m"`. Both are read on one internal scale,
//! seconds, so a duration means the same length of time whatever the
//! column's own unit is.
//!
//! A duration is written as polars writes one (`rolling(period=)`): whole
//! numbers, each followed by a fixed-length unit, largest first, such as
//! `"10m"`, `"1h30m"` or `"250ms"`. The calendar units polars also has
//! (`mo`, `q`, `y`) are refused, because a month has no fixed length, and so
//! is `i`, which counts rows rather than time.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::spec::Num;

const NS_PER_S: i64 = 1_000_000_000;

/// Fixed-length units, in the order a duration is written, with their
/// length in nanoseconds. `format_duration` uses every one but `w`, so a
/// fortnight reads `14d` rather than `2w`.
const UNITS: [(&str, i64); 8] = [
    ("w", 7 * 86_400 * NS_PER_S),
    ("d", 86_400 * NS_PER_S),
    ("h", 3_600 * NS_PER_S),
    ("m", 60 * NS_PER_S),
    ("s", NS_PER_S),
    ("ms", 1_000_000),
    ("us", 1_000),
    ("ns", 1),
];

/// `v` units of `per` units a second, as seconds, split so that the whole
/// seconds are exact and the remainder is rounded once. The same instant
/// written in milliseconds, microseconds or nanoseconds therefore reads as
/// the same `f64`, which is what makes a clock's unit irrelevant to the fit.
pub fn seconds_of(v: i64, per: i64) -> f64 {
    v.div_euclid(per) as f64 + v.rem_euclid(per) as f64 / per as f64
}

/// A duration's length in nanoseconds, from its text: `"10m"`, `"1h30m"`,
/// `"-5s"`. The error says what is wrong and shows the right form.
pub fn parse_duration(text: &str) -> Result<i64, String> {
    let bad = |why: String| format!("{text:?} is not a duration: {why}");
    let t = text.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest.trim_start()),
        None => (false, t.strip_prefix('+').unwrap_or(t).trim_start()),
    };
    if body.is_empty() {
        return Err(bad(
            "it is empty; write a number and a unit, as in \"10m\"".into()
        ));
    }
    let mut total: i128 = 0;
    let mut rest = body;
    while !rest.is_empty() {
        let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits == 0 {
            return Err(bad(format!(
                "expected a whole number at {rest:?}; write a number and a unit, as in \"10m\""
            )));
        }
        let (num, after) = rest.split_at(digits);
        if after.starts_with('.') {
            return Err(bad(
                "use whole numbers of a smaller unit, as in \"1h30m\" for an hour and a half"
                    .into(),
            ));
        }
        let letters = after.len() - after.trim_start_matches(|c: char| c.is_alphabetic()).len();
        let (unit, after) = after.split_at(letters);
        let n: i128 = num
            .parse()
            .map_err(|_| bad(format!("{num} is too large")))?;
        let per = match unit {
            "" => {
                return Err(bad(format!(
                    "{num} has no unit; write one, as in \"{num}s\" or \"{num}m\""
                )));
            }
            "µs" | "μs" => 1_000,
            "mo" => {
                return Err(bad(
                    "a month has no fixed length; write days, as in \"30d\"".into(),
                ));
            }
            "q" => {
                return Err(bad(
                    "a quarter has no fixed length; write days, as in \"91d\"".into(),
                ));
            }
            "y" => {
                return Err(bad(
                    "a year has no fixed length; write days, as in \"365d\"".into(),
                ));
            }
            "i" => {
                return Err(bad(
                    "\"i\" counts rows, not time; use a number of rows".into()
                ));
            }
            u => match UNITS.iter().find(|(name, _)| *name == u) {
                Some((_, per)) => *per,
                None => {
                    return Err(bad(format!(
                        "unknown unit {u:?}; use ns, us, ms, s, m, h, d or w"
                    )));
                }
            },
        };
        total += n * i128::from(per);
        if total > i128::from(i64::MAX) {
            return Err(bad(
                "it is longer than 292 years, the most a clock can hold".into(),
            ));
        }
        rest = after.trim_start();
    }
    let ns = total as i64;
    Ok(if neg { -ns } else { ns })
}

/// The text polars would write for a duration of `ns` nanoseconds, largest
/// unit first and zero parts left out: `600e9` is `"10m"`, `5400e9` is
/// `"1h30m"`, and zero is `"0s"`.
pub fn format_duration(ns: i64) -> String {
    if ns == 0 {
        return "0s".into();
    }
    let mut out = String::new();
    if ns < 0 {
        out.push('-');
    }
    let mut left = ns.unsigned_abs();
    for (name, per) in UNITS.iter().skip(1) {
        let per = per.unsigned_abs();
        if left >= per {
            out.push_str(&format!("{}{name}", left / per));
            left %= per;
        }
    }
    out
}

/// A duration as a spec wrote it: the text, kept for display and for the
/// round trip, and its length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Duration {
    pub text: String,
    pub nanos: i64,
}

impl Duration {
    pub fn parse(text: &str) -> Result<Self, String> {
        Ok(Duration {
            text: text.to_string(),
            nanos: parse_duration(text)?,
        })
    }

    pub fn seconds(&self) -> f64 {
        seconds_of(self.nanos, NS_PER_S)
    }
}

/// A quantity measured in clock units.
#[derive(Debug, Clone, PartialEq)]
pub enum Span {
    /// A plain number of the clock column's units, or of rows when the spec
    /// has no clock. `inf` where the parameter allows it.
    Units(f64),
    /// A length of time, which only a temporal clock can measure.
    Duration(Duration),
}

impl Span {
    /// The quantity in clock units: a duration in seconds, the scale a
    /// temporal clock is read on.
    pub fn value(&self) -> f64 {
        match self {
            Span::Units(v) => *v,
            Span::Duration(d) => d.seconds(),
        }
    }

    pub fn is_duration(&self) -> bool {
        matches!(self, Span::Duration(_))
    }

    /// A number whose meaning depends on the clock's unit: anything but
    /// `0` and `±inf`, which mean the same in every unit.
    pub fn is_unit_bound_number(&self) -> bool {
        matches!(self, Span::Units(v) if *v != 0.0 && v.is_finite())
    }

    /// The value as a field name shows it: a duration as written, a
    /// number as [`crate::spec::num_label`] renders it.
    pub fn label(&self) -> String {
        match self {
            Span::Units(v) => crate::spec::num_label(*v),
            Span::Duration(d) => d.text.clone(),
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.label())
    }
}

impl Serialize for Span {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Span::Units(v) => Num(*v).serialize(s),
            Span::Duration(d) => s.serialize_str(&d.text),
        }
    }
}

struct SpanVisitor;

impl serde::de::Visitor<'_> for SpanVisitor {
    type Value = Span;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number of clock units (\"inf\" allowed) or a duration such as \"10m\"")
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Span, E> {
        Ok(Span::Units(v))
    }

    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Span, E> {
        Ok(Span::Units(v as f64))
    }

    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Span, E> {
        Ok(Span::Units(v as f64))
    }

    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Span, E> {
        if let Some(n) = Num::from_word(v) {
            return Ok(Span::Units(n.0));
        }
        Duration::parse(v).map(Span::Duration).map_err(E::custom)
    }
}

impl<'de> Deserialize<'de> for Span {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(SpanVisitor)
    }
}

/// One clock-unit quantity or a list of them: a `halflife` grid, one model
/// instance per value (docs/PLAN.md §4.1), or a per-slot `kalman` halflife.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SpanList {
    One(Span),
    List(Vec<Span>),
}

impl SpanList {
    pub fn spans(&self) -> &[Span] {
        match self {
            SpanList::One(s) => std::slice::from_ref(s),
            SpanList::List(v) => v,
        }
    }

    /// Every value in clock units, in order.
    pub fn to_vec(&self) -> Vec<f64> {
        self.spans().iter().map(Span::value).collect()
    }
}

impl<'de> Deserialize<'de> for SpanList {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = SpanList;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(
                    "a number of clock units, a duration such as \"10m\", or a list of them \
                     (\"inf\" allowed)",
                )
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<SpanList, E> {
                SpanVisitor.visit_f64(v).map(SpanList::One)
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<SpanList, E> {
                SpanVisitor.visit_i64(v).map(SpanList::One)
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<SpanList, E> {
                SpanVisitor.visit_u64(v).map(SpanList::One)
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<SpanList, E> {
                SpanVisitor.visit_str(v).map(SpanList::One)
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, seq: A) -> Result<SpanList, A::Error> {
                Vec::<Span>::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
                    .map(SpanList::List)
            }
        }

        d.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_parse_to_nanoseconds() {
        let s = NS_PER_S;
        for (text, ns) in [
            ("10m", 600 * s),
            ("1h30m", 5_400 * s),
            ("1h 30m", 5_400 * s),
            ("250ms", 250_000_000),
            ("1us500ns", 1_500),
            ("1µs", 1_000),
            ("2d", 172_800 * s),
            ("1w", 604_800 * s),
            ("-5s", -5 * s),
            (" 3s ", 3 * s),
            ("0s", 0),
        ] {
            assert_eq!(parse_duration(text), Ok(ns), "{text}");
        }
    }

    #[test]
    fn what_is_not_a_fixed_duration_is_refused_with_the_fix() {
        for (text, says) in [
            ("10", "has no unit"),
            ("", "empty"),
            ("1.5h", "1h30m"),
            ("1mo", "month has no fixed length"),
            ("1q", "quarter"),
            ("1y", "year"),
            ("3i", "counts rows"),
            ("5x", "unknown unit"),
            ("m", "whole number"),
            ("99999999999w", "292 years"),
        ] {
            let err = parse_duration(text).unwrap_err();
            assert!(
                err.contains(says) && err.contains(&format!("{text:?}")),
                "{text}: {err}"
            );
        }
    }

    #[test]
    fn a_duration_is_written_largest_unit_first() {
        let s = NS_PER_S;
        for (ns, text) in [
            (600 * s, "10m"),
            (5_400 * s, "1h30m"),
            (90 * s, "1m30s"),
            (250_000_000, "250ms"),
            (1_500, "1us500ns"),
            (14 * 86_400 * s, "14d"),
            (-5 * s, "-5s"),
            (0, "0s"),
        ] {
            assert_eq!(format_duration(ns), text);
            assert_eq!(parse_duration(text), Ok(ns), "{text} round-trips");
        }
    }

    #[test]
    fn one_instant_in_any_unit_is_the_same_number_of_seconds() {
        // 2024-01-01 00:00:01.25 UTC, in each unit a Datetime column has.
        let ms: i64 = 1_704_067_201_250;
        let secs = seconds_of(ms, 1_000);
        assert_eq!(seconds_of(ms * 1_000, 1_000_000).to_bits(), secs.to_bits());
        assert_eq!(
            seconds_of(ms * 1_000_000, NS_PER_S).to_bits(),
            secs.to_bits()
        );
        // Before the epoch too: the whole seconds floor and the rest is positive.
        assert_eq!(seconds_of(-1_500, 1_000), -1.5);
        // A duration's length is read on the same scale.
        assert_eq!(Duration::parse("10m").unwrap().seconds(), 600.0);
    }

    #[test]
    fn a_span_reads_a_number_a_word_or_a_duration_and_writes_it_back() {
        for (json, value, back) in [
            ("600", 600.0, "600.0"),
            ("\"inf\"", f64::INFINITY, "\"inf\""),
            ("\"10m\"", 600.0, "\"10m\""),
            ("\"90s\"", 90.0, "\"90s\""),
        ] {
            let span: Span = serde_json::from_str(json).unwrap();
            assert_eq!(span.value(), value, "{json}");
            assert_eq!(serde_json::to_string(&span).unwrap(), back, "{json}");
        }
        let err = serde_json::from_str::<Span>("\"10\"")
            .unwrap_err()
            .to_string();
        assert!(err.contains("has no unit"), "{err}");
    }

    #[test]
    fn a_list_reads_one_value_or_several() {
        let one: SpanList = serde_json::from_str("\"5m\"").unwrap();
        assert_eq!(one.to_vec(), vec![300.0]);
        let grid: SpanList = serde_json::from_str("[\"5m\", \"30m\"]").unwrap();
        assert_eq!(grid.to_vec(), vec![300.0, 1_800.0]);
        assert_eq!(
            grid.spans().iter().map(Span::label).collect::<Vec<_>>(),
            vec!["5m", "30m"]
        );
    }

    #[test]
    fn zero_and_infinity_bind_to_no_unit() {
        assert!(!Span::Units(0.0).is_unit_bound_number());
        assert!(!Span::Units(f64::INFINITY).is_unit_bound_number());
        assert!(Span::Units(600.0).is_unit_bound_number());
        assert!(!Span::Duration(Duration::parse("10m").unwrap()).is_unit_bound_number());
    }
}
