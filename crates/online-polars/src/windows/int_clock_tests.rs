//! An integer clock's short cuts, which skip the 128-bit conversions, are
//! the exact comparisons they stand for, bit for bit, for every pair of
//! values the column can hold (extremes included) and every cap or span.
use proptest::prelude::*;

use super::*;

/// 64-bit values that gather where the short cuts change arm: the extremes,
/// the neighbourhoods of 0 and of 2^53, and anything.
fn clocks() -> impl Strategy<Value = i64> {
    let edge = 1i64 << 53;
    prop_oneof![
        3 => any::<i64>(),
        1 => Just(i64::MIN),
        1 => Just(i64::MAX),
        2 => -4i64..4,
        2 => (-4i64..4).prop_map(move |d| edge + d),
        2 => (-4i64..4).prop_map(move |d| -edge + d),
        1 => (-4i64..4).prop_map(|d| i64::MAX - 4 + d.abs()),
        1 => (-4i64..4).prop_map(|d| i64::MIN + d.abs()),
    ]
}

/// Doubles that gather where a comparison with a step can go either way.
fn caps() -> impl Strategy<Value = f64> {
    let edge = (1u64 << 53) as f64;
    prop_oneof![
        3 => any::<f64>(),
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(0.0),
        1 => Just(-0.0),
        2 => (-5i32..5).prop_map(|d| f64::from(d) / 2.0),
        2 => (-4i32..4).prop_map(move |d| edge + f64::from(d) * 2.0),
        2 => (-4i32..4).prop_map(move |d| -edge + f64::from(d) * 2.0),
        1 => Just(f64::MAX),
        1 => Just(f64::MIN),
        1 => Just(1.7e38),
        1 => Just(9.3e18),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20_000))]

    #[test]
    fn the_integer_elapsed_is_the_128_bit_step_rounded_once(c in clocks(), p in clocks()) {
        let want = (i128::from(c) - i128::from(p)) as f64;
        let got = elapsed(ClockValue::I64(c), ClockValue::I64(p));
        prop_assert_eq!(got.to_bits(), want.to_bits());
        prop_assert_eq!(int_step_f64(c, p).to_bits(), want.to_bits());
    }

    #[test]
    fn the_integer_past_cap_is_the_exact_comparison(c in clocks(), p in clocks(), cap in caps()) {
        let step = i128::from(c) - i128::from(p);
        let want = online_core::cmp_int_f64(step, cap) == Some(Ordering::Greater);
        prop_assert_eq!(past_cap(ClockValue::I64(c), ClockValue::I64(p), cap), want);
    }

    #[test]
    fn the_integer_is_after_is_the_sign_of_the_step(c in clocks(), p in clocks()) {
        let (now, last) = (ClockValue::I64(c), ClockValue::I64(p));
        prop_assert_eq!(is_after(now, last), elapsed(now, last) > 0.0);
        prop_assert_eq!(is_after(now, last), i128::from(c) > i128::from(p));
    }

    #[test]
    fn the_integer_gap_against_a_span_is_the_exact_comparison(c in clocks(), p in clocks(), span in caps()) {
        let n = i128::from(c) - i128::from(p);
        let want = Stamp::Int(n, 0.0).cmp_span_ns(Stamp::Int(0, 0.0), span, None);
        prop_assert_eq!(Gap::Int(n).cmp_span(span, None), want);
        prop_assert_eq!(Gap::Int(n).cmp_span(span, Some(5)), want);
    }
}
