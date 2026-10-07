//! How a refusal names a size against the budget it is over.
//!
//! A refusal printed the size in GiB to two decimals beside a budget in MiB,
//! so a small budget read "would need 0.00 GiB, over the 0.000001 MiB"
//! (review 2026-10-06, CD16). One unit, MiB, and enough digits.

/// `mib`, a size over `budget` (both in MiB), with enough digits to read as
/// over it: three significant figures, or as many more decimals as it takes
/// for the figure shown to exceed the budget, so a size just past the budget
/// is not shown equal to it and a small one is not shown as 0.
pub(crate) fn mib_over(mib: f64, budget: f64) -> String {
    for decimals in 0..=20 {
        let text = format!("{mib:.decimals$}");
        let significant = text
            .trim_start_matches(['0', '.'])
            .chars()
            .filter(char::is_ascii_digit)
            .count();
        if significant >= 3 && text.parse::<f64>().is_ok_and(|shown| shown > budget) {
            return text;
        }
    }
    format!("{mib}")
}

#[cfg(test)]
mod tests {
    use super::mib_over;

    #[test]
    fn a_size_reads_above_its_budget_in_one_unit() {
        // 80 bytes against a millionth of a MiB: not "0.00".
        assert_eq!(mib_over(80.0 / f64::from(1 << 20), 1e-6), "0.0000763");
        // Three figures where they are enough.
        assert_eq!(mib_over(838.86, 256.0), "839");
        assert_eq!(mib_over(1.5, 1.0), "1.50");
        assert_eq!(mib_over(12.34, 8.0), "12.3");
        // A size a few bytes past the budget is shown past it.
        let just = 256.0 + 8.0 / f64::from(1 << 20);
        let shown = mib_over(just, 256.0);
        assert_eq!(shown, "256.00001");
        assert!(shown.parse::<f64>().unwrap() > 256.0);
    }
}
