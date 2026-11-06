//! Numbers as a message prints them: at most three significant figures
//! (docs/WRITING.md §11), with no trailing zero, in plain notation from 0.001 to below a million
//! and in exponent notation outside it. A figure a remedy sets a bound by
//! rounds toward the side that keeps the remedy true: a value to stay below
//! rounds down, and one to pass rounds up.

/// Which way a figure rounds to its three significant figures.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Round {
    Down,
    Up,
}

/// `x` to three significant figures, rounded `how`.
pub(crate) fn three_figures(x: f64, how: Round) -> String {
    if !x.is_finite() || x == 0.0 {
        return format!("{x}");
    }
    // The exponent of the leading digit, taken again after rounding: 9.996
    // rounds to 10.0, a decade up, and keeps three figures there.
    let lead = |v: f64| v.abs().log10().floor() as i32;
    let round = |e: i32| {
        let unit = 10f64.powi(e - 2);
        // The quotient's last bits are the division's, not the value's: 3e-7
        // over 1e-9 is 299.99999999999994. Rounded to nine decimals first,
        // a value on a figure stays on it before the floor or the ceiling.
        let q = ((x / unit) * 1e9).round() / 1e9;
        let q = match how {
            Round::Down => q.floor(),
            Round::Up => q.ceil(),
        };
        (q * unit, unit)
    };
    let mut e = lead(x);
    let (mut r, mut unit) = round(e);
    if r != 0.0 && lead(r) != e {
        e = lead(r);
        (r, unit) = round(e);
    }
    let text = |r: f64, e: i32| {
        if !(-3..6).contains(&e) {
            let s = format!("{r:.2e}");
            let (m, p) = s.split_once('e').expect("an exponent");
            format!("{}e{p}", trim(m))
        } else {
            let decimals = usize::try_from(2 - e).unwrap_or(0);
            trim(&format!("{r:.decimals$}")).to_string()
        }
    };
    // The snap above can carry a value just under a figure onto it:
    // 0.8999999999999999 reads 0.9. As printed, a figure rounded down is
    // never above the value, and one rounded up never below it.
    let shown = text(r, e);
    let read: f64 = shown.parse().unwrap_or(r);
    match how {
        Round::Down if read > x => {
            // One unit down, in the decade below where that crosses one:
            // 9.999999999999 reads 9.99, not 9.9.
            let below = r - unit;
            if lead(below) < e {
                text(below + unit - unit / 10.0, e - 1)
            } else {
                text(below, e)
            }
        }
        Round::Up if read < x => text(r + unit, e),
        _ => shown,
    }
}

/// A decimal with its trailing zeros, and a bare point, dropped.
pub(crate) fn trim(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_figures_in_every_range_and_direction() {
        assert_eq!(three_figures(7.72497, Round::Down), "7.72");
        assert_eq!(three_figures(7.7299, Round::Down), "7.72");
        assert_eq!(three_figures(7.7201, Round::Up), "7.73");
        assert_eq!(three_figures(9.996, Round::Up), "10");
        assert_eq!(three_figures(0.9996, Round::Up), "1");
        assert_eq!(three_figures(9.996, Round::Down), "9.99");
        assert_eq!(three_figures(10.0, Round::Down), "10");
        assert_eq!(three_figures(7.7, Round::Down), "7.7");
        assert_eq!(three_figures(12345.0, Round::Down), "12300");
        assert_eq!(three_figures(0.0012345, Round::Down), "0.00123");
        assert_eq!(three_figures(3e-7, Round::Down), "3e-7");
        assert_eq!(three_figures(1.2345e-7, Round::Down), "1.23e-7");
        assert_eq!(three_figures(-0.12345, Round::Down), "-0.124");
        assert_eq!(three_figures(0.0, Round::Down), "0");
        assert_eq!(three_figures(0.3 * 3.0, Round::Down), "0.899");
        assert_eq!(three_figures(9.999999999999, Round::Down), "9.99");
        assert_eq!(three_figures(0.9999999999999, Round::Down), "0.999");
    }
}
