//! A float in a spec must survive the crossing into Rust bit-for-bit.
//!
//! Specs reach the Rust side as JSON. `serde_json`'s default float parser is
//! fast rather than correctly rounded, and lands up to one ulp away from the
//! value Python wrote -- `-0.41215148805088475` came back as
//! `-0.4121514880508847`. The crate ships a `float_roundtrip` feature that
//! parses exactly; this crate and the two around it enable it.
//!
//! One ulp is nothing for a halflife, and everything for a bin edge. The
//! edges a caller passes to `marginal(bin_edges=)` are usually quantiles read
//! back from an earlier run, which means they are *data values* -- so a row
//! sitting exactly on an edge is the common case, not a coincidence, and a
//! last-bit shift moves it into the neighbouring bin. That is how this was
//! found: a replay that should have matched to the bit was off by one row in
//! two adjacent bins.

use online_polars::Spec;

/// Values whose shortest exact decimal needs all 17 significant digits, which
/// is where a fast parser gives up the last bit.
const AWKWARD: [f64; 5] = [
    -0.41215148805088475,
    0.10000000000000002,
    1.7976931348623157e308,
    5e-324,
    0.30000000000000004,
];

#[test]
fn a_spec_float_crosses_into_rust_unchanged() {
    for v in AWKWARD {
        let json = format!(
            r#"[{{"name":"m","model":{{"type":"marginal","bin_edges":[[{v:?}]]}},
                 "targets":["y"],"features":["x"],"halflife":{v:?}}}]"#
        );
        let specs: Vec<Spec> =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("{v:?} did not parse: {e}"));
        let text = format!("{:?}", specs[0]);
        assert!(
            text.contains(&format!("{v:?}")),
            "{v:?} was changed crossing into Rust: {text}"
        );
    }
}
