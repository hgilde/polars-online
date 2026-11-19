use super::num_label;

#[test]
fn ordinary_values_render_exactly_as_before() {
    // The compact form must not change any name the suite already pins.
    assert_eq!(num_label(1e-6), "0.000001");
    assert_eq!(num_label(1e-7 * 10.0), "0.000001");
    assert_eq!(num_label(0.1), "0.1");
    assert_eq!(num_label(0.5), "0.5");
    assert_eq!(num_label(100.0), "100");
    assert_eq!(num_label(250.5), "250.5");
    assert_eq!(num_label(3200.0), "3200");
    assert_eq!(num_label(0.0), "0");
    // One number, one name (review 2026-10-06, PC10).
    assert_eq!(num_label(-0.0), "0");
}

#[test]
fn extreme_values_render_compactly() {
    // The bug this exists for: 1e-300 was a 311-character field name.
    assert_eq!(num_label(1e-300), "1e-300");
    assert_eq!(num_label(2.5e8), "2.5e8");
    assert_eq!(num_label(1e7), "1e7");
    assert_eq!(num_label(1e9), "1e9");
    assert_eq!(num_label(-1e-300), "-1e-300");
    assert!(num_label(1e-300).len() < 10);
}

#[test]
fn the_thresholds_are_where_the_doc_says() {
    // Just inside: plain. Just outside: scientific.
    assert_eq!(num_label(1e-6), "0.000001");
    assert_eq!(num_label(9.999e-7), "9.999e-7");
    assert_eq!(num_label(9_999_999.0), "9999999");
    assert_eq!(num_label(1e7), "1e7");
}
