//! The checks of single spec parameters that more than one kind shares,
//! split from `spec.rs` to keep it under the repository's file-size cap.

use super::{HUBER_SCALES, Num, Span, spec_words_refusal};

/// `huber_scale` one of [`HUBER_SCALES`] and `huber_scale_half_life` above
/// 0, `"inf"` included (no forgetting), as `trend_half_life` is held.
pub(super) fn check_huber_scale(
    name: &str,
    scale: Option<&str>,
    half_life: Option<&Span>,
) -> Result<(), String> {
    if let Some(w) = scale
        && !HUBER_SCALES.contains(&w)
    {
        return Err(spec_words_refusal(name, "huber_scale", &HUBER_SCALES, w));
    }
    if let Some(h) = half_life.filter(|h| h.value() <= 0.0 || h.value().is_nan()) {
        return Err(format!(
            "spec {name:?}: huber_scale_half_life must be > 0 (\"inf\" forgets no residual), got {h}"
        ));
    }
    Ok(())
}

pub(super) fn positive(v: f64) -> bool {
    v > 0.0
}

pub(super) fn non_negative(v: f64) -> bool {
    v >= 0.0
}

/// The first value that appears twice in a grid, if any. Grid entries become
/// field-name suffixes, so a repeated value is always a mistake. Compared
/// as numbers, so `-0.0` repeats `0.0`: bit by bit the two were two
/// instances of one model (review 2026-10-06, PC10).
pub(super) fn first_duplicate(vals: &[f64]) -> Option<f64> {
    vals.iter()
        .enumerate()
        .find(|(i, v)| vals[..*i].iter().any(|u| u == *v))
        .map(|(_, v)| *v)
}

/// A `refit_budget`, where given: above 0 MiB, `inf` for no bound.
pub(super) fn check_refit_budget(name: &str, budget: Option<Num>) -> Result<(), String> {
    match budget {
        Some(b) if b.0.is_nan() || b.0 <= 0.0 => Err(format!(
            "spec {name:?}: refit_budget must be > 0 MiB (\"inf\" is no bound), got {}",
            b.0
        )),
        _ => Ok(()),
    }
}

/// `marginal`'s bin refit keys (docs/PLAN.md task 278): `bin_refit_rows`
/// and `refit_budget` only for `bin_stat = "huber"` with a finite cut --
/// under the mean, or a cut that clips nothing, the one pass is already the
/// batch fit -- and the budget above 0.
pub(super) fn check_bin_refit(
    name: &str,
    stat: Option<&str>,
    delta: Option<Num>,
    rows_given: bool,
    budget: Option<Num>,
) -> Result<(), String> {
    let huber_cut = stat == Some("huber") && delta.is_none_or(|d| d.0.is_finite());
    for (key, given) in [
        ("bin_refit_rows", rows_given),
        ("refit_budget", budget.is_some()),
    ] {
        if given && !huber_cut {
            return Err(format!(
                "spec {name:?}: marginal {key} is for the refit of bin_stat = \"huber\" with a \
                 bin_huber_delta below inf, and does not apply here: under the mean, or a cut that \
                 clips nothing, the one pass is already the batch fit"
            ));
        }
    }
    check_refit_budget(name, budget)
}

/// `ridge` for the IRLS models: finite and non-negative (zero is plain least
/// squares).
pub(super) fn check_ridge(name: &str, ridge: Option<f64>) -> Result<(), String> {
    if let Some(r) = ridge.filter(|r| !non_negative(*r) || !r.is_finite()) {
        return Err(format!(
            "spec {name:?}: ridge must be finite and >= 0, got {r}"
        ));
    }
    Ok(())
}

/// `solve_every`: clock units between solves; zero solves every row, and
/// `inf` -- or any value at or past `online_core::SOLVE_NEVER`, past the end
/// of any stream -- never (task 278). A negative value silently meant
/// "every row" too, NaN meant "never".
pub(super) fn check_solve_every(name: &str, v: Option<&Span>) -> Result<(), String> {
    if let Some(v) = v.filter(|v| !non_negative(v.value())) {
        return Err(format!(
            "spec {name:?}: solve_every must be >= 0 (0 solves every row, inf never), got {v}"
        ));
    }
    Ok(())
}

/// `feature_sets`' entries, `ewridge`'s and `huber`'s alike: at least one
/// set, each named once, none empty, each naming columns of `features`,
/// each once.
pub(super) fn check_feature_sets(
    spec: &str,
    features: &[String],
    feature_sets: Option<&[(String, Vec<String>)]>,
) -> Result<(), String> {
    let Some(fs) = feature_sets else {
        return Ok(());
    };
    // `[]` is every feature, one set, to the model, and no slot at all to
    // the field names, which rendered none for a model emitting one per
    // ridge (review 2026-09-12, V23).
    if fs.is_empty() {
        return Err(format!(
            "spec {spec:?}: feature_sets names no set; leave it out to fit every feature"
        ));
    }
    let mut names = std::collections::HashSet::new();
    for (name, cols) in fs {
        // A name twice renders two slots under one field name, which only
        // the bank's tripwire caught (S7).
        if !names.insert(name.as_str()) {
            return Err(format!(
                "spec {spec:?}: feature_sets names {name:?} more than once"
            ));
        }
        if cols.is_empty() {
            return Err(format!("spec {spec:?}: feature set {name:?} is empty"));
        }
        let mut seen = std::collections::HashSet::new();
        for c in cols {
            if !features.contains(c) {
                return Err(format!(
                    "spec {spec:?}: feature set {name:?} references unknown feature {c:?}"
                ));
            }
            // A column twice split its coefficient across two identical
            // slots, without a word (S7).
            if !seen.insert(c.as_str()) {
                return Err(format!(
                    "spec {spec:?}: feature set {name:?} lists {c:?} more than once"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_checks_refuse_what_they_name() {
        assert!(positive(1e-300) && !positive(0.0) && !positive(-1.0) && !positive(f64::NAN));
        assert!(non_negative(0.0) && !non_negative(-1e-300) && !non_negative(f64::NAN));
        assert!(check_ridge("m", None).is_ok());
        assert!(check_ridge("m", Some(0.0)).is_ok() && check_ridge("m", Some(2.5)).is_ok());
        for bad in [-1.0, f64::INFINITY, f64::NAN] {
            let e = check_ridge("m", Some(bad)).unwrap_err();
            assert!(e.contains("ridge must be finite and >= 0"), "{bad}: {e}");
        }
        assert!(check_refit_budget("m", Some(Num(f64::INFINITY))).is_ok());
        for bad in [0.0, -1.0, f64::NAN] {
            assert!(check_refit_budget("m", Some(Num(bad))).is_err(), "{bad}");
        }
        assert_eq!(first_duplicate(&[1.0, 2.0, -0.0, 0.0]), Some(0.0));
        assert_eq!(first_duplicate(&[1.0, 2.0]), None);
    }
}
