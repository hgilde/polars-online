//! The checks of single spec parameters that more than one kind shares,
//! split from `spec.rs` to keep it under the repository's file-size cap.

use super::{HUBER_SCALES, Num, Span, spec_words_refusal};

/// A spec's refusal, every problem its checks found, in one message
/// (WRITING §11: report every problem where finding them all is cheap). One
/// problem is the message it was written as, `spec "m": ...`. Several are
/// listed one to a line under one `spec "m": N problems.`, each without the
/// spec's name, since the first line gives it. A model's own check names
/// its kind (`rcov: kernel ...`), which the spec already says, so that is
/// dropped too. A check that stopped early, because the checks after it read
/// what a problem refuses, says so on a line of its own, last.
pub fn render_refusal(spec: &str, kind: &str, no: &online_core::Refusals) -> Result<(), String> {
    if no.is_empty() {
        return Ok(());
    }
    let prefix = format!("spec {spec:?}: ");
    let model = format!("{kind}: ");
    let mut lines: Vec<&str> = Vec::new();
    for p in no.problems() {
        let p = p.strip_prefix(prefix.as_str()).unwrap_or(p);
        let p = p.strip_prefix(model.as_str()).unwrap_or(p);
        // The same problem reached by two checks is said once.
        if !lines.contains(&p) {
            lines.push(p);
        }
    }
    let mut out = match lines.as_slice() {
        [one] => format!("{prefix}{one}"),
        many => {
            let mut s = format!("{prefix}{} problems.", many.len());
            for l in many {
                s.push_str("\n- ");
                s.push_str(l);
            }
            s
        }
    };
    if no.stopped() {
        out.push('\n');
        out.push_str(stopped_line(lines.len()));
    }
    Err(out)
}

/// The last line of a refusal whose checks stopped before they had read
/// every setting, for one problem or several.
pub fn stopped_line(problems: usize) -> &'static str {
    if problems == 1 {
        "Fix this, then run again to check the spec's other settings."
    } else {
        "Fix these, then run again to check the spec's other settings."
    }
}

/// `huber_scale` one of [`HUBER_SCALES`] and `huber_scale_half_life` above
/// 0, `"inf"` included (no forgetting), as `trend_half_life` is held.
pub(super) fn check_huber_scale(
    name: &str,
    scale: Option<&str>,
    half_life: Option<&Span>,
) -> Result<(), String> {
    let mut no = online_core::Refusals::new();
    if let Some(w) = scale
        && !HUBER_SCALES.contains(&w)
    {
        no.refuse(spec_words_refusal(name, "huber_scale", &HUBER_SCALES, w));
    }
    if let Some(h) = half_life.filter(|h| h.value() <= 0.0 || h.value().is_nan()) {
        no.refuse(format!(
            "spec {name:?}: huber_scale_half_life must be above 0, but it is {h}. Set \
             huber_scale_half_life to a positive number of clock units, or to \"inf\" to forget \
             no residual, or leave it out to use the spec's half_life."
        ));
    }
    no.finish()
}

/// `v > 0` as a named predicate, so that `!positive(v)` refuses NaN too (a
/// plain `v <= 0.0` lets it through, and NaN in any of these parameters is
/// a state that never washes out).
pub(super) fn positive(v: f64) -> bool {
    v > 0.0
}

pub(super) fn non_negative(v: f64) -> bool {
    v >= 0.0
}

/// Which values a range refusal reports: the parameter's one value, or
/// every bad value of a list, each as the spec gave it. A list whose values
/// must differ from each other, a grid's, says so in its remedy.
pub(crate) enum Got {
    One(String),
    Of(Vec<String>),
    /// The bad values, and whether the list holds good ones beside them.
    Distinct(Vec<String>, bool),
}

impl Got {
    /// A grid's bad values (`bad`), or its one value where the parameter
    /// may be given as a single number (`scalar`) and the list holds one.
    pub(crate) fn list(all: usize, bad: Vec<String>, scalar: bool) -> Self {
        if scalar && all == 1 {
            Got::One(bad.into_iter().next().unwrap_or_default())
        } else {
            let others = all > bad.len();
            Got::Distinct(bad, others)
        }
    }
}

/// The refusal of a number a spec takes as finite and at least 0 (and, with
/// `below`, less than it), in the shape every message takes (WRITING §11):
/// where it happened, what is wrong and the values it got, then the remedy
/// as a sentence of its own. `who` is where (`spec "m"`, or `with_windows`),
/// `unit` the units the number is in (`" clock units"`, or empty), and
/// `zero` what 0 does, where a value of 0 means something a reader would set
/// it for. A list reports every bad value at once. One check, so every
/// parameter it serves says the same thing in the same words.
pub(crate) fn at_least_zero(
    who: &str,
    key: &str,
    unit: &str,
    got: &Got,
    zero: Option<&str>,
    below: Option<f64>,
) -> String {
    let range = below.map_or_else(
        || format!("finite and at least 0{unit}"),
        |b| format!("at least 0 and below {}{unit}", super::num_label(b)),
    );
    let positive = |many: bool| {
        let s = if many { "s" } else { "" };
        below.map_or_else(
            || format!("{}finite positive value{s}", if many { "" } else { "a " }),
            |b| {
                format!(
                    "{}positive value{s} below {}",
                    if many { "" } else { "a " },
                    super::num_label(b)
                )
            },
        )
    };
    let zero = zero.map_or_else(String::new, |z| format!(" {z}"));
    let (is, remedy) = match got {
        Got::One(v) => (
            format!("it is {v}"),
            format!("Set {key} to 0{zero}, or to {}.", positive(false)),
        ),
        Got::Of(v) if v.len() == 1 => (
            format!("one of its values is {}", v[0]),
            format!("Set that value to 0{zero}, or to {}.", positive(false)),
        ),
        Got::Distinct(v, others) if v.len() == 1 => (
            format!("one of its values is {}", v[0]),
            if *others {
                format!(
                    "Set that value to 0{zero} or to {}, different from the other values.",
                    positive(false)
                )
            } else {
                format!("Set that value to 0{zero}, or to {}.", positive(false))
            },
        ),
        Got::Of(v) => (
            format!("its values {} are not", listed(v)),
            format!("Set those values to 0{zero}, or to {}.", positive(true)),
        ),
        Got::Distinct(v, others) => (
            format!("its values {} are not", listed(v)),
            format!(
                "Set each to 0{} or to {}, all different from each other{}.",
                zero.is_empty()
                    .then(String::new)
                    .unwrap_or_else(|| format!(" ({})", zero.trim())),
                positive(false),
                if *others {
                    " and from the other values"
                } else {
                    ""
                },
            ),
        ),
    };
    format!("{who}: {key} must be {range}, but {is}. {remedy}")
}

/// `a, b and c`.
pub(super) fn listed(v: &[String]) -> String {
    match v.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => v.join(""),
    }
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
            "spec {name:?}: refit_budget must be above 0 MiB, but it is {} MiB. Set \
             refit_budget to a positive number of MiB, or to inf for no limit.",
            super::num_label(b.0)
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
    let mut no = online_core::Refusals::new();
    for (key, given) in [
        ("bin_refit_rows", rows_given),
        ("refit_budget", budget.is_some()),
    ] {
        if given && !huber_cut {
            no.refuse(format!(
                "spec {name:?}: {key} applies only under bin_stat=\"huber\" with \
                 bin_huber_delta below inf: under bin_stat=\"mean\", or bin_huber_delta=inf, one \
                 pass over the rows already gives the refit's result. Set bin_stat=\"huber\" with \
                 bin_huber_delta below inf, or leave {key} out."
            ));
        }
    }
    if huber_cut {
        no.take(check_refit_budget(name, budget));
    }
    no.finish()
}

/// What a `ridge` of 0 does, as a range refusal says it.
pub(super) const NO_SHRINKAGE: &str = "for no shrinkage";

/// `ridge` for the IRLS models: finite and non-negative (zero is plain least
/// squares).
pub(super) fn check_ridge(name: &str, ridge: Option<f64>) -> Result<(), String> {
    if let Some(r) = ridge.filter(|r| !non_negative(*r) || !r.is_finite()) {
        return Err(at_least_zero(
            &format!("spec {name:?}"),
            "ridge",
            "",
            &Got::One(super::num_label(r)),
            Some(NO_SHRINKAGE),
            None,
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
            "spec {name:?}: solve_every must be at least 0 clock units, but it is {v}. Set \
             solve_every to 0 to solve on every row, to a positive number of clock units, or to \
             \"inf\" for no solve on the clock, or leave it out for the default cadence."
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
            "spec {spec:?}: feature_sets must name at least one set, but it names none. Leave \
             feature_sets out to fit every feature."
        ));
    }
    let mut no = online_core::Refusals::new();
    let mut names = std::collections::HashSet::new();
    for (name, cols) in fs {
        // A name twice renders two slots under one field name, which only
        // the bank's tripwire caught (S7).
        if !names.insert(name.as_str()) {
            no.refuse(format!(
                "spec {spec:?}: feature_sets names {name:?} more than once, but a set's name \
                 is part of its output fields' names. Rename all but one of them."
            ));
        }
        if cols.is_empty() {
            no.refuse(format!(
                "spec {spec:?}: feature set {name:?} lists no feature. List at least one feature \
                 in it, or remove the set."
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for c in cols {
            if !features.contains(c) {
                no.refuse(format!(
                    "spec {spec:?}: feature set {name:?} lists {c:?}, which is not one of \
                     features. Add {c:?} to features, or remove it from the set."
                ));
            }
            // A column twice split its coefficient across two identical
            // slots, without a word (S7).
            else if !seen.insert(c.as_str()) {
                no.refuse(format!(
                    "spec {spec:?}: feature set {name:?} lists {c:?} more than once. List each \
                     feature once in a set."
                ));
            }
        }
    }
    no.finish()
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
            assert!(
                e.contains("ridge must be finite and at least 0"),
                "{bad}: {e}"
            );
        }
        assert!(check_refit_budget("m", Some(Num(f64::INFINITY))).is_ok());
        for bad in [0.0, -1.0, f64::NAN] {
            assert!(check_refit_budget("m", Some(Num(bad))).is_err(), "{bad}");
        }
        assert_eq!(first_duplicate(&[1.0, 2.0, -0.0, 0.0]), Some(0.0));
        assert_eq!(first_duplicate(&[1.0, 2.0]), None);
    }
}
