//! The readiness warning that `min_weight` has not been met (WRITING §11),
//! split from `stream.rs` to keep it under the repository's file-size cap.

use online_core::Decay;

use super::{half_life_of, projected_figure};
use crate::spec::Spec;

/// Where a readiness warning sends its reader: the reference section on
/// the gates, which a message keeps linking once it has shipped.
const READY_LINK: &str =
    "https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready";

/// What a "min_weight has not been met" notice projects from: the weight
/// the target's rows top out at, at their spacing and weight so far
/// ([`settled_weight`]), the weight the gate read on the row, and the row's
/// own weight.
pub(super) struct Projection {
    pub(super) ceiling: f64,
    pub(super) weight: f64,
    pub(super) row_weight: f64,
}

/// The warning that `min_weight` has not been met for a half-life, with the
/// model within 5% of its steady state, in the shape every message takes
/// (WRITING §11): what has not been met, what that does to the output, and
/// what to change, with the reference that explains the gates. Its forms: a
/// target the model has learned no weighted value of; a projection at the
/// rows' spacing and weight so far, with the half-life (or `lam`) at which
/// the projected weight would reach the floor; and, with no figure, rows
/// too uneven to project or a hard window, whose weight does not settle as
/// the decay does. A model that predicts no target names none. A floor the
/// bank raised above the spec's (`ew_cov`'s two rows) is named as raised,
/// and lowering `min_weight` is not offered for it. A spec with several
/// half-lives names the instance.
pub(super) fn unmet_floor_notice(
    spec: &Spec,
    decay: Decay,
    tj: usize,
    floor: f64,
    at: Projection,
) -> String {
    use crate::figures::{Round, three_figures};
    let named = (!spec.model.predicts_no_target())
        .then(|| spec.targets.get(tj))
        .flatten();
    let instance = match spec.decays() {
        Ok(d) if d.len() > 1 => format!(
            "the instance for half_life {} (the part of the model that fits that half-life): ",
            crate::bank::clock_amount(spec, half_life_of(decay))
        ),
        _ => String::new(),
    };
    let given = spec
        .min_periods_per_target()
        .get(tj)
        .copied()
        .unwrap_or(floor);
    let raised = floor > given;
    let setting = if raised {
        format!("min_weight = {given}, lifted to {floor}, the least weight a covariance needs,")
    } else {
        format!("min_weight = {floor}")
    };
    let target = named.map_or_else(String::new, |t| format!(" for target {t:?}"));
    let unmet = format!(
        "{instance}{setting} has not been met{target} on any row for the last half-life, with the \
         model within 5% of its steady state"
    );
    let withheld = named.map_or_else(
        || "the model's output stays withheld".to_string(),
        |t| format!("predictions for target {t:?} stay withheld"),
    );
    let decay_name = match decay {
        Decay::Lam(_) => "lam",
        Decay::Halflife(_) => "half_life",
    };
    if at.weight == 0.0 {
        let lacks = named.map_or_else(
            || "no row with a positive weight".to_string(),
            |t| format!("no row with a value of target {t:?} and a positive weight"),
        );
        return format!(
            "{unmet}. The model has learned {lacks}, so {withheld} until such rows bring its weight \
             to {floor}. Check that rows reach the model with {}a positive weight, as {READY_LINK} \
             explains.",
            named.map_or("", |_| "a value of the target and ")
        );
    }
    let windowed = spec
        .model
        .window_parts()
        .is_some_and(|(window, _)| window.is_some());
    let figure = (!windowed)
        .then(|| projected_figure(at.ceiling))
        .flatten()
        .map(|_| three_figures(at.ceiling, Round::Down));
    let Some(figure) = figure else {
        let why = if windowed {
            "The hard window limits the weight, in addition to the decay, so the warning cannot say \
             where the weight tops out"
        } else {
            "The rows so far are too uneven (such as a target present on only its first row, or \
             row weights far apart) to project where the weight the gate reads tops out"
        };
        let more = if windowed {
            format!("{decay_name} or window_size")
        } else {
            decay_name.to_string()
        };
        let remedy = if raised {
            format!("Increase {more}")
        } else {
            format!("Lower min_weight, or increase {more}")
        };
        return format!(
            "{unmet}, so {withheld} until the weight the gate reads reaches {floor}. {why}. {remedy}, as \
             {READY_LINK} explains."
        );
    };
    let (rate, faster) = if spec.clock.is_some() {
        ("row rate", "the rows come faster or carry more weight")
    } else {
        ("row weights", "the rows carry more weight")
    };
    let to =
        floor_reaching(spec, decay, &at, floor).map_or_else(String::new, |v| format!(" above {v}"));
    let remedy = if raised {
        format!("Increase {decay_name}{to}")
    } else {
        format!("Lower min_weight below {figure}, or increase {decay_name}{to}")
    };
    format!(
        "{unmet}. At the {rate} seen so far, the weight the gate reads tops out near {figure}, so \
         {withheld} unless {faster}. {remedy}, as {READY_LINK} explains."
    )
}

/// The half-life (or `lam`) above which the projected weight reaches
/// `floor`, at the rows' spacing and weight so far, in the spec's own terms,
/// rounded up so the remedy holds; `None` where none does. The ceiling
/// `C = w/(1 − 2^(−d/h))` gives the spacing in half-lives,
/// `d/h = −log2(1 − w/C)`, and the floor `N` is reached where
/// `h > d/−log2(1 − w/N)`.
fn floor_reaching(spec: &Spec, decay: Decay, at: &Projection, floor: f64) -> Option<String> {
    use crate::figures::{Round, three_figures};
    let h = half_life_of(decay);
    let w = at.row_weight;
    let d = h * -(1.0 - w / at.ceiling).log2();
    let need = d / -(1.0 - w / floor).log2();
    if !(need.is_finite() && need > 0.0 && w < floor) {
        return None;
    }
    Some(match decay {
        // A factor near 1 keeps the decimals that tell it from 1 (WRITING
        // §11).
        Decay::Lam(_) => {
            // `lam` above the threshold: its distance from 1 to three
            // figures, rounded down, so `lam` rounds up and stays below 1.
            let gap: f64 = three_figures(1.0 - (-1.0 / need).exp2(), Round::Down)
                .parse()
                .ok()?;
            let decimals = usize::try_from(2 - gap.log10().floor() as i32).unwrap_or(0);
            crate::figures::trim(&format!("{:.decimals$}", 1.0 - gap)).to_string()
        }
        Decay::Halflife(_) => {
            let up: f64 = three_figures(need, Round::Up).parse().ok()?;
            crate::bank::clock_amount(spec, up)
        }
    })
}
