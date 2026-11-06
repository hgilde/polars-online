//! The readiness warning that `min_weight` has not been met (WRITING §11),
//! split from `stream.rs` to keep it under the repository's file-size cap.

use online_core::Decay;

use super::{half_life_of, projected_figure, settled_weight};
use crate::spec::Spec;

/// Where a readiness warning sends its reader: the reference section on
/// the gates, which a message keeps linking once it has shipped.
const READY_LINK: &str =
    "https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready";

/// What a "min_weight has not been met" notice projects from: the weight
/// the target's rows top out at, at their spacing and weight so far
/// ([`settled_weight`]), the weight the gate read on the row, and the row's
/// own weight; and, for a model that keeps a weight per target, where the
/// shared weight tops out and whether the target's is short of it.
///
/// The projection reads one weight at one time and takes it for its steady
/// level, which it is on a stream whose rows each carry the target. A
/// target the rows stopped carrying is not at a steady level: its weight
/// only decays, toward 0, and the projection read that falling weight as a
/// ceiling (task 350). One decay's running sums at one time cannot tell a
/// target that stopped from one that comes on a share of the rows; that
/// takes the weight at a second time, which the state does not keep. But
/// the shared weight -- every row the model learned, the target's or not,
/// at the same decay -- tells either from a target every row carries: the
/// target's weight is below it. Where it is, the most the target's weight
/// can reach at the rows' spacing and weight so far is the shared weight's
/// ceiling, and the notice says the target is short of the rows rather than
/// where its own weight tops out.
pub(super) struct Projection {
    pub(super) ceiling: f64,
    pub(super) weight: f64,
    pub(super) row_weight: f64,
    /// Where the shared weight tops out ([`settled_weight`]): what a target
    /// every row carried would read. The target's own `ceiling` where the
    /// model keeps only the shared weight.
    pub(super) shared: f64,
    /// Whether the target's weight is below the shared weight by more than
    /// the two carry in rounding: the rows the model learned did not all
    /// carry the target.
    pub(super) short: bool,
}

impl Projection {
    /// The projection for a target whose weight is `weight` and whose own
    /// ceiling is `ceiling`, beside the shared weight `shared` where the
    /// model keeps a weight per target, on a row of weight `row_weight` at
    /// the settled fraction `settled`, each weight carried through at most
    /// `steps` decay steps.
    ///
    /// Two weights the same rows built are equal but for rounding: each is
    /// within `4u·n` of itself after `n` steps ([`settled_weight`]), so they
    /// differ by at most `4u·n·(S + T)`, and their difference rounds by `u`
    /// of `S`. A target is short where `S − T` passes `8u·n·(S + T)`, which
    /// one row it missed passes until that row's weight has decayed below
    /// it.
    pub(super) fn read(
        ceiling: f64,
        weight: f64,
        shared: Option<f64>,
        row_weight: f64,
        settled: f64,
        steps: f64,
    ) -> Self {
        let (shared, short) = match shared {
            Some(s) => {
                let noise = 8.0 * (f64::EPSILON / 2.0) * steps.max(1.0) * (s.abs() + weight.abs());
                (
                    settled_weight(s, row_weight, settled, steps),
                    s - weight > noise,
                )
            }
            None => (ceiling, false),
        };
        Self {
            ceiling,
            weight,
            row_weight,
            shared,
            short,
        }
    }
}

/// The warning that `min_weight` has not been met for a half-life, with the
/// model within 5% of its steady state, in the shape every message takes
/// (WRITING §11): what has not been met, what that does to the output, and
/// what to change, with the reference that explains the gates. Its forms:
///
/// - a target the model has learned no weighted value of;
/// - a hard window, whose weight does not settle as the decay does: no
///   figure;
/// - a row of weight 0, which adds nothing, so no regular stream of rows
///   like it reads the weight there is: no figure;
/// - a target the model has not learned from every row
///   ([`Projection::short`]): where rows that each carried it would reach
///   the floor, no figure, as its weight falls on each row without it;
///   where they would not, the shared weight's ceiling and the half-life (or
///   `lam`) at which it would reach the floor;
/// - a target every row carries: its projection at the rows' spacing and
///   weight so far, with that half-life (or `lam`);
/// - and, with no figure, rows too uneven to project.
///
/// A model that predicts no target names none. A floor the bank raised
/// above the spec's (`ew_cov`'s two rows) is named as raised, and lowering
/// `min_weight` is not offered for it. A spec with several half-lives names
/// the instance.
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
    if windowed {
        let remedy = if raised {
            format!("Increase {decay_name} or window_size")
        } else {
            format!("Lower min_weight, or increase {decay_name} or window_size")
        };
        return format!(
            "{unmet}, so {withheld} until the weight the gate reads reaches {floor}. The hard window \
             limits the weight, in addition to the decay, so the warning cannot say where the \
             weight tops out. {remedy}, as {READY_LINK} explains."
        );
    }
    if at.row_weight <= 0.0 {
        let remedy = if raised {
            format!(
                "Check that rows reach the model with a positive weight, or increase {decay_name}"
            )
        } else {
            format!(
                "Check that rows reach the model with a positive weight, lower min_weight, or \
                 increase {decay_name}"
            )
        };
        return format!(
            "{unmet}, so {withheld} until the weight the gate reads reaches {floor}. This warning \
             comes on a row of weight 0, which adds nothing to the weight, so it cannot say where \
             the weight tops out. {remedy}, as {READY_LINK} explains."
        );
    }
    // A target the model has not learned from every row: the rows' own
    // ceiling is the shared weight's, and the target's weight falls on each
    // row it misses.
    let short = named.filter(|_| at.short);
    if let Some(t) = short
        && projected_figure(at.shared).is_some()
        && at.shared >= floor
    {
        return format!(
            "{unmet}. The model has not learned target {t:?} from every row, and its weight falls \
             on each row without it, so {withheld} until rows with a value of the target bring \
             that weight to {floor}. Check that rows reach the model with a value of target \
             {t:?}, as {READY_LINK} explains."
        );
    }
    let ceiling = if short.is_some() {
        at.shared
    } else {
        at.ceiling
    };
    let Some(figure) = projected_figure(ceiling).map(|_| three_figures(ceiling, Round::Down))
    else {
        let remedy = if raised {
            format!("Increase {decay_name}")
        } else {
            format!("Lower min_weight, or increase {decay_name}")
        };
        return format!(
            "{unmet}, so {withheld} until the weight the gate reads reaches {floor}. The rows so far \
             are too uneven (such as one row far heavier than the rest) to project where the \
             weight the gate reads tops out. {remedy}, as {READY_LINK} explains."
        );
    };
    let (rate, faster) = if spec.clock.is_some() {
        ("row rate", "the rows come faster or carry more weight")
    } else {
        ("row weights", "the rows carry more weight")
    };
    let to = floor_reaching(spec, decay, ceiling, at.row_weight, floor)
        .map_or_else(String::new, |v| format!(" above {v}"));
    let remedy = if raised {
        format!("Increase {decay_name}{to}")
    } else {
        format!("Lower min_weight below {figure}, or increase {decay_name}{to}")
    };
    let (every, carries) = short.map_or_else(
        || (String::new(), String::new()),
        |t| {
            (
                format!(
                    " if every row carries a value of target {t:?}, and the model has not learned \
                     the target from every row"
                ),
                ", and each carries a value of the target".to_string(),
            )
        },
    );
    format!(
        "{unmet}. At the {rate} seen so far, the weight the gate reads tops out near {figure}{every}, \
         so {withheld} unless {faster}{carries}. {remedy}, as {READY_LINK} explains."
    )
}

/// The half-life (or `lam`) above which the projected weight reaches
/// `floor`, at the rows' spacing and weight so far, in the spec's own terms,
/// rounded up so the remedy holds; `None` where none does. The ceiling
/// `C = w/(1 − 2^(−d/h))` gives the spacing in half-lives,
/// `d/h = −log2(1 − w/C)`, and the floor `N` is reached where
/// `h > d/−log2(1 − w/N)`, for rows of weight `w` whose weight tops out at
/// `ceiling`.
fn floor_reaching(spec: &Spec, decay: Decay, ceiling: f64, w: f64, floor: f64) -> Option<String> {
    use crate::figures::{Round, three_figures};
    let h = half_life_of(decay);
    let d = h * -(1.0 - w / ceiling).log2();
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
