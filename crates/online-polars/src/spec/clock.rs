//! A spec's clock: the parameters measured in clock units, the scale they
//! are written in, and the clock policy a spec or `with_windows` asks for.
//! Split from `spec.rs` to keep it under the repository's file-size cap.

use online_core::{ClockCfg, OnClockReset, SessionGap};

use super::{ModelKind, SessionGapSpec, Spec, non_negative, positive};
use crate::span::{Span, SpanList};

/// Every parameter measured in clock units, by where it sits: `"*"` for the
/// spec's own, otherwise the model's `type`. Each is a [`Span`] or a
/// [`SpanList`], so each takes a duration under a temporal clock
/// (docs/PLAN.md task 88), and [`Spec::clock_spans`] reads them from here,
/// so a field listed here is one the spec's consistency checks see.
///
/// `clock_fields_are_exactly_the_fields_that_take_a_duration` walks every
/// field serde knows and holds this table to the types: a clock-unit field
/// left off it, or a field on it that refuses a duration, fails the test.
pub const CLOCK_FIELDS: &[(&str, &[&str])] = &[
    (
        "*",
        &[
            "half_life",
            "gap_cap",
            "restart_after_step_back",
            "session_gap",
            "embargo",
            "drift_threshold",
            "coef_every",
            "calibration_half_life",
            "breaks_half_life",
            "robust_se_half_life",
            "specification_half_life",
            "tails_half_life",
            "influence_half_life",
            "feature_health_half_life",
        ],
    ),
    (
        "ewridge",
        &[
            "long_half_life",
            "solve_every",
            "window_size",
            "snapshot_every",
        ],
    ),
    (
        "lasso",
        &[
            "select_half_life",
            "solve_every",
            "window_size",
            "snapshot_every",
        ],
    ),
    ("kalman", &["coef_half_life", "revert_half_life"]),
    ("huber", &["solve_every", "huber_scale_half_life"]),
    ("quantile", &["solve_every"]),
    ("ew_cov", &["pca_every", "window_size", "snapshot_every"]),
    ("holt", &["trend_half_life"]),
    ("sgd", &["huber_scale_half_life"]),
    ("micro", &["prune_every"]),
    ("ew_class", &["window_size", "snapshot_every"]),
    ("marginal", &["window_size", "snapshot_every"]),
];

/// The parameters that take a duration on a temporal clock or a number that
/// binds to no clock unit, each meaning something of its own: `bocpd`'s
/// `hazard`, a number of rows between changepoints on any spec, or a
/// duration, the expected time between them (docs/PLAN.md task 179). A
/// number here is legal beside durations, so these are not
/// [`CLOCK_FIELDS`]; [`Spec::clock_spans`] reads one only when it is a
/// duration, which then meets every rule a clock parameter does.
/// `clock_fields_are_exactly_the_fields_that_take_a_duration` holds the two
/// tables together to the types, and `a_duration_or_a_count_is_read_only_
/// as_a_duration` holds this one to `clock_spans`.
pub const DURATION_OR_UNIT_FREE_FIELDS: &[(&str, &[&str])] = &[("bocpd", &["hazard"])];

/// The parameters that are a rate *per* clock unit: a decay factor per unit
/// (`lam`) and a variance per unit (`kalman`'s `q`). A rate has no duration
/// form, so a temporal clock refuses one; the half-life it stands in for
/// takes a duration instead.
pub const CLOCK_RATES: &[(&str, &[&str])] = &[("*", &["lam"]), ("kalman", &["q"])];

/// What a spec's clock-unit parameters are written in, which decides the
/// clock column it can read (docs/PLAN.md task 88).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockScale {
    /// No parameter is tied to a unit: each is `0`, `inf` or left out. A
    /// numeric clock and a temporal one both read the spec.
    Free,
    /// Plain numbers of clock units, which only a numeric clock (or none)
    /// can give a meaning. Names the first such parameter.
    Numbers(&'static str),
    /// Durations, which only a temporal clock can measure. Names the first.
    Durations(&'static str),
}

/// A clock policy in a caller's own words: a spec's, or `po.stream.with_windows`'
/// (docs/PLAN.md task 78: one set of clock rules, stated once).
pub(crate) struct ClockPolicy<'a> {
    /// How a message names the caller: `spec "name"`, or `windows`.
    pub who: &'a str,
    pub clock: Option<&'a str>,
    pub gap_cap: Option<&'a Span>,
    pub restart_after_step_back: Option<&'a Span>,
    pub session: Option<&'a str>,
    pub session_gap: Option<&'a SessionGapSpec>,
    /// For a spec, whether it closes its groups on a session change, which
    /// stands in for `session_gap`; `None` for a caller with no model, whose
    /// messages speak of its windows.
    pub spec_closes_on_session: Option<bool>,
    /// The group column: each group's windows close on its own clock.
    pub group: Option<&'a str>,
    /// Whether a window looks ahead: a `with_windows` formula's forward
    /// operator, or a spec's formula target, which always holds one.
    pub looks_ahead: bool,
}

/// The [`ClockCfg`] a clock policy asks for, or why it cannot run.
pub(crate) fn clock_cfg_of(p: &ClockPolicy<'_>) -> Result<ClockCfg, String> {
    let who = p.who;
    let model = p.spec_closes_on_session.is_some();
    // Every problem is reported at once; none of these checks reads what
    // another refuses.
    let mut no = online_core::Refusals::new();
    if p.clock.is_some() && p.gap_cap.is_none() {
        no.refuse(if model {
            format!(
                "{who}: gap_cap is required when clock is given, but it is not set. Set gap_cap \
                 to the longest clock step the model should decay across in full."
            )
        } else {
            format!(
                "{who}: gap_cap is required when clock is given, but it is not set. Set gap_cap \
                 to the longest clock step a window should span."
            )
        });
    }
    // The cap is on the clock's step. Without a clock every row is one step,
    // and a cap was taken and applied to it: `gap_cap = 0.5` halved every
    // decay step, marked every row capped and cleared every lag at every row
    // (task 160, PB2). Refused, as `restart_after_step_back` is.
    if p.clock.is_none() && p.gap_cap.is_some() {
        no.refuse(format!(
            "{who}: gap_cap needs clock, but clock is not set: without a clock every row is \
             one step, with nothing to cap. Set clock, or leave gap_cap out."
        ));
    }
    // A window looking ahead closes when a row past its far edge arrives on
    // its group's clock, and on a clock a group silent past `gap_cap` is cut.
    // Without a clock column a group's clock counts only its own rows and
    // there is no cap, so a silent group left its windows open, and
    // `with_windows`, which returns rows in input order, held every later
    // row of every group behind them to the end of the input: 500,001 rows
    // and 117 MiB more over 2M rows, where the docs promise about one
    // window (task 173, PC1). Refused, by `with_windows` and by a spec with
    // a window target alike, the user's call: one rule, one message.
    if p.looks_ahead && p.group.is_some() && p.clock.is_none() {
        no.refuse(if model {
            format!(
                "{who}: a formula target under group needs clock, but clock is not set: \
                 without one, a group that falls silent holds the rows its formula target \
                 waits on until that group's next row. Set clock and gap_cap, or leave group \
                 out."
            )
        } else {
            format!(
                "{who}: a window looking ahead under group needs clock, but clock is not set: \
                 without one, a group that falls silent keeps its windows open, and every later \
                 row waits behind them. Set clock and gap_cap, or leave group out."
            )
        });
    }
    // The cap is a cap on a step and nothing else (task 120, decided
    // 2026-09-28). A negative one clipped every delta to it, so the decay
    // *grew* (`n_eff` ran to 6e7 on 50 rows with `gap_cap = -5`); NaN
    // poisons the clock. Zero froze the clock, and every positive gap
    // then read as a break, so `embargo` released each held label on
    // the next row (since task 153 a break releases nothing early): no
    // decay is `half_life = "inf"`, or no clock. Infinity
    // took the break away, and handed a model an infinite step at a
    // session change.
    if let Some(m) = p.gap_cap {
        let v = m.value();
        if v == 0.0 {
            no.refuse(if model {
                format!(
                    "{who}: gap_cap must be above 0, but it is 0, which stops the clock. Set \
                     gap_cap to a positive number of clock units, or set half_life to inf for \
                     no decay."
                )
            } else {
                format!(
                    "{who}: gap_cap must be above 0, but it is 0, which stops the clock. Set \
                     gap_cap to a positive number of clock units."
                )
            });
        } else if v == f64::INFINITY {
            no.refuse(if model {
                format!(
                    "{who}: gap_cap must be finite, but it is inf: it is the longest step the \
                     model decays across, and a longer step is a clock break. Set gap_cap above \
                     every gap the model should decay across in full."
                )
            } else {
                format!(
                    "{who}: gap_cap must be finite, but it is inf: it is the longest step a \
                     window spans, and a longer step ends every window open across it. Set \
                     gap_cap above every gap a window should span."
                )
            });
        } else if !(v.is_finite() && positive(v)) {
            no.refuse(format!(
                "{who}: gap_cap must be finite and above 0, but it is {m}. Set gap_cap to a \
                 positive number of clock units."
            ));
        }
    }
    let session_gap =
        match p.session_gap {
            None => None,
            Some(SessionGapSpec::Gap(g)) if g.value() == f64::INFINITY => {
                no.refuse(format!(
                "{who}: session_gap must be finite, but it is inf. Set session_gap to the clock \
                 step to take at a session change, or to \"reset\" to reset {} there.",
                if model { "the model" } else { "every group's windows" }
            ));
                None
            }
            Some(SessionGapSpec::Gap(g)) if !(g.value().is_finite() && non_negative(g.value())) => {
                no.refuse(format!(
                    "{who}: session_gap must be at least 0 or \"reset\", but it is {g}. Set \
                 session_gap to a step of 0 or more clock units, or to \"reset\"."
                ));
                None
            }
            Some(SessionGapSpec::Gap(g)) => Some(SessionGap::Gap(g.value())),
            Some(SessionGapSpec::Word(w)) if w == "reset" => Some(SessionGap::Reset),
            Some(SessionGapSpec::Word(w)) => {
                no.refuse(format!(
                    "{who}: session_gap must be a number, a duration or \"reset\", but it is \
                 {w:?}. Set session_gap to a number of clock units, a duration such as \"5m\", \
                 or \"reset\"."
                ));
                None
            }
        };
    let session_gap = match session_gap {
        None if !model && p.session.is_some() => Some(SessionGap::Gap(0.0)),
        given => given,
    };
    if p.session.is_some()
        && p.session_gap.is_none()
        && session_gap.is_none()
        && p.spec_closes_on_session != Some(true)
    {
        no.refuse(format!(
            "{who}: session needs session_gap, but session_gap is not set. Set session_gap to \
             the clock step to take at a session change or to \"reset\", or set \
             group_close=\"session\" to close the group there."
        ));
    }
    if let Some(v) = p
        .restart_after_step_back
        .filter(|v| !(v.value().is_finite() && non_negative(v.value())))
    {
        no.refuse(super::checks::at_least_zero(
            who,
            "restart_after_step_back",
            " clock units",
            &super::checks::Got::One(v.to_string()),
            Some(if model {
                "to reset the model at every step back"
            } else {
                "to reset every group's windows at every step back"
            }),
            None,
        ));
    }
    no.finish()?;
    Ok(ClockCfg {
        // A row-count clock steps by one row and has no cap.
        gap_cap: p.gap_cap.map_or(f64::INFINITY, Span::value),
        on_clock_reset: if p.restart_after_step_back.is_some() {
            OnClockReset::ResetState
        } else {
            OnClockReset::Error
        },
        session_gap,
        min_backwards_jump: p.restart_after_step_back.map_or(0.0, Span::value),
    })
}

impl Spec {
    /// Every clock-unit value this spec sets, with its parameter's name, in
    /// [`CLOCK_FIELDS`] order. A `half_life` grid gives one entry per value.
    /// `every_clock_field_is_walked` holds this to the table.
    pub fn clock_spans(&self) -> Vec<(&'static str, Span)> {
        fn put(out: &mut Vec<(&'static str, Span)>, name: &'static str, s: Option<&Span>) {
            out.extend(s.map(|s| (name, s.clone())));
        }
        fn put_list(out: &mut Vec<(&'static str, Span)>, name: &'static str, l: Option<&SpanList>) {
            if let Some(l) = l {
                out.extend(l.spans().iter().map(|s| (name, s.clone())));
            }
        }
        let mut out = Vec::new();
        put_list(&mut out, "half_life", self.half_life.as_ref());
        put(&mut out, "gap_cap", self.gap_cap.as_ref());
        put(
            &mut out,
            "restart_after_step_back",
            self.restart_after_step_back.as_ref(),
        );
        if let Some(SessionGapSpec::Gap(g)) = &self.session_gap {
            put(&mut out, "session_gap", Some(g));
        }
        put(&mut out, "embargo", self.embargo.as_ref());
        put(&mut out, "drift_threshold", self.drift_threshold.as_ref());
        put(&mut out, "coef_every", self.coef_every.as_ref());
        // The diagnostics' own memories (task 221).
        for (key, _, _, memory) in self.diagnostic_memories() {
            put(&mut out, key, memory);
        }
        // A formula target's operators measure in the same clock (review
        // R1, D7): with them here a number beside durations is refused at
        // the spec, and the embargo check compares like units.
        for t in self.targets.defs() {
            if let Some(tree) = &t.formula {
                for op in tree.operators() {
                    put(&mut out, "half_life", op.half_life.as_ref());
                    put(&mut out, "window_size", op.window_size.as_ref());
                }
            }
        }
        match &self.model {
            ModelKind::EwRidge {
                long_half_life,
                solve_every,
                window_size: window,
                snapshot_every,
                ..
            } => {
                put(&mut out, "long_half_life", long_half_life.as_ref());
                put(&mut out, "solve_every", solve_every.as_ref());
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "snapshot_every", snapshot_every.as_ref());
            }
            ModelKind::Lasso {
                select_half_life,
                solve_every,
                window_size: window,
                snapshot_every,
                ..
            } => {
                put(&mut out, "select_half_life", select_half_life.as_ref());
                put(&mut out, "solve_every", solve_every.as_ref());
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "snapshot_every", snapshot_every.as_ref());
            }
            ModelKind::Kalman {
                coef_half_life,
                revert_half_life,
                ..
            } => {
                put_list(&mut out, "coef_half_life", coef_half_life.as_ref());
                put_list(&mut out, "revert_half_life", revert_half_life.as_ref());
            }
            ModelKind::Huber {
                solve_every,
                huber_scale_half_life,
                ..
            } => {
                put(&mut out, "solve_every", solve_every.as_ref());
                put(
                    &mut out,
                    "huber_scale_half_life",
                    huber_scale_half_life.as_ref(),
                );
            }
            ModelKind::Quantile { solve_every, .. } => {
                put(&mut out, "solve_every", solve_every.as_ref());
            }
            ModelKind::Sgd {
                huber_scale_half_life,
                ..
            } => {
                put(
                    &mut out,
                    "huber_scale_half_life",
                    huber_scale_half_life.as_ref(),
                );
            }
            ModelKind::EwCov {
                pca_every,
                window_size: window,
                snapshot_every,
                ..
            } => {
                put(&mut out, "pca_every", pca_every.as_ref());
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "snapshot_every", snapshot_every.as_ref());
            }
            ModelKind::EwClass {
                window_size: window,
                snapshot_every,
                ..
            }
            | ModelKind::Marginal {
                window_size: window,
                snapshot_every,
                ..
            } => {
                put(&mut out, "window_size", window.as_ref());
                put(&mut out, "snapshot_every", snapshot_every.as_ref());
            }
            ModelKind::Holt {
                trend_half_life, ..
            } => {
                put(&mut out, "trend_half_life", trend_half_life.as_ref());
            }
            ModelKind::Micro { prune_every, .. } => {
                put(&mut out, "prune_every", prune_every.as_ref());
            }
            // A number of rows binds to no clock unit
            // ([`DURATION_OR_UNIT_FREE_FIELDS`]).
            ModelKind::Bocpd {
                hazard: Some(h), ..
            } if h.is_duration() => {
                put(&mut out, "hazard", Some(h));
            }
            _ => {}
        }
        out
    }

    /// The first rate per clock unit the spec sets to something a unit
    /// changes: `lam` other than 1, or a nonzero `q`.
    fn clock_rate(&self) -> Option<&'static str> {
        if self.lam.is_some_and(|l| l != 1.0) {
            return Some("lam");
        }
        match &self.model {
            ModelKind::Kalman { q: Some(q), .. } if q.iter().any(|v| v.0 != 0.0) => Some("q"),
            _ => None,
        }
    }

    /// Whether the spec's clock parameters are numbers, durations or neither,
    /// refusing a spec that mixes the two: a half-life of `"10m"` beside a
    /// `gap_cap` of `300` says nothing about what the `300` is in.
    pub fn clock_scale(&self) -> Result<ClockScale, String> {
        let spans = self.clock_spans();
        let duration = spans.iter().find(|(_, s)| s.is_duration()).map(|(f, _)| *f);
        // A rate first: it has no duration form, and its message names the
        // parameter to give instead. With a finite cap required beside it
        // (task 120), the cap would otherwise always be the number named.
        let number = self.clock_rate().or_else(|| {
            spans
                .iter()
                .find(|(_, s)| s.is_unit_bound_number())
                .map(|(f, _)| *f)
        });
        match (duration, number) {
            (Some("half_life"), Some("lam")) => Err(format!(
                "spec {:?}: half_life is a duration, but lam is set too, and lam is a decay over \
                 one clock unit, which a duration cannot give. Leave lam out.",
                self.name
            )),
            (Some(d), Some("lam")) if self.clock.is_none() => Err(format!(
                "spec {:?}: {d} is a duration, but the spec has no clock to measure it, and lam \
                 is a decay over one row. Set clock to a column of times (Datetime, Date or \
                 Duration) and give half_life as a duration in place of lam, or {}.",
                self.name,
                without_clock(d)
            )),
            (Some(d), Some("lam")) => Err(format!(
                "spec {:?}: {d} is a duration, but lam is a decay over one clock unit, which a \
                 duration cannot give. Give half_life as a duration in place of lam, with a \
                 Datetime, Date or Duration clock.",
                self.name
            )),
            (Some(d), Some("q")) => Err(format!(
                "spec {:?}: {d} is a duration, but q is the noise over one clock unit, which a \
                 duration cannot give. Leave q out and give coef_half_life as a duration, from \
                 which q is derived.",
                self.name
            )),
            (Some(d), Some(n)) if d == n => Err(format!(
                "spec {:?}: {d} mixes durations and plain numbers, but a plain number has no \
                 unit to read beside a duration. Give every value the same way: durations for \
                 a Datetime, Date or Duration clock, numbers for a numeric one.",
                self.name
            )),
            (Some(d), Some(n)) => Err(format!(
                "spec {:?}: {d} is a duration but {n} is a plain number, which has no unit to \
                 read beside it. Give every clock parameter the same way: durations for a \
                 Datetime, Date or Duration clock, numbers for a numeric one.",
                self.name
            )),
            // `gap_cap` caps a clock's step, and has no number-of-rows form:
            // without a clock it is refused whatever it is.
            (Some("gap_cap"), None) if self.clock.is_none() => Err(format!(
                "spec {:?}: gap_cap is a duration, but the spec has no clock to measure it. Set \
                 clock to a column of times (Datetime, Date or Duration), or leave gap_cap out.",
                self.name
            )),
            (Some(d), None) if self.clock.is_none() => Err(format!(
                "spec {:?}: {d} is a duration, but the spec has no clock to measure it, and \
                 without one each row is one clock unit. Set clock to a column of times \
                 (Datetime, Date or Duration), or {}.",
                self.name,
                without_clock(d)
            )),
            (Some(d), None) => Ok(ClockScale::Durations(d)),
            (None, Some(n)) => Ok(ClockScale::Numbers(n)),
            (None, None) => Ok(ClockScale::Free),
        }
    }
}

/// The way out without a clock for a clock parameter given as a duration:
/// a number of rows, but for the two that read a clock's step, which mean
/// nothing without one.
fn without_clock(d: &str) -> String {
    match d {
        "gap_cap" | "restart_after_step_back" => format!("leave {d} out"),
        _ => format!("give {d} as a number of rows"),
    }
}
