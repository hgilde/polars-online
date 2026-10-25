//! The configurations of the models a spec does not build field by field:
//! `audit`, `bocpd`, `corrchange`, `hmm`, `rcov` and `deco`, each read from
//! its spec, and `deco`'s block names; and the models `huber`, `quantile`
//! and `sgd` build from their specs (moved out of `stream.rs` under the
//! 250 KB cap a source file is held to).

#[allow(unused_imports)]
use super::*;

/// An `audit` spec's [`online_core::AuditCfg`]: its columns are the spec's
/// features, and the clock's `gap_cap` is the stream's.
pub fn audit_cfg(spec: &Spec) -> Result<online_core::AuditCfg, String> {
    let ModelKind::Audit {
        pairs,
        distinct_cap,
    } = &spec.model
    else {
        return Err("not an audit spec".into());
    };
    let cfg = online_core::AuditCfg {
        n_columns: spec.features.len(),
        pairs: pairs.unwrap_or(false),
        distinct_cap: distinct_cap.unwrap_or(online_core::DISTINCT_CAP),
        gap_cap: spec.gap_cap.as_ref().map(Span::value),
        has_clock: spec.clock.is_some(),
    };
    cfg.validate()?;
    Ok(cfg)
}

/// A `bocpd` spec's [`BocpdCfg`]; every parameter check is the model's.
pub fn bocpd_cfg(spec: &Spec) -> Result<BocpdCfg, String> {
    let ModelKind::Bocpd {
        hazard,
        hazard_col,
        covariance_type,
        mean_prior,
        mean_precision_prior,
        degrees_of_freedom_prior,
        covariance_prior,
        robust_beta,
        prune_below,
        max_run,
        warm_rows,
    } = &spec.model
    else {
        return Err("not a bocpd spec".into());
    };
    let emission = match (covariance_type.as_deref(), robust_beta) {
        (None | Some("diag"), None) => BocpdEmission::Diag,
        (None | Some("diag"), Some(b)) => {
            // Measured (`bocpd.rs`, `a_sustained_shift_survives_a_small_beta_
            // _and_not_a_large_one`): 0.1 ignores a 20-sigma row outright and
            // still finds a real shift within five rows; above ~0.2 nothing
            // is ever detected. Left out, the predictive is not tempered.
            if !(b.is_finite() && *b > 0.0) {
                return Err(format!(
                    "bocpd: robust_beta must be finite and > 0 (0.1 ignores a 20-sigma row and \
                     still dates a four-sigma shift; leave it out for the untempered \
                     predictive), got {b}"
                ));
            }
            BocpdEmission::Robust
        }
        (Some("full"), None) => BocpdEmission::Gaussian,
        (Some("full"), Some(_)) => {
            return Err(
                "bocpd: robust_beta tempers the per-feature predictive, and covariance_type = \
                 \"full\" is a normal-inverse-Wishart over all the features: give \
                 covariance_type = \"diag\" with it, or leave robust_beta out"
                    .into(),
            );
        }
        (Some(other), _) => {
            return Err(crate::words_refusal(
                "covariance_type",
                &["full", "diag"],
                other,
            ));
        }
    };
    Ok(BocpdCfg {
        n_features: spec.k(),
        // A number is the expected rows between changepoints; a duration,
        // which only a temporal clock reads (`Spec::clock_scale`), the
        // expected time between them, in the clock's seconds (task 179).
        hazard: hazard.as_ref().map_or(250.0, Span::value),
        hazard_from_row: hazard_col.is_some(),
        // The covariance's words and the tempering, which `emission` once
        // said in one word: `"gaussian"` was `"full"`, `"robust"` the
        // per-feature pair tempered (task 244, P-2).
        emission,
        prior_mean: mean_prior.clone(),
        prior_kappa: mean_precision_prior.unwrap_or(1.0),
        prior_nu: *degrees_of_freedom_prior,
        prior_scale: covariance_prior.clone(),
        robust_beta: robust_beta.unwrap_or(0.0),
        prune_below: prune_below.unwrap_or(1e-6),
        max_run: max_run.unwrap_or(10_000),
        min_weight: spec.min_periods_or_default(),
        warm_rows: *warm_rows,
        hazard_on_clock: hazard.as_ref().is_some_and(Span::is_duration),
    })
}

/// A `corrchange` spec's [`CorrChangeCfg`]; every parameter check is the
/// model's.
pub fn corrchange_cfg(spec: &Spec) -> Result<CorrChangeCfg, String> {
    let ModelKind::CorrChange {
        kind,
        window_rows: span_rows,
        alpha,
        alpha_adjust,
        bandwidth,
        scalar,
        crit,
        n_perm,
        permute_every_rows,
        perm_block,
        norm,
        seed,
        change_action,
        monitor_rows,
        boundary_gamma,
    } = &spec.model
    else {
        return Err("not a corrchange spec".into());
    };
    let (kind, name) = match kind.as_deref() {
        None | Some("monitor") => (CorrChangeKind::Monitor, "monitor"),
        Some("window") => (CorrChangeKind::Window, "window"),
        Some("sequential") => (CorrChangeKind::Sequential, "sequential"),
        Some(other) => {
            return Err(crate::words_refusal(
                "kind",
                &["monitor", "sequential", "window"],
                other,
            ));
        }
    };
    let Some(span_rows) = *span_rows else {
        return Err(format!(
            "corrchange: kind = {name:?} needs `window_rows`, {}",
            match kind {
                CorrChangeKind::Sequential =>
                    "the rows of history each monitoring period is \
                                               tested against",
                _ => "the rows per comparison block",
            }
        ));
    };
    // A parameter that belongs to another kind is refused, not ignored: a
    // `crit` given to `"monitor"` changed nothing, in silence. A value equal
    // to the builders' default (`norm = "l1"`, `change_action = "flag"`) is
    // taken as unset, since the Python builder writes those whether or not
    // asked.
    let refuse = |set: bool, param: &str, kinds: &str| -> Result<(), String> {
        if set {
            return Err(format!(
                "corrchange: {param} applies to kind = {kinds}, not {name:?}"
            ));
        }
        Ok(())
    };
    let reset = match change_action.as_deref() {
        None | Some("flag") => false,
        Some("reset") => true,
        Some(other) => {
            return Err(crate::words_refusal(
                "change_action",
                &["flag", "reset"],
                other,
            ));
        }
    };
    let (monitor, window, sequential) = (
        kind == CorrChangeKind::Monitor,
        kind == CorrChangeKind::Window,
        kind == CorrChangeKind::Sequential,
    );
    if !window {
        refuse(n_perm.is_some(), "n_perm", "\"window\"")?;
        refuse(
            permute_every_rows.is_some(),
            "permute_every_rows",
            "\"window\"",
        )?;
        refuse(perm_block.is_some(), "perm_block", "\"window\"")?;
        refuse(seed.is_some(), "seed", "\"window\"")?;
        refuse(
            norm.as_deref().is_some_and(|n| n != "l1"),
            "norm",
            "\"window\"",
        )?;
        refuse(
            reset,
            "change_action",
            "\"window\" (a \"monitor\" span and a \"sequential\" cycle end at their flag \
             already)",
        )?;
    }
    refuse(
        monitor && crit.is_some(),
        "crit",
        "\"window\" or \"sequential\" (\"monitor\"'s is the Kolmogorov quantile)",
    )?;
    refuse(
        window && bandwidth.is_some(),
        "bandwidth",
        "\"monitor\" or \"sequential\"",
    )?;
    // `"window"` takes its permutation quantile at `alpha` itself, so a
    // spread over the pairs changed nothing there (task 160, CD4). The
    // builders' default, "bonferroni", is taken as unset.
    refuse(
        window && alpha_adjust.as_deref().is_some_and(|a| a != "bonferroni"),
        "alpha_adjust",
        "\"monitor\" or \"sequential\" (\"window\" takes its permutation quantile at alpha)",
    )?;
    if !sequential {
        refuse(monitor_rows.is_some(), "monitor_rows", "\"sequential\"")?;
        refuse(boundary_gamma.is_some(), "boundary_gamma", "\"sequential\"")?;
    }
    Ok(CorrChangeCfg {
        n_features: spec.k(),
        kind,
        span_rows,
        alpha: alpha.unwrap_or(0.05),
        alpha_adjust: alpha_adjust.clone().unwrap_or_else(|| "bonferroni".into()),
        bandwidth: *bandwidth,
        scalar: scalar.unwrap_or(false),
        decay: online_core::Decay::Lam(1.0),
        crit: *crit,
        n_perm: n_perm.unwrap_or(200),
        permute_every_rows: permute_every_rows.unwrap_or(50),
        perm_block: perm_block.unwrap_or(1),
        norm: match norm.as_deref() {
            None | Some("l1") => ChangeNorm::L1,
            Some("linf") => ChangeNorm::LInf,
            Some(other) => {
                return Err(crate::words_refusal("norm", &["l1", "linf"], other));
            }
        },
        seed: seed.unwrap_or(0),
        reset,
        // W&G's `T = 1`: as many rows monitored as the history has.
        monitor_rows: if sequential {
            monitor_rows.unwrap_or(span_rows)
        } else {
            0
        },
        boundary_gamma: boundary_gamma.unwrap_or(0.0),
    })
}

/// An `hmm` spec's [`HmmCfg`]. The decay is the caller's; every other check
/// is the model's, so `Spec::validate` gets the same messages.
pub fn hmm_cfg(spec: &Spec) -> Result<HmmCfg, String> {
    let ModelKind::Hmm {
        k,
        covariance_type,
        precision_prior,
        learn,
        transition_prior,
        transition,
        means,
        covs,
        warm_rows,
        init,
        seed,
        exog_tvtp,
        tvtp_coef,
    } = &spec.model
    else {
        return Err("not an hmm spec".into());
    };
    let tvtp = match (exog_tvtp, tvtp_coef) {
        (Some(_), Some(ab)) if ab.len() == 2 => Some((ab[0].clone(), ab[1].clone())),
        (Some(_), _) => {
            return Err(
                "hmm exog_tvtp needs tvtp_coef = [A, B], each a K x K matrix flattened row-major"
                    .into(),
            );
        }
        (None, Some(_)) => {
            return Err("hmm tvtp_coef needs exog_tvtp (the column it reads)".into());
        }
        (None, None) => None,
    };
    // A parameter its mode does not read is refused, not ignored (review
    // 2026-10-06, CE4): under `tvtp_coef` no transition count is learned,
    // and with the states given nothing is seeded. A value given is told
    // from the default here, where both are `None` until filled.
    for (param, given) in [
        ("transition", transition.is_some()),
        ("transition_prior", transition_prior.is_some()),
    ] {
        if given && tvtp.is_some() {
            return Err(format!(
                "hmm: {param} does not apply with tvtp_coef: the matrix is softmax(A + B·z) and \
                 no transition count is learned, so there is no prior to spread"
            ));
        }
    }
    for (param, given) in [
        ("warm_rows", warm_rows.is_some()),
        ("init", init.is_some()),
        ("seed", seed.is_some()),
    ] {
        if given && means.is_some() && covs.is_some() {
            return Err(format!(
                "hmm: {param} does not apply with means and covs given: the states are not \
                 seeded from the rows, so there is no warm-up"
            ));
        }
    }
    Ok(HmmCfg {
        n_features: spec.k(),
        k: *k,
        decay: online_core::Decay::Lam(1.0),
        covariance: match covariance_type {
            Some(c) => Covariance::parse(c)?,
            None => Covariance::Full,
        },
        precision_prior: *precision_prior,
        min_weight: spec.min_periods_or_default(),
        learn: learn.unwrap_or(true),
        transition_prior: transition_prior.unwrap_or(1.0),
        transition: transition.clone(),
        means: means.clone(),
        covs: covs.clone(),
        warm_rows: warm_rows.unwrap_or(50),
        seed_rule: crate::stream::init_rule(init.as_deref())?,
        seed: seed.unwrap_or(0),
        tvtp,
    })
}

/// An `rcov` spec's [`RcovCfg`]. Every parameter check is the model's, so
/// `Spec::validate` and the CLI get one set of messages.
pub fn rcov_cfg(spec: &Spec) -> Result<RcovCfg, String> {
    let ModelKind::Rcov {
        kind,
        kernel,
        bandwidth,
        jitter,
        theta,
        psd,
        block_rows,
        max_bandwidth,
        preavg_rows,
        noise_stride,
        iv_stride,
    } = &spec.model
    else {
        return Err("not an rcov spec".into());
    };
    let kind = match kind.as_deref() {
        None | Some("kernel") => RcovKind::Kernel,
        Some("preavg") => RcovKind::Preavg,
        Some("plain") => RcovKind::Plain,
        Some(other) => {
            return Err(crate::words_refusal(
                "kind",
                &["kernel", "preavg", "plain"],
                other,
            ));
        }
    };
    // A parameter its kind does not read is refused, not ignored, as
    // `bandwidth` and `preavg_rows` are (review 2026-10-06, CE4): the jitter
    // and the ring are the kernel's, `theta` sets the pre-averaging window.
    // The builders leave both `None` unless given.
    for (param, given, owner) in [
        ("jitter", jitter.is_some(), RcovKind::Kernel),
        ("max_bandwidth", max_bandwidth.is_some(), RcovKind::Kernel),
        ("theta", theta.is_some(), RcovKind::Preavg),
    ] {
        if given && kind != owner {
            return Err(format!(
                "rcov: {param} applies to kind = {:?}, not {:?}",
                owner.as_str(),
                kind.as_str()
            ));
        }
    }
    Ok(RcovCfg {
        n_features: spec.k(),
        kind,
        kernel: kernel.clone().unwrap_or_else(|| "parzen".into()),
        bandwidth: *bandwidth,
        jitter: jitter.unwrap_or(2),
        theta: theta.unwrap_or(1.0),
        psd: psd.unwrap_or(true),
        block_rows: *block_rows,
        max_bandwidth: *max_bandwidth,
        preavg_rows: *preavg_rows,
        noise_stride: noise_stride.unwrap_or(1),
        iv_stride: iv_stride.unwrap_or(20),
    })
}

/// The named blocks of a `deco` spec, if it has any.
fn blocks_named(model: &ModelKind) -> Option<&Vec<(String, Vec<String>)>> {
    match model {
        ModelKind::Deco { blocks, .. } => blocks.as_ref(),
        _ => None,
    }
}

/// A `deco` spec's [`DecoCfg`], with the block *names* resolved to feature
/// positions. The decay is the caller's (one instance per half-life); every
/// other check is `DecoCfg::validate`'s, so `Spec::validate` gets the same
/// messages the model would give.
pub fn deco_cfg(spec: &Spec) -> Result<DecoCfg, String> {
    let ModelKind::Deco {
        dynamics,
        alpha,
        beta,
        blocks,
    } = &spec.model
    else {
        return Err("not a deco spec".into());
    };
    let dynamics = match dynamics.as_deref() {
        None | Some("ew") => DecoDynamics::Ew,
        Some("linear") => DecoDynamics::Linear,
        Some(other) => {
            return Err(crate::words_refusal("dynamics", &["ew", "linear"], other));
        }
    };
    let blocks = match blocks {
        None => Vec::new(),
        Some(named) if named.is_empty() => {
            // An empty list is not "one block of everything" and not the
            // unblocked form either; whichever was meant, say which
            // (docs/REVIEW-E54-E64.md D1).
            // One sentence: continued across the line without a `\`, the
            // literal carried a run of 18 spaces (task 160, PB8).
            return Err("deco blocks is empty; leave it out for the unblocked \
                        equicorrelation, or name at least one block"
                .into());
        }
        Some(named) => named
            .iter()
            .map(|(name, cols)| {
                cols.iter()
                    .map(|c| {
                        spec.features.iter().position(|f| f == c).ok_or_else(|| {
                            format!("deco block {name:?} names {c:?}, which is not a feature")
                        })
                    })
                    .collect::<Result<Vec<usize>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    // A JSON or TOML spec writes `blocks` as an array of pairs, so two of
    // them can carry the same name; the collision then surfaces as two
    // output fields called `u_u` rather than as the block list's problem.
    if let Some(named) = blocks_named(&spec.model) {
        let mut seen = std::collections::HashSet::new();
        if let Some((dup, _)) = named.iter().find(|(n, _)| !seen.insert(n.as_str())) {
            return Err(format!(
                "deco block {dup:?} is named twice; block names are the output labels, so \
                 they have to be distinct"
            ));
        }
    }
    Ok(DecoCfg {
        n_features: spec.k(),
        decay: online_core::Decay::Lam(1.0),
        dynamics,
        alpha: *alpha,
        beta: *beta,
        blocks,
        min_weight: spec.min_periods_or_default(),
    })
}

/// The block names of a `deco` spec, in emission order; empty when it has
/// none, which is the unblocked model.
pub fn deco_block_names(spec: &Spec) -> Vec<String> {
    match &spec.model {
        ModelKind::Deco {
            blocks: Some(named),
            ..
        } => named.iter().map(|(n, _)| n.clone()).collect(),
        _ => Vec::new(),
    }
}

/// A `huber` spec's model: `ewridge`'s sums per target, each row at its Huber weight.
pub fn huber_model(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let ModelKind::Huber {
        huber_delta,
        huber_scale,
        huber_scale_half_life,
        ridge,
        standardize,
        solve_every,
        max_rows_between_solves,
        feature_sets: _,
        ridge_scale: _,
        emit_huber_scale,
        huber_refit_rows,
    } = &spec.model
    else {
        return Err("not a huber spec".into());
    };
    let cfg = RobustCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay,
        // The 95%-efficiency constant (Huber 1981; statsmodels'
        // `HuberT`, scikit-learn's 1.35): task 195, U3.
        loss: RobustLoss::Huber {
            delta: huber_delta.map_or(1.345, |n| n.0),
        },
        ridge: ridge.unwrap_or(1e-6),
        standardize: *standardize,
        min_weight: spec.min_periods_or_default(),
        solve_every: solve_every
            .as_ref()
            .map_or_else(|| spec.solve_every_default(decay), Span::value),
        max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
        solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
        quantile_eps: 1e-3,
        huber_scale: huber_scale_of(huber_scale.as_deref()),
        huber_scale_decay: huber_scale_half_life
            .as_ref()
            .map(|h| Decay::Halflife(h.value())),
        feature_sets: huber_feature_sets(spec),
        ridge_scale: huber_ridge_scale_sum(spec),
        emit_scale: *emit_huber_scale,
        // On by default, but under the sum-scale prior, which the refit
        // does not hold, and an infinite cut, least squares, with no
        // outliers to refit for (`RobustCfg::refit_rows`): there it is off
        // unless asked for, and asked for it is refused.
        refit_rows: huber_refit_rows.unwrap_or(
            if huber_ridge_scale_sum(spec) || huber_delta.is_some_and(|d| d.0.is_infinite()) {
                0
            } else {
                online_core::DEFAULT_REFIT_ROWS
            },
        ),
    };
    let mut m = Robust::new(cfg)?;
    // Each target's own threshold, for its own first solve (S9b).
    m.set_target_min_weight(spec.min_periods_per_target())?;
    Ok(AnyModel::Robust(Box::new(m)))
}

/// A `huber` spec's feature sets as the model takes them: each set's
/// columns as positions in the spec's features, which `Spec::validate` has
/// held to name them; none without sets.
fn huber_feature_sets(spec: &Spec) -> Vec<(String, Vec<usize>)> {
    let ModelKind::Huber {
        feature_sets: Some(sets),
        ..
    } = &spec.model
    else {
        return Vec::new();
    };
    sets.iter()
        .map(|(name, cols)| {
            let idx = cols
                .iter()
                .filter_map(|c| spec.features.iter().position(|f| f == c))
                .collect();
            (name.clone(), idx)
        })
        .collect()
}

/// Whether a `huber` spec asks for RLS's sum-scale prior.
fn huber_ridge_scale_sum(spec: &Spec) -> bool {
    matches!(
        spec.model,
        ModelKind::Huber {
            ridge_scale: crate::spec::RidgeScale::Sum,
            ..
        }
    )
}

/// A `quantile` spec's model.
pub fn quantile_model(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let ModelKind::Quantile {
        quantile,
        ridge,
        standardize,
        solve_every,
        max_rows_between_solves,
        quantile_eps,
    } = &spec.model
    else {
        return Err("not a quantile spec".into());
    };
    let cfg = RobustCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay,
        loss: RobustLoss::Quantile { tau: *quantile },
        ridge: ridge.unwrap_or(1e-6),
        standardize: *standardize,
        min_weight: spec.min_periods_or_default(),
        solve_every: solve_every
            .as_ref()
            .map_or_else(|| spec.solve_every_default(decay), Span::value),
        max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
        solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
        quantile_eps: quantile_eps.unwrap_or(0.2),
        // The band is drawn in the EW root mean square, on the
        // model's clock (task 271 leaves the quantile fit as it was).
        huber_scale: HuberScale::Rms,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
    };
    let mut m = Robust::new(cfg)?;
    // Each target's own threshold, for its own first solve (S9b).
    m.set_target_min_weight(spec.min_periods_per_target())?;
    Ok(AnyModel::Robust(Box::new(m)))
}

/// An `sgd` spec's model, its loss and schedule read from their words.
pub fn sgd_model(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let ModelKind::Sgd {
        loss,
        huber_delta,
        huber_scale,
        huber_scale_half_life,
        huber_refit_rows,
        quantile,
        eps,
        learning_rate,
        schedule,
        power,
        l2,
        clip_gradient,
        standardize,
        coef_min,
        coef_max,
        strict_binary,
        coef_sum,
    } = &spec.model
    else {
        return Err("not a sgd spec".into());
    };
    let loss = match loss.as_deref().unwrap_or("squared") {
        "squared" => SgdLoss::Squared,
        // In units of the target's residual std, `huber`'s constant
        // under `huber`'s name (task 195, U1 and U3).
        "huber" => SgdLoss::Huber {
            delta: huber_delta.map_or(1.345, |n| n.0),
        },
        "quantile" => SgdLoss::Quantile {
            tau: quantile.ok_or("sgd: loss \"quantile\" needs a `quantile` level")?,
        },
        // 1% of the target's own spread (task 203), kept by task
        // 207's sweep, which measured it against the residual's
        // spread capped by the target's.
        "epsilon_insensitive" => SgdLoss::EpsilonInsensitive {
            eps: eps.unwrap_or(0.01),
        },
        "poisson" => SgdLoss::Poisson,
        "logistic" => SgdLoss::Logistic,
        other => return Err(format!("unknown sgd loss {other:?}")),
    };
    let sched = match schedule.as_deref().unwrap_or("constant") {
        "constant" => LearningRate::Constant,
        "inv_scaling" => LearningRate::InvScaling {
            power: power.unwrap_or(0.5),
        },
        "adagrad" => LearningRate::AdaGrad,
        other => return Err(format!("unknown sgd schedule {other:?}")),
    };
    let cfg = SgdCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        decay,
        loss,
        learning_rate: learning_rate.unwrap_or(0.01),
        schedule: sched,
        l2: l2.unwrap_or(0.0),
        min_weight: spec.min_periods_or_default(),
        // Finite by default: see SgdCfg::clip_gradient.
        clip_gradient: clip_gradient.map_or(1e3, |n| n.0),
        standardize: *standardize,
        constraint: constraint(spec.k(), coef_min, coef_max, *coef_sum),
        strict_binary: *strict_binary,
        // `huber`'s scale under the Huber loss; every other loss
        // reads none, and the spec refuses one beside it.
        huber_scale: if matches!(loss, SgdLoss::Huber { .. }) {
            huber_scale_of(huber_scale.as_deref())
        } else {
            HuberScale::Rms
        },
        huber_scale_decay: huber_scale_half_life
            .as_ref()
            .filter(|_| matches!(loss, SgdLoss::Huber { .. }))
            .map(|h| Decay::Halflife(h.value())),
        // `huber`'s refit under the Huber loss, on by default; every other
        // loss has none, and the spec refuses one beside it.
        // Off under an infinite cut, the squared loss, with no outliers to
        // refit for.
        refit_rows: match loss {
            SgdLoss::Huber { delta } if delta.is_finite() => {
                huber_refit_rows.unwrap_or(online_core::DEFAULT_REFIT_ROWS)
            }
            SgdLoss::Huber { .. } => huber_refit_rows.unwrap_or(0),
            _ => 0,
        },
    };
    Ok(AnyModel::Sgd(Box::new(Sgd::new(cfg)?)))
}

#[cfg(test)]
mod refit_tests {
    //! The refit's default (docs/PLAN.md task 277): on at
    //! `DEFAULT_REFIT_ROWS` for `huber` and `sgd(loss = "huber")`, off
    //! where the refit does not apply -- the sum-scale prior, an infinite
    //! cut, another loss -- and as given where given, refused there.
    use super::*;

    fn spec(model: &str) -> Spec {
        let text = format!(
            "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x\"]\nhalf_life = 10.0\n[model]\n{model}\n"
        );
        toml::from_str(&text).unwrap_or_else(|e| panic!("did not parse: {e}\n{text}"))
    }

    fn huber(model: &str) -> Result<u32, String> {
        match huber_model(
            &spec(&format!("type = \"huber\"\n{model}")),
            Decay::Halflife(10.0),
        )? {
            AnyModel::Robust(m) => Ok(m.cfg().refit_rows),
            _ => unreachable!(),
        }
    }

    fn sgd(model: &str) -> Result<u32, String> {
        match sgd_model(
            &spec(&format!("type = \"sgd\"\n{model}")),
            Decay::Halflife(10.0),
        )? {
            AnyModel::Sgd(m) => Ok(m.cfg().refit_rows),
            _ => unreachable!(),
        }
    }

    #[test]
    fn the_refit_is_on_by_default_where_it_applies() {
        let on = online_core::DEFAULT_REFIT_ROWS;
        assert_eq!(huber(""), Ok(on));
        assert_eq!(huber("huber_refit_rows = 7"), Ok(7));
        assert_eq!(huber("huber_refit_rows = 0"), Ok(0));
        assert_eq!(huber("ridge_scale = \"sum\""), Ok(0));
        assert_eq!(huber("huber_delta = inf"), Ok(0));
        assert!(huber("ridge_scale = \"sum\"\nhuber_refit_rows = 7").is_err());
        assert!(huber("huber_delta = inf\nhuber_refit_rows = 7").is_err());
        assert_eq!(sgd("loss = \"huber\""), Ok(on));
        assert_eq!(sgd("loss = \"huber\"\nhuber_refit_rows = 7"), Ok(7));
        assert_eq!(sgd("loss = \"huber\"\nhuber_delta = inf"), Ok(0));
        assert!(sgd("loss = \"huber\"\nhuber_delta = inf\nhuber_refit_rows = 7").is_err());
        assert_eq!(sgd(""), Ok(0));
        assert_eq!(sgd("loss = \"quantile\"\nquantile = 0.5"), Ok(0));
    }
}
