//! The configurations of the models a spec does not build field by field:
//! `audit`, `bocpd`, `corrchange`, `hmm`, `rcov` and `deco`, each read from
//! its spec, and `deco`'s block names (moved out of `stream.rs` under the
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
        emission,
        prior_mean,
        prior_kappa,
        prior_nu,
        prior_scale,
        robust_beta,
        prune_below,
        max_run,
        warm_rows,
    } = &spec.model
    else {
        return Err("not a bocpd spec".into());
    };
    Ok(BocpdCfg {
        n_features: spec.k(),
        // A number is the expected rows between changepoints; a duration,
        // which only a temporal clock reads (`Spec::clock_scale`), the
        // expected time between them, in the clock's seconds (task 179).
        hazard: hazard.as_ref().map_or(250.0, Span::value),
        hazard_from_row: hazard_col.is_some(),
        emission: match emission.as_deref() {
            None | Some("diag") => BocpdEmission::Diag,
            Some("gaussian") => BocpdEmission::Gaussian,
            Some("robust") => BocpdEmission::Robust,
            Some(other) => {
                return Err(format!(
                    "unknown bocpd emission {other:?}; expected \"gaussian\", \"diag\" or \
                     \"robust\""
                ));
            }
        },
        prior_mean: prior_mean.clone(),
        prior_kappa: prior_kappa.unwrap_or(1.0),
        prior_nu: *prior_nu,
        prior_scale: prior_scale.clone(),
        // Measured (`bocpd.rs`, `a_sustained_shift_survives_a_small_beta_
        // _and_not_a_large_one`): 0.1 ignores a 20-sigma row outright and
        // still finds a real shift within five rows; above ~0.2 nothing is
        // ever detected.
        robust_beta: robust_beta.unwrap_or(match emission.as_deref() {
            Some("robust") => 0.1,
            _ => 0.0,
        }),
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
        span_rows,
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
        reset_on_flag: reset,
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
            return Err(format!(
                "unknown corrchange kind {other:?}; expected \"monitor\", \"sequential\" or \
                 \"window\""
            ));
        }
    };
    let Some(span_rows) = *span_rows else {
        return Err(format!(
            "corrchange: kind = {name:?} needs `span_rows`, {}",
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
    // to the builders' default (`norm = "l1"`, `reset = false`) is taken as
    // unset, since the Python builder writes those whether or not asked.
    let refuse = |set: bool, param: &str, kinds: &str| -> Result<(), String> {
        if set {
            return Err(format!(
                "corrchange: {param} applies to kind = {kinds}, not {name:?}"
            ));
        }
        Ok(())
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
            *reset == Some(true),
            "reset_on_flag",
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
                return Err(format!(
                    "unknown corrchange norm {other:?}; expected \"l1\" or \"linf\""
                ));
            }
        },
        seed: seed.unwrap_or(0),
        reset: reset.unwrap_or(false),
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
        covariance,
        precision_prior,
        learn,
        transition_prior,
        transition,
        means,
        covs,
        warm_rows,
        seed_rule,
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
        ("seed_rule", seed_rule.is_some()),
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
        covariance: match covariance {
            Some(c) => Covariance::parse("hmm", c)?,
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
        seed_rule: match seed_rule.as_deref() {
            None | Some("lloyd") => SeedRule::Lloyd,
            Some("first") => SeedRule::First,
            Some("farthest") => SeedRule::Farthest,
            Some("kmeanspp") => SeedRule::Kmeanspp,
            Some(other) => {
                return Err(format!(
                    "unknown hmm seed_rule {other:?}; expected first, farthest, kmeanspp or lloyd"
                ));
            }
        },
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
            return Err(format!(
                "unknown rcov kind {other:?}; expected \"kernel\", \"preavg\" or \"plain\""
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
            return Err(format!(
                "unknown deco dynamics {other:?}; expected \"ew\" or \"linear\""
            ));
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
