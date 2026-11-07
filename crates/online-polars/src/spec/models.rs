//! Each model's own settings, checked by the spec: the arm of
//! `Spec::validate_each` that reads `self.model`, kept apart for its size.

use super::*;

impl Spec {
    /// Every problem in the model's own settings, added to `no`.
    pub(super) fn validate_model(&self, no: &mut online_core::Refusals) {
        // The decay parameter a spec set, named in the refusals of the
        // models that take none.
        let decay_key = if self.half_life.is_some() {
            "half_life"
        } else {
            "lam"
        };
        // A `coef_prior` of the wrong shape, `rls`'s and `ewridge`'s: what
        // it must hold, and what it holds (`2 lists of 3`, or `2 lists, of 3
        // and 2 coefficients`).
        let coef_prior_refusal = |c: &[Vec<f64>], k_total: usize| {
            let mut lens: Vec<usize> = c.iter().map(Vec::len).collect();
            lens.dedup();
            let s = if c.len() == 1 { "" } else { "s" };
            let holds = match lens.as_slice() {
                [] => "none".to_string(),
                [one] => format!("{} list{s} of {one}", c.len()),
                _ => format!(
                    "{} list{s}, of {} coefficients",
                    c.len(),
                    c.iter()
                        .map(|v| v.len().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            };
            let each = if self.m() == 1 {
                "the target".to_string()
            } else {
                format!("each of the {} targets", self.m())
            };
            let order = if self.fit_intercept {
                "the intercept first, then one for each feature"
            } else {
                "one for each feature"
            };
            format!(
                "spec {:?}: coef_prior must hold one list of {k_total} coefficients for {each}, \
             but it holds {holds}. Give one list for each target, {order} in the order of \
             features.",
                self.name
            )
        };
        match &self.model {
            ModelKind::Holt {
                trend_half_life,
                trend,
            } => {
                if *trend == Some(false) && trend_half_life.is_some() {
                    no.refuse(format!(
                        "spec {:?}: trend_half_life sets how fast the trend forgets, but \
                     trend=False holds the trend at 0. Leave trend_half_life out, or set \
                     trend=True.",
                        self.name
                    ));
                }
                if let Some(h) = trend_half_life
                    .as_ref()
                    .filter(|h| h.value() <= 0.0 || h.value().is_nan())
                {
                    no.refuse(format!(
                        "spec {:?}: trend_half_life must be above 0, but it is {h}. Set \
                     trend_half_life to a positive number of clock units, or to inf to fit \
                     the trend over the whole history.",
                        self.name
                    ));
                }
            }
            ModelKind::Pa { mode, c, eps, .. } => {
                const MODE: [&str; 3] = ["pa", "pa1", "pa2"];
                if let Some(md) = mode
                    && !MODE.contains(&md.as_str())
                {
                    no.refuse(spec_words_refusal(&self.name, "mode", &MODE, md));
                }
                if let Some(v) = c.filter(|v| v.0 <= 0.0 || v.0.is_nan()) {
                    no.refuse(format!(
                        "spec {:?}: c must be above 0, but it is {}. Set c to a positive value: \
                     the larger it is, the further one row can move the coefficients, and \
                     inf caps nothing.",
                        self.name,
                        num_label(v.0)
                    ));
                } else if c.is_some() && mode.as_deref() == Some("pa") {
                    // Refused where nothing reads it, as `sgd`'s `power` is.
                    no.refuse(format!(
                        "spec {:?}: c caps the step under mode=\"pa1\" and damps it under \
                     mode=\"pa2\", but mode is \"pa\", which reads neither. Leave c out, or \
                     set mode=\"pa1\" or \"pa2\".",
                        self.name
                    ));
                }
                // A tube of `inf` holds every row: a model that never learns
                // (review 2026-09-12, S27).
                if let Some(v) = eps.filter(|v| *v < 0.0 || !v.is_finite()) {
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "eps",
                        "",
                        &Got::One(num_label(v)),
                        Some("to move the fit on every row with a residual other than 0"),
                        None,
                    ));
                }
            }
            ModelKind::Sgd {
                loss,
                huber_delta,
                huber_scale,
                huber_scale_half_life,
                huber_refit_rows,
                refit_budget,
                quantile,
                eps,
                learning_rate,
                schedule,
                power,
                strict_binary,
                ..
            } => {
                no.take(check_huber_scale(
                    &self.name,
                    huber_scale.as_deref(),
                    huber_scale_half_life.as_ref(),
                ));
                no.take(check_refit_budget(&self.name, *refit_budget));
                const OK: [&str; 6] = [
                    "squared",
                    "huber",
                    "quantile",
                    "epsilon_insensitive",
                    "poisson",
                    "logistic",
                ];
                // A loss outside the words is said once, and the checks of
                // what each loss uses below read none.
                let loss_known = loss.as_ref().is_none_or(|l| OK.contains(&l.as_str()));
                if let Some(l) = loss {
                    if !loss_known {
                        no.refuse(spec_words_refusal(&self.name, "loss", &OK, l));
                    }
                    if l == "quantile" && quantile.is_none() {
                        no.refuse(format!(
                            "spec {:?}: loss=\"quantile\" needs a quantile level, but quantile \
                         is not set. Set quantile to the level to fit, such as 0.5 for the \
                         median.",
                            self.name
                        ));
                    }
                }
                const SCHEDULE: [&str; 3] = ["constant", "inv_scaling", "adagrad"];
                let schedule_known = schedule
                    .as_ref()
                    .is_none_or(|sc| SCHEDULE.contains(&sc.as_str()));
                if let Some(sc) = schedule
                    && !schedule_known
                {
                    no.refuse(spec_words_refusal(&self.name, "schedule", &SCHEDULE, sc));
                }
                if let Some(v) = learning_rate.filter(|v| *v <= 0.0 || !v.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: learning_rate must be finite and above 0, but it is {}. Set \
                     learning_rate to a finite positive step size, such as 0.01.",
                        self.name,
                        num_label(v)
                    ));
                }
                // Unchecked, a NaN reached the core's `f64::clamp`, which
                // panics on a NaN bound.
                if let Some(d) = huber_delta.filter(|d| !positive(d.0)) {
                    no.refuse(format!(
                        "spec {:?}: huber_delta must be above 0, but it is {}. Set huber_delta \
                     to a positive cut in units of the residuals' scale, or to inf for the \
                     squared loss.",
                        self.name,
                        num_label(d.0)
                    ));
                }
                // A parameter of a loss or a schedule the spec does not use
                // is refused, as a switch that is off is: each was taken and
                // ignored. The builder has refused them since task 160 (YA8);
                // a dict or a TOML file is refused here, in its words (YA8b).
                // A null is the default, not a value given.
                let loss = loss.as_deref().unwrap_or("squared");
                let schedule = schedule.as_deref().unwrap_or("constant");
                for (key, given, what, owner, chosen) in [
                    ("huber_delta", huber_delta.is_some(), "loss", "huber", loss),
                    ("huber_scale", huber_scale.is_some(), "loss", "huber", loss),
                    (
                        "huber_scale_half_life",
                        huber_scale_half_life.is_some(),
                        "loss",
                        "huber",
                        loss,
                    ),
                    (
                        "huber_refit_rows",
                        huber_refit_rows.is_some(),
                        "loss",
                        "huber",
                        loss,
                    ),
                    (
                        "refit_budget",
                        refit_budget.is_some(),
                        "loss",
                        "huber",
                        loss,
                    ),
                    ("quantile", quantile.is_some(), "loss", "quantile", loss),
                    ("eps", eps.is_some(), "loss", "epsilon_insensitive", loss),
                    ("strict_binary", *strict_binary, "loss", "logistic", loss),
                    (
                        "power",
                        power.is_some(),
                        "schedule",
                        "inv_scaling",
                        schedule,
                    ),
                ] {
                    let known = if what == "loss" {
                        loss_known
                    } else {
                        schedule_known
                    };
                    if given && known && chosen != owner {
                        no.refuse(format!(
                            "spec {:?}: {key} applies only under {what}={owner:?}, but {what} is \
                         {chosen:?}, which does not use it. Leave {key} out, or set \
                         {what}={owner:?}.",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::EwCov {
                stats,
                // Either value is a choice.
                bias: _,
                precision_prior,
                mahal_quantiles,
                pca,
                pca_every,
                max_rows_between_pcas,
                lags,
                // Checked with `ew_ridge`'s (`gram_threads_given`).
                gram_threads: _,
                window_size: window,
                // Checked with every windowed kind's (`window_cadence`).
                snapshot_every: _,
                max_rows_between_snapshots: _,
                window_budget: _,
                closed: _,
            } => {
                if let Some(w) = window
                    && (!w.value().is_finite() || w.value() <= 0.0)
                {
                    no.refuse(format!(
                        "spec {:?}: window_size must be finite and above 0, but it is {w}. Set \
                     window_size to the age, in clock units, past which a row counts for \
                     nothing, or leave it out for no hard window.",
                        self.name
                    ));
                }
                const OK: [&str; 8] = [
                    "mean",
                    "var",
                    "std",
                    "cov",
                    "corr",
                    "partial_corr",
                    "mahal",
                    "lag_corr",
                ];
                if let Some(stats) = stats {
                    // `stats = []` stays legal: it is the accumulate-only use
                    // (docs/ENHANCEMENTS.md E43), which writes `weight_sum`
                    // alone and keeps the Gram, not a spelling of absent.
                    let unknown: Vec<&String> = stats
                        .iter()
                        .filter(|st| !OK.contains(&st.as_str()))
                        .collect();
                    for st in &unknown {
                        let renamed = RENAMED_VALUES.iter().find(|(old, _)| *old == st.as_str());
                        no.refuse(match renamed {
                            Some((old, new)) => format!(
                                "spec {:?}: stats lists {old:?}, which was renamed {new:?}. \
                             Write {new:?} in stats.",
                                self.name
                            ),
                            None => spec_words_refusal(&self.name, "stats", &OK, st),
                        });
                    }
                    let pairwise: Vec<&str> = stats
                        .iter()
                        .map(String::as_str)
                        .filter(|st| ["cov", "corr", "partial_corr"].contains(st))
                        .collect();
                    if self.k() < 2 && !pairwise.is_empty() {
                        let listed = checks::listed(
                            &pairwise
                                .iter()
                                .map(|s| format!("{s:?}"))
                                .collect::<Vec<_>>(),
                        );
                        no.refuse(format!(
                            "spec {:?}: stats lists {listed}, which {} two features, but \
                         features names {}. Name at least two columns in features, or \
                         leave {listed} out of stats.",
                            self.name,
                            if pairwise.len() == 1 {
                                "compares"
                            } else {
                                "compare"
                            },
                            columns_counted(self.k())
                        ));
                    }
                    if stats.iter().any(|st| st == "partial_corr") && precision_prior.is_none() {
                        no.refuse(format!(
                            "spec {:?}: \"partial_corr\" in stats needs precision_prior, but it \
                         is not set: the partial correlations are read from the precision \
                         matrix it regularizes. Set precision_prior, such as 1e-6, or leave \
                         \"partial_corr\" out of stats.",
                            self.name
                        ));
                    }
                    if stats.iter().any(|st| st == "mahal") && precision_prior.is_none() {
                        no.refuse(format!(
                            "spec {:?}: \"mahal\" in stats needs precision_prior, but it is not \
                         set: the covariances cannot be inverted until more rows than \
                         features have been learned. Set precision_prior, such as 1e-6, or \
                         leave \"mahal\" out of stats.",
                            self.name
                        ));
                    }
                }
                if let Some(p) = precision_prior.filter(|p| *p <= 0.0 || !p.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: precision_prior must be finite and above 0, but it is {}. \
                     Set precision_prior to a small positive value added to each variance \
                     before the covariances are inverted, such as 1e-6.",
                        self.name,
                        num_label(p)
                    ));
                }
                let has_lag_corr = stats
                    .as_ref()
                    .is_some_and(|st| st.iter().any(|s| s == "lag_corr"));
                // An empty list is refused by the lags' own check below.
                if has_lag_corr && lags.is_none() {
                    no.refuse(format!(
                        "spec {:?}: \"lag_corr\" in stats needs lags, but lags is not set. Set \
                     lags to the lags to report, such as [1, 2, 5], or leave \"lag_corr\" \
                     out of stats.",
                        self.name
                    ));
                }
                if let Some(lags) = lags {
                    // The list's own rules are the running sums', so the CLI
                    // and the bank get one message, each problem of it named
                    // by the spec.
                    no.take(
                        online_core::EwLagCov::new(self.k().max(1), lags.clone())
                            .map(|_| ())
                            .map_err(|e| {
                                e.lines()
                                    .map(|l| format!("spec {:?}: {l}", self.name))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            }),
                    );
                }
                if let Some(levels) = mahal_quantiles {
                    // An empty list asked for nothing and was taken without a
                    // word, where `resid_quantiles = []` is refused (review
                    // 2026-10-06, YA5).
                    if levels.is_empty() {
                        no.refuse(format!(
                            "spec {:?}: mahal_quantiles must list at least one level, but it is \
                         empty. Give one or more levels, such as [0.99], or leave \
                         mahal_quantiles out.",
                            self.name
                        ));
                    } else {
                        let has_mahal = stats
                            .as_ref()
                            .is_some_and(|st| st.iter().any(|s| s == "mahal"));
                        if !has_mahal {
                            no.refuse(format!(
                                "spec {:?}: mahal_quantiles needs \"mahal\" in stats, but stats \
                             does not list it. Add \"mahal\" to stats, or leave out \
                             mahal_quantiles.",
                                self.name
                            ));
                        }
                        let bad: Vec<String> = levels
                            .iter()
                            .filter(|q| !(**q > 0.0 && **q < 1.0))
                            .map(|q| num_label(*q))
                            .collect();
                        if !bad.is_empty() {
                            no.refuse(format!(
                                "spec {:?}: mahal_quantiles must each be above 0 and below 1, \
                             but {} {}. Set each to a level above 0 and below 1, such as \
                             0.99.",
                                self.name,
                                if bad.len() == 1 {
                                    "one of them is"
                                } else {
                                    "these are not:"
                                },
                                bad.join(", ")
                            ));
                        }
                    }
                }
                // `0` was a second spelling of absent (review 2026-10-06,
                // PC9; docs/PLAN.md task 196, U7).
                if *pca == Some(0) {
                    no.refuse(format!(
                        "spec {:?}: pca must be at least 1, but it is 0, which asks for no \
                     component. Set pca to the number of components, or leave it out for \
                     none.",
                        self.name
                    ));
                }
                if let Some(r) = pca
                    && *r > self.k()
                {
                    no.refuse(format!(
                        "spec {:?}: pca must be at most the number of features ({}), but it is \
                     {r}. Set pca to {} or fewer components.",
                        self.name,
                        self.k(),
                        self.k()
                    ));
                }
                if let Some(e) = pca_every
                    .as_ref()
                    .filter(|e| !non_negative(e.value()) || !e.value().is_finite())
                {
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "pca_every",
                        " clock units",
                        &Got::One(e.to_string()),
                        Some("to refresh the principal components on every row"),
                        None,
                    ));
                }
                for (key, given) in [
                    ("pca_every", pca_every.is_some()),
                    ("max_rows_between_pcas", max_rows_between_pcas.is_some()),
                ] {
                    // `pca = 0` is refused above, and says so once.
                    if given && pca.is_none() {
                        no.refuse(format!(
                            "spec {:?}: {key} schedules the refreshes of the principal \
                         components, but pca, the number of them, is not set. Set pca, or \
                         leave {key} out.",
                            self.name
                        ));
                    }
                }
            }
            ModelKind::KMeans {
                k,
                init,
                update_every_rows,
                split_merge,
                split_merge_every_rows,
                dead_frac,
                standardize,
                scale_floor,
                ..
            } => {
                if *k == 0 {
                    no.refuse(format!(
                        "spec {:?}: n_clusters must be at least 1, but it is 0. Set n_clusters \
                     to the number of clusters to fit.",
                        self.name
                    ));
                }
                if let Some(rule) = init
                    && !INIT_WORDS.contains(&rule.as_str())
                {
                    no.refuse(spec_words_refusal(&self.name, "init", &INIT_WORDS, rule));
                }
                if update_every_rows.is_some_and(|v| v == 0) {
                    no.refuse(format!(
                        "spec {:?}: update_every_rows must be at least 1, but it is 0. Set \
                     update_every_rows to 1 or more, where 1 moves the centers on every row.",
                        self.name
                    ));
                }
                if split_merge_every_rows.is_some_and(|v| v == 0) {
                    no.refuse(format!(
                        "spec {:?}: split_merge_every_rows must be at least 1, but it is 0. Set \
                     split_merge_every_rows to the learned rows between split-merge checks, 1 \
                     or more.",
                        self.name
                    ));
                }
                if let Some(v) = split_merge.filter(|v| *v < 0.0 || !v.is_finite()) {
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "split_merge",
                        "",
                        &Got::One(num_label(v)),
                        Some("to learn every row, with no split or merge"),
                        None,
                    ));
                }
                if let Some(v) = dead_frac.filter(|v| *v < 0.0 || !v.is_finite()) {
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "dead_frac",
                        "",
                        &Got::One(num_label(v)),
                        Some("to never re-place a center whose weight has decayed"),
                        Some(0.5),
                    ));
                }
                if let Some(v) = scale_floor.filter(|v| *v < 0.0 || !v.is_finite()) {
                    no.refuse(at_least_zero(
                    &format!("spec {:?}", self.name),
                    "scale_floor",
                    "",
                    &Got::One(num_label(v)),
                    Some(
                        "to measure distances by each feature's exponentially weighted variance alone",
                    ),
                    None,
                ));
                }
                if *standardize == Some(false) && scale_floor.is_some() {
                    no.refuse(format!(
                        "spec {:?}: scale_floor floors the variances that distances are measured \
                     in under standardize=True, but standardize is False, which measures in \
                     the features' own units. Leave scale_floor out, or set \
                     standardize=True.",
                        self.name
                    ));
                }
            }
            ModelKind::Micro {
                eps,
                beta_mu,
                max_clusters,
                prune_every,
                macro_link,
                standardize,
                scale_floor,
                ..
            } => {
                if !(eps.is_finite() && *eps > 0.0) {
                    no.refuse(format!(
                        "spec {:?}: eps must be finite and above 0, but it is {}. Set eps to \
                     the largest spread along each feature that one of the small clusters the \
                     model keeps may have, in the feature's standard deviations (its own \
                     units under standardize=False), such as 0.3.",
                        self.name,
                        num_label(*eps)
                    ));
                }
                if let Some(v) = beta_mu.filter(|v| !(v.is_finite() && *v > 0.0)) {
                    no.refuse(format!(
                        "spec {:?}: beta_mu must be finite and above 0, but it is {}. Set \
                     beta_mu to the weight, in rows of the stream's mean weight, at which one \
                     of the small clusters the model keeps stops counting as an outlier, such \
                     as 3.",
                        self.name,
                        num_label(v)
                    ));
                }
                if max_clusters.is_some_and(|v| v == 0) {
                    no.refuse(format!(
                        "spec {:?}: max_clusters must be at least 1, but it is 0. Set \
                     max_clusters to the most small clusters the model may keep at once.",
                        self.name
                    ));
                }
                if let Some(e) = prune_every
                    .as_ref()
                    .filter(|e| !non_negative(e.value()) || !e.value().is_finite())
                {
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "prune_every",
                        " clock units",
                        &Got::One(e.to_string()),
                        Some("to prune and join, on every row, the small clusters the model keeps"),
                        None,
                    ));
                }
                if let Some(v) = macro_link.filter(|v| *v < 0.0 || !v.is_finite()) {
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "macro_link",
                        "",
                        &Got::One(num_label(v)),
                        Some("to join none of the small clusters the model keeps"),
                        None,
                    ));
                }
                if let Some(v) = scale_floor.filter(|v| *v < 0.0 || !v.is_finite()) {
                    no.refuse(at_least_zero(
                    &format!("spec {:?}", self.name),
                    "scale_floor",
                    "",
                    &Got::One(num_label(v)),
                    Some(
                        "to measure distances by each feature's exponentially weighted variance alone",
                    ),
                    None,
                ));
                }
                if *standardize == Some(false) && scale_floor.is_some() {
                    no.refuse(format!(
                        "spec {:?}: scale_floor floors the variances that distances are measured \
                     in under standardize=True, but standardize is False, which measures in \
                     the features' own units. Leave scale_floor out, or set \
                     standardize=True.",
                        self.name
                    ));
                }
            }
            ModelKind::EwClass {
                classes,
                covariance_type,
                precision_prior,
                window_size: window,
                // Checked with every windowed kind's (`window_cadence`).
                snapshot_every: _,
                max_rows_between_snapshots: _,
                window_budget: _,
                closed: _,
            } => {
                if let Some(w) = window
                    && (!w.value().is_finite() || w.value() <= 0.0)
                {
                    no.refuse(format!(
                        "spec {:?}: window_size must be finite and above 0, but it is {w}. Set \
                     window_size to the age, in clock units, past which a row counts for \
                     nothing, or leave it out for no hard window.",
                        self.name
                    ));
                }
                if self.targets.len() != 1 {
                    no.refuse(format!(
                        "spec {:?}: ew_class reads one target, the column of classes, but \
                     targets names {}. Set targets to that column alone.",
                        self.name,
                        columns_counted(self.targets.len())
                    ));
                }
                if classes.len() < 2 {
                    no.refuse(format!(
                        "spec {:?}: classes must list at least 2 classes, but it lists {}. List \
                     every class the column of classes holds in classes.",
                        self.name,
                        classes.len()
                    ));
                }
                let mut seen = std::collections::HashSet::new();
                if let Some(dup) = classes.iter().find(|c| !seen.insert(c.as_str())) {
                    no.refuse(format!(
                        "spec {:?}: classes must name each class once, but it lists {dup:?} \
                     more than once. Remove the repeats from classes.",
                        self.name
                    ));
                }
                if classes.iter().any(|c| c.is_empty()) {
                    no.refuse(format!(
                        "spec {:?}: classes must name each class, but one of its names is \
                     empty. Give that class a name.",
                        self.name
                    ));
                }
                if let Some(c) = covariance_type {
                    no.take(
                        online_core::Covariance::parse(c)
                            .map(|_| ())
                            .map_err(|e| format!("spec {:?}: {e}", self.name)),
                    );
                }
                if !(precision_prior.is_finite() && *precision_prior > 0.0) {
                    no.refuse(format!(
                        "spec {:?}: precision_prior must be finite and above 0, but it is {}. \
                     Set precision_prior to the positive ridge each class's covariance \
                     starts with.",
                        self.name,
                        num_label(*precision_prior)
                    ));
                }
            }
            // Checked above, with the features: its refusals come before the
            // shared checks so that `half_life` is named for what it is here.
            ModelKind::SeqTest { .. } => {}
            // The shared checks (non-empty features and targets, no column on
            // both sides, a decay, `min_weight` per target, no residual
            // diagnostics) and one of its own: given edges and the knobs for
            // learning them are two ways of saying where the bins are, and a
            // spec that says both is a spec with a mistake in it. The core
            // checks the edges and the lags themselves.
            ModelKind::Marginal {
                bins,
                bin_rule,
                bin_warm_rows,
                bin_edges,
                bin_budget,
                bin_stat,
                bin_huber_delta,
                bin_refit_rows,
                refit_budget,
                window_size: window,
                lags,
                cross_lags,
                window_lags,
                feature_moments,
                ..
            } => {
                /// `a`, `a and b`, `a, b and c`.
                fn listed_and(v: &[&str]) -> String {
                    match v.split_last() {
                        Some((last, rest)) if !rest.is_empty() => {
                            format!("{} and {last}", rest.join(", "))
                        }
                        _ => v.join(""),
                    }
                }
                // Where the feature moments are kept (docs/PLAN.md task 125).
                // The core refuses `"shared"` with a window too, and says
                // why; here it is refused where the spec is made. With lags
                // it is taken: each feature's autocovariance at each lag is
                // then kept once, beside the shared moments.
                match feature_moments.as_deref() {
                    None | Some("per_target") => {}
                    Some("shared") if window.is_some() => {
                        no.refuse(format!(
                            "spec {:?}: feature_moments=\"shared\" and window_size cannot both \
                         be set: the hard window removes a row's moments about each \
                         feature's mean over its target's rows, and the shared mean also \
                         moves on rows where that target is null, so the moments the hard \
                         window removes would not match the ones it added. Set \
                         feature_moments=\"per_target\", or leave out window_size.",
                            self.name
                        ));
                    }
                    Some("shared") => {}
                    Some(other) => {
                        no.refuse(spec_words_refusal(
                            &self.name,
                            "feature_moments",
                            &["per_target", "shared"],
                            other,
                        ));
                    }
                }
                // An empty list asks for nothing, as `ew_cov`'s does.
                if lags.as_ref().is_some_and(Vec::is_empty) {
                    no.refuse(format!(
                        "spec {:?}: lags must hold at least one lag, but it is empty. Give at \
                     least one lag, such as [1], or leave lags out.",
                        self.name
                    ));
                }
                // Lags under a window cost the snapshot its lag moments, and
                // the spec says so by name before it pays (the user,
                // 2026-09-28: "implement but with an api parameter that
                // documents the impact"; docs/PLAN.md task 137).
                let n_lags = lags.as_ref().map_or(0, Vec::len);
                match (window.is_some(), n_lags > 0, window_lags.unwrap_or(false)) {
                    (true, true, false) => {
                        let (p, t) = (self.k() as f64, self.m() as f64);
                        let l = n_lags as f64;
                        let c = cross_lags.as_ref().map_or(l, |c| c.len() as f64);
                        let base = (3.0 * p + 5.0) * t;
                        let more = 2.0 * l * t + 2.0 * (l + c) * p * t;
                        no.refuse(format!(
                            "spec {:?}: lags under a hard window need window_lags=True, since \
                         each snapshot then also holds the lag moments: {more} values \
                         beside the {base} it holds without them, {} times the size. Set \
                         window_lags=True to accept that, adding cross_lags=[] to hold the \
                         fewest, or leave out window_size or lags.",
                            self.name,
                            crate::figures::three_figures(
                                (base + more) / base,
                                crate::figures::Round::Up
                            )
                        ));
                    }
                    (window_set, lags_set, true) if !(window_set && lags_set) => {
                        no.refuse(format!(
                            "spec {:?}: window_lags applies only with both window_size and lags, \
                         but {}. Leave window_lags out, or set {}.",
                            self.name,
                            match (window_set, lags_set) {
                                (false, false) => "neither is set",
                                (false, true) => "window_size is not set",
                                _ => "lags is not set",
                            },
                            match (window_set, lags_set) {
                                (false, false) => "window_size and lags",
                                (false, true) => "window_size",
                                _ => "lags",
                            }
                        ));
                    }
                    _ => {}
                }
                if bin_edges.is_some()
                    && (bins.is_some() || bin_rule.is_some() || bin_warm_rows.is_some())
                {
                    let learning: Vec<&str> = [
                        ("bins", bins.is_some()),
                        ("bin_rule", bin_rule.is_some()),
                        ("bin_warm_rows", bin_warm_rows.is_some()),
                    ]
                    .into_iter()
                    .filter_map(|(k, given)| given.then_some(k))
                    .collect();
                    no.refuse(format!(
                        "spec {:?}: bin_edges fixes the bins, but {} also set, which {} how to \
                     learn them. Leave out bin_edges, or {}.",
                        self.name,
                        match learning.as_slice() {
                            [one] => format!("{one} is"),
                            many => format!("{} are", listed_and(many)),
                        },
                        if learning.len() == 1 { "sets" } else { "set" },
                        listed_and(&learning)
                    ));
                }
                if bin_budget.is_some() && bins.is_none() && bin_edges.is_none() {
                    no.refuse(format!(
                        "spec {:?}: bin_budget bounds the memory of the bins, but neither bins \
                     nor bin_edges is set. Set bins or bin_edges, or leave bin_budget out.",
                        self.name
                    ));
                }
                // Each bin's location (docs/PLAN.md task 272 (c)): a
                // statistic of the bins, so it needs them; its cut belongs to
                // `"huber"` alone, and a cut given under the mean was taken
                // and ignored.
                match bin_stat.as_deref() {
                    None | Some("mean") | Some("huber") => {}
                    Some(other) => {
                        no.refuse(spec_words_refusal(
                            &self.name,
                            "bin_stat",
                            &["mean", "huber"],
                            other,
                        ));
                    }
                }
                if bin_stat.is_some() && bins.is_none() && bin_edges.is_none() {
                    no.refuse(format!(
                        "spec {:?}: bin_stat sets each bin's location, but neither bins nor \
                     bin_edges is set. Set bins or bin_edges, or leave bin_stat out.",
                        self.name
                    ));
                }
                if bin_huber_delta.is_some() && bin_stat.as_deref() != Some("huber") {
                    no.refuse(format!(
                        "spec {:?}: bin_huber_delta is the cut of bin_stat=\"huber\", but \
                     bin_stat is {:?}. Leave bin_huber_delta out, or set \
                     bin_stat=\"huber\".",
                        self.name,
                        bin_stat.as_deref().unwrap_or("mean")
                    ));
                } else if let Some(d) = bin_huber_delta.filter(|d| !positive(d.0)) {
                    no.refuse(format!(
                        "spec {:?}: bin_huber_delta must be above 0, but it is {}. Set \
                     bin_huber_delta to a positive cut, or to inf for the mean.",
                        self.name,
                        num_label(d.0)
                    ));
                }
                no.take(checks::check_bin_refit(
                    &self.name,
                    bin_stat.as_deref(),
                    *bin_huber_delta,
                    bin_refit_rows.is_some(),
                    *refit_budget,
                ));
                // The lag ring is sized before the first row: `[2^62]`
                // panicked inside the builder (review 2026-10-06, CD10).
                if let Some(&l) = lags.as_ref().and_then(|l| l.iter().max()) {
                    no.take(
                        online_core::check_lag_ceiling("lags", l)
                            .map_err(|e| format!("spec {:?}: {e}", self.name)),
                    );
                }
            }
            // Every parameter check is `DecoCfg::validate`'s, so that the
            // CLI and the bank get the same messages; only
            // the block *names* are resolved here, where the feature list is.
            ModelKind::Bocpd {
                hazard, hazard_col, ..
            } => {
                // The model reads `targets[0]` as the hazard whatever the name
                // here says, so the two must be one column (review
                // 2026-09-12, C20).
                // The slot is the column itself, not a table: a table's name
                // means nothing here, and its column is what the model would
                // read (review 2026-09-26, D4: the names were compared, so a
                // table named like the hazard put another column in the slot).
                if let Some(h) = hazard_col
                    && self.targets.defs() != [crate::targets::TargetDef::plain(h.clone())]
                {
                    no.refuse(format!(
                        "spec {:?}: bocpd reads hazard_col {h:?} as its one target, but targets \
                     is {:?}. Set targets to [{h:?}], the column itself.",
                        self.name,
                        self.targets.as_slice()
                    ));
                }
                if self.half_life.is_some() || self.lam.is_some() {
                    no.refuse(format!(
                        "spec {:?}: {} does not apply to bocpd, which forgets at each break, at \
                     the rate hazard sets. Leave {} out, and set hazard instead.",
                        self.name, decay_key, decay_key
                    ));
                }
                // A column's value is the chance of a break after its row,
                // and a duration's the chance of the step before each row:
                // a null falling back to the duration would put both on one
                // boundary (task 179).
                if let (Some(h), Some(d)) =
                    (hazard_col, hazard.as_ref().filter(|d| d.is_duration()))
                {
                    no.refuse(format!(
                        "spec {:?}: hazard_col {h:?} and a hazard given as a duration ({d}) \
                     cannot both be set: the column gives a chance of a break after each \
                     row, and the duration a chance over each clock step. Give hazard as a \
                     number of rows, or leave out hazard_col.",
                        self.name
                    ));
                }
                no.take(crate::stream::bocpd_cfg(self).map(|_| ()));
            }
            ModelKind::CorrChange { scalar, kind, .. } => {
                if !scalar.unwrap_or(false) && (self.half_life.is_some() || self.lam.is_some()) {
                    // `scalar` itself applies only to the two kinds that
                    // take it, so it is offered only there.
                    no.refuse(if kind.as_deref() == Some("window") {
                        format!(
                            "spec {:?}: {decay_key} does not apply to corrchange under \
                         kind=\"window\", which does not decay. Leave {decay_key} out.",
                            self.name
                        )
                    } else {
                        format!(
                            "spec {:?}: {decay_key} applies to corrchange only under \
                         scalar=True, where it sets how fast the spread each feature is \
                         divided by forgets. Leave {decay_key} out, or set scalar=True.",
                            self.name
                        )
                    });
                }
                // The model's own checks too, `n_perm`'s ceiling among them,
                // so a spec meets them here (review 2026-10-06, CD10).
                // The configuration's own problems stop the model's checks,
                // which read the configuration it builds.
                match crate::stream::corrchange_cfg(self) {
                    Ok(c) => {
                        no.take(c.validate());
                    }
                    Err(e) => {
                        no.take(Err(e));
                        let _ = no.stop_here();
                    }
                }
            }
            ModelKind::Hmm { k, exog_tvtp, .. } => {
                // `k` sizes the transition matrix and the states before the
                // first row, as `kmeans`' sizes its centres; held here before
                // the configuration is built from it (review 2026-10-06,
                // CF2's sibling). The configuration reads it, so a refused
                // `k` stops the model's checks here.
                let k_ok = no.take(online_core::Hmm::check_k(*k));
                // As `bocpd`'s `hazard_col` (review 2026-09-12, C20).
                if let Some(z) = exog_tvtp
                    && self.targets.defs() != [crate::targets::TargetDef::plain(z.clone())]
                {
                    no.refuse(format!(
                        "spec {:?}: hmm reads exog_tvtp {z:?} as its one target, but targets is \
                     {:?}. Set targets to [{z:?}], the column itself.",
                        self.name,
                        self.targets.as_slice()
                    ));
                }
                if k_ok {
                    no.take(crate::stream::hmm_cfg(self).map(|_| ()));
                } else {
                    let _ = no.stop_here();
                }
            }
            ModelKind::Audit { .. } => {
                if self.half_life.is_some() || self.lam.is_some() {
                    no.refuse(format!(
                        "spec {:?}: {decay_key} does not apply to audit, which counts every row \
                     with no decay. Leave {decay_key} out.",
                        self.name
                    ));
                }
                if self.weight.is_some() {
                    no.refuse(format!(
                        "spec {:?}: weight does not apply to audit, which counts a value \
                     whatever its row weighs. Leave weight out. To audit the weight column, \
                     list it among the columns the audit reads.",
                        self.name
                    ));
                }
                if self.embargo.is_some() {
                    no.refuse(format!(
                        "spec {:?}: embargo does not apply to audit, which has no target to \
                     wait for. Leave embargo out.",
                        self.name
                    ));
                }
                if self.min_weight.is_some() {
                    // Taken and ignored until review 6 (F-10).
                    no.refuse(format!(
                        "spec {:?}: min_weight does not apply to audit, which predicts nothing \
                     to withhold. Leave min_weight out.",
                        self.name
                    ));
                }
                if self.group_close.is_some() {
                    // A closed group's counts left the bank with its stream,
                    // so `audit()` and `check()` never saw them (review 6,
                    // C-3 and F-3).
                    no.refuse(format!(
                        "spec {:?}: group_close does not apply to audit: a closed group's counts \
                     would leave the bank with it, where bank.audit() and bank.check() \
                     cannot read them. Leave group_close out.",
                        self.name
                    ));
                }
                no.take(crate::stream::audit_cfg(self).map(|_| ()));
            }
            ModelKind::Rcov { .. } => {
                if self.group.is_none() || self.group_close.is_none() {
                    const SET_GROUP: &str = "group to the column that names each block";
                    const SET_CLOSE: &str = "group_close to \"monotone\" or \"session\"";
                    let (missing, set) = match (self.group.is_none(), self.group_close.is_none()) {
                        (true, true) => (
                            "group and group_close are",
                            format!("{SET_GROUP}, and {SET_CLOSE}"),
                        ),
                        (true, false) => ("group is", SET_GROUP.to_string()),
                        _ => ("group_close is", SET_CLOSE.to_string()),
                    };
                    no.refuse(format!(
                        "spec {:?}: rcov needs group and group_close, since it writes each \
                     block's covariance when its group closes, but {missing} not set. Set \
                     {set}.",
                        self.name
                    ));
                }
                if self.half_life.is_some() || self.lam.is_some() {
                    no.refuse(format!(
                        "spec {:?}: {decay_key} does not apply to rcov, whose covariance is a \
                     sum over each block that group_close ends, with no decay. Leave \
                     {decay_key} out.",
                        self.name
                    ));
                }
                no.take(crate::stream::rcov_cfg(self).map(|_| ()));
            }
            // The configuration checks the blocks by name, a column that is
            // not a feature among them.
            ModelKind::Deco { .. } => {
                no.take(crate::stream::deco_cfg(self).map(|_| ()));
            }
            ModelKind::Ftrl {
                alpha,
                beta,
                l1,
                l2,
                ..
            } => {
                // At `inf` each is no setting: `alpha` leaves `-z/l2`, and
                // `beta`, `l1` or `l2` zeroes every coordinate for ever
                // (review 2026-09-12, S27).
                if let Some(a) = alpha.filter(|a| *a <= 0.0 || !a.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: alpha must be finite and above 0, but it is {}. Set alpha to \
                     a finite positive learning rate, such as the default 0.1.",
                        self.name,
                        num_label(a)
                    ));
                }
                for (name, v, zero) in [
                    ("beta", beta, None),
                    ("l1", l1, Some("for no L1 penalty")),
                    ("l2", l2, Some("for no L2 penalty")),
                ] {
                    if let Some(v) = v.filter(|v| *v < 0.0 || !v.is_finite()) {
                        no.refuse(at_least_zero(
                            &format!("spec {:?}", self.name),
                            name,
                            "",
                            &Got::One(num_label(v)),
                            zero,
                            None,
                        ));
                    }
                }
            }
            ModelKind::Huber {
                huber_delta,
                huber_scale,
                huber_scale_half_life,
                ridge,
                solve_every,
                feature_sets,
                refit_budget,
                ..
            } => {
                if let Some(d) = huber_delta.filter(|d| !positive(d.0)) {
                    no.refuse(format!(
                        "spec {:?}: huber_delta must be above 0, but it is {}. Set huber_delta \
                     to a positive cut in units of the residuals' scale, or to inf for least \
                     squares.",
                        self.name,
                        num_label(d.0)
                    ));
                }
                no.take(check_huber_scale(
                    &self.name,
                    huber_scale.as_deref(),
                    huber_scale_half_life.as_ref(),
                ));
                no.take(check_ridge(&self.name, *ridge));
                no.take(check_solve_every(&self.name, solve_every.as_ref()));
                no.take(check_feature_sets(
                    &self.name,
                    &self.features,
                    feature_sets.as_deref(),
                ));
                no.take(check_refit_budget(&self.name, *refit_budget));
            }
            ModelKind::Quantile {
                quantile,
                quantile_eps,
                ridge,
                solve_every,
                refit_budget,
                ..
            } => {
                no.take(check_refit_budget(&self.name, *refit_budget));
                if !(0.0 < *quantile && *quantile < 1.0) {
                    no.refuse(format!(
                        "spec {:?}: quantile must be above 0 and below 1, but it is {}. Set \
                     quantile to the level to fit, such as 0.5 for the median.",
                        self.name,
                        num_label(*quantile)
                    ));
                }
                // A floor of `inf` weighs every row 0: a model that never
                // learns (review 2026-09-12, S27).
                if let Some(e) = quantile_eps.filter(|e| !positive(*e) || !e.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: quantile_eps must be finite and above 0, but it is {}. Set \
                     quantile_eps to the half-width of the band the fit steps in, in units \
                     of the residuals' scale.",
                        self.name,
                        num_label(e)
                    ));
                }
                no.take(check_ridge(&self.name, *ridge));
                no.take(check_solve_every(&self.name, solve_every.as_ref()));
            }
            ModelKind::Kalman {
                coef_half_life,
                q,
                obs_var,
                p0,
                revert_half_life,
                ..
            } => {
                let k_total = self.k() + usize::from(self.fit_intercept);
                // The intercept is a coefficient only where the spec fits one.
                let (included, each) = if self.fit_intercept {
                    (", intercept included", "the intercept first")
                } else {
                    ("", "one for each feature in order")
                };
                // One way of giving the process noise: `coef_half_life` was
                // required, and ignored beside `q` (review 2026-10-06, PC6).
                match (coef_half_life, q) {
                    (Some(_), Some(_)) => {
                        no.refuse(format!(
                            "spec {:?}: coef_half_life and q cannot both be set: q gives the \
                         drift of each coefficient outright, and coef_half_life derives it. \
                         Leave out coef_half_life, or q.",
                            self.name
                        ));
                    }
                    (None, None) => {
                        no.refuse(format!(
                            "spec {:?}: kalman needs coef_half_life or q, but neither is set. Set \
                         coef_half_life to how fast a coefficient may drift, or q to its \
                         drift outright.",
                            self.name
                        ));
                    }
                    _ => {}
                }
                // A list one value or one per coefficient long; each value
                // above 0, `inf` meaning `what_inf`.
                // Each value is shown as the spec gave it: a duration as its
                // text, not its seconds.
                let mut per_coef = |key: &str, hs: &[Span], what_inf: &str| {
                    if hs.len() != 1 && hs.len() != k_total {
                        no.refuse(format!(
                            "spec {:?}: {key} must list one value, or one for each of the \
                         {k_total} coefficients{included}, but it lists {}. Give one value, \
                         which every coefficient takes, or a list of {k_total}, {each}.",
                            self.name,
                            hs.len()
                        ));
                    }
                    let bad: Vec<String> = hs
                        .iter()
                        .filter(|h| !positive(h.value()))
                        .map(Span::to_string)
                        .collect();
                    if !bad.is_empty() {
                        let (is, which) = match (hs.len(), bad.len()) {
                            (1, _) => ("it is", key.to_string()),
                            (_, 1) => ("one of its values is", "that value".to_string()),
                            _ => ("these values are not:", "each".to_string()),
                        };
                        no.refuse(format!(
                            "spec {:?}: {key} must be above 0, but {is} {}. Set {which} to a \
                         positive number of clock units, or to inf {what_inf}.",
                            self.name,
                            bad.join(", ")
                        ));
                    }
                };
                if let Some(hl) = coef_half_life {
                    per_coef(
                        "coef_half_life",
                        hl.spans(),
                        "to hold the coefficient fixed",
                    );
                }
                if let Some(rs) = revert_half_life {
                    per_coef(
                        "revert_half_life",
                        rs.spans(),
                        "for a coefficient that drifts with no pull toward 0",
                    );
                }
                if let Some(q) = q.as_ref().filter(|q| q.len() != k_total) {
                    no.refuse(format!(
                        "spec {:?}: q must list one value for each of the {k_total} \
                     coefficients{included}, but it lists {}. Give one value for each \
                     coefficient, {each}.",
                        self.name,
                        q.len()
                    ));
                }
                let bad: Vec<String> = q
                    .iter()
                    .flatten()
                    .filter(|v| !non_negative(v.0) || !v.0.is_finite())
                    .map(|v| num_label(v.0))
                    .collect();
                if !bad.is_empty() {
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "q",
                        "",
                        &Got::Of(bad),
                        Some("for no drift over time"),
                        None,
                    ));
                }
                if let Some(v) = obs_var.filter(|v| !positive(*v) || !v.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: obs_var must be finite and above 0, but it is {}. Set \
                     obs_var to the target's noise variance, or leave it out to estimate it \
                     from the residuals.",
                        self.name,
                        num_label(v)
                    ));
                }
                if let Some(v) = p0.filter(|v| !positive(*v) || !v.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: p0 must be finite and above 0, but it is {}. Set p0 to each \
                     slope's starting variance, as a multiple of the target's noise variance, \
                     such as the default 1.",
                        self.name,
                        num_label(v)
                    ));
                }
            }
            ModelKind::Lasso {
                penalties,
                l1_ratio,
                select_half_life,
                solve_every,
                tol,
                ..
            } => {
                if penalties.is_empty() {
                    no.refuse(format!(
                        "spec {:?}: penalties must list at least one value, but it is empty. Set \
                     penalties to one or more values, largest first.",
                        self.name
                    ));
                }
                let bad: Vec<String> = penalties
                    .iter()
                    .filter(|l| !non_negative(**l) || !l.is_finite())
                    .map(|l| num_label(*l))
                    .collect();
                if !bad.is_empty() {
                    // A list always: `penalties` takes no single number. The
                    // order is checked only among good values, below.
                    no.refuse(at_least_zero(
                        &format!("spec {:?}", self.name),
                        "penalties",
                        "",
                        &Got::Distinct(bad.clone(), penalties.len() > bad.len()),
                        None,
                        None,
                    ));
                } else if let Some(w) = penalties.windows(2).find(|w| w[0] <= w[1]) {
                    // Strictly: a repeated penalty is two identical slots
                    // with the same field name.
                    let but = if w[0] == w[1] {
                        format!("{} is listed twice", num_label(w[0]))
                    } else {
                        format!("{} comes before {}", num_label(w[0]), num_label(w[1]))
                    };
                    no.refuse(format!(
                        "spec {:?}: penalties must run from the largest to the smallest, each \
                     once, but {but}. List each penalty once, largest first.",
                        self.name,
                    ));
                }
                if let Some(r) = l1_ratio.filter(|r| !(0.0..=1.0).contains(r)) {
                    no.refuse(format!(
                        "spec {:?}: l1_ratio must be from 0 to 1, but it is {}. Set l1_ratio to \
                     1 for the lasso, to 0 for ridge, or to a value between 0 and 1 for an \
                     elastic net.",
                        self.name,
                        num_label(r)
                    ));
                }
                if let Some(h) = select_half_life.as_ref().filter(|h| !positive(h.value())) {
                    no.refuse(format!(
                        "spec {:?}: select_half_life must be above 0, but it is {h}. Set \
                     select_half_life to a positive number of clock units, to inf to rank \
                     the penalties on every row alike, or leave it out to rank them by \
                     their errors over the model's half-life.",
                        self.name
                    ));
                }
                no.take(check_solve_every(&self.name, solve_every.as_ref()));
                if let Some(t) = tol.filter(|t| !positive(*t) || !t.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: tol must be finite and above 0, but it is {}. Set tol to how \
                     far a coefficient may still move, as a share of the target's spread, \
                     when a solve stops.",
                        self.name,
                        num_label(t)
                    ));
                }
            }
            ModelKind::Rls { delta, coef_prior } => {
                if let Some(r) = delta.filter(|r| !positive(*r) || !r.is_finite()) {
                    no.refuse(format!(
                        "spec {:?}: delta must be finite and above 0, but it is {}. Set delta to \
                     a positive value: the larger it is, the longer the coefficients stay \
                     near their prior.",
                        self.name,
                        num_label(r)
                    ));
                }
                let k_total = self.k() + usize::from(self.fit_intercept);
                if let Some(c) = coef_prior
                    && (c.len() != self.m() || c.iter().any(|v| v.len() != k_total))
                {
                    no.refuse(coef_prior_refusal(c, k_total));
                }
            }
            ModelKind::EwRidge {
                ridge,
                feature_sets,
                coef_prior,
                session_shrink,
                long_half_life,
                solve_every,
                ..
            } => {
                if let Some(r) = ridge {
                    let rs = r.to_vec();
                    // A negative ridge makes the system indefinite and the
                    // coefficients garbage; NaN/inf make them zero. Zero is
                    // legal: plain least squares, rescued by the jitter
                    // fallback when singular.
                    let bad: Vec<String> = rs
                        .iter()
                        .filter(|r| !non_negative(**r) || !r.is_finite())
                        .map(|r| num_label(*r))
                        .collect();
                    if !bad.is_empty() {
                        no.refuse(at_least_zero(
                            &format!("spec {:?}", self.name),
                            "ridge",
                            "",
                            &Got::list(rs.len(), bad, true),
                            Some(NO_SHRINKAGE),
                            None,
                        ));
                    } else if let Some(dup) = first_duplicate(&rs) {
                        no.refuse(format!(
                            "spec {:?}: ridge must list each value once, since each names its \
                         own fields, but it lists {} more than once. Remove the repeats \
                         from ridge.",
                            self.name,
                            num_label(dup)
                        ));
                    }
                }
                no.take(check_solve_every(&self.name, solve_every.as_ref()));
                if let Some(h) = long_half_life.as_ref().filter(|h| !positive(h.value())) {
                    no.refuse(format!(
                        "spec {:?}: long_half_life must be above 0, but it is {h}. Set \
                     long_half_life to a positive number of clock units, or to inf for a \
                     long-run fit of every row.",
                        self.name
                    ));
                }
                if let Some(f) = session_shrink.filter(|f| !(0.0..=1.0).contains(f)) {
                    no.refuse(format!(
                        "spec {:?}: session_shrink must be from 0 to 1, but it is {}. Set \
                     session_shrink to the share of the long-run fit to blend in at each \
                     session change: 0 for none, 1 for all of it.",
                        self.name,
                        num_label(f)
                    ));
                }
                if session_shrink.is_some() && long_half_life.is_none() {
                    no.refuse(format!(
                        "spec {:?}: session_shrink blends in a long-run fit, but long_half_life, \
                     the half-life of that fit, is not set. Set long_half_life, or leave \
                     session_shrink out.",
                        self.name
                    ));
                }
                if session_shrink.is_some() && self.session.is_none() {
                    no.refuse(format!(
                        "spec {:?}: session_shrink blends at each session change, but session \
                     is not set. Set session to the column whose changes mark the \
                     sessions, or leave session_shrink out.",
                        self.name
                    ));
                }
                // Two prescriptions for one row. Under `session_gap = "reset"`
                // the reset wins and the blend never runs; under `group_close
                // = "session"` the stream starts over at the change, so the
                // twin is never read (review 2026-09-12, S22).
                if session_shrink.is_some() {
                    if matches!(&self.session_gap, Some(SessionGapSpec::Word(w)) if w == "reset") {
                        no.refuse(format!(
                            "spec {:?}: session_shrink and session_gap=\"reset\" cannot both be \
                         set: the reset starts the model over at each session change, so \
                         there is no fit to blend. Leave out session_shrink, or set \
                         session_gap to a number of clock units or a duration.",
                            self.name
                        ));
                    }
                    if self.closes_on_session() {
                        no.refuse(format!(
                            "spec {:?}: session_shrink and group_close=\"session\" cannot both \
                         be set: the group closes at each session change and starts over, \
                         so there is no fit to blend. Leave out session_shrink, or replace \
                         group_close=\"session\" with a session_gap in clock units or a \
                         duration.",
                            self.name
                        ));
                    }
                }
                if long_half_life.is_some() && session_shrink.is_none() {
                    no.refuse(format!(
                        "spec {:?}: long_half_life sets the half-life of the long-run fit \
                     session_shrink blends in, but session_shrink is not set. Set \
                     session_shrink, or leave long_half_life out.",
                        self.name
                    ));
                }
                if let Some(c) = coef_prior {
                    let k_total = self.k() + usize::from(self.fit_intercept);
                    if c.len() != self.m() || c.iter().any(|v| v.len() != k_total) {
                        no.refuse(coef_prior_refusal(c, k_total));
                    }
                }
                no.take(check_feature_sets(
                    &self.name,
                    &self.features,
                    feature_sets.as_deref(),
                ));
            }
        }
    }
}
