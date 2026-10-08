//! The output index: every field a spec writes, with the machine values
//! its name encodes and the buffer `assemble` fills it from. Moved whole
//! out of `bank.rs`, which stood at the source-size cap
//! (`tests/test_repo_hygiene.py`).

use super::*;

/// `ew_cov` output: one f64 column per statistic slot, plus `n_eff`.
/// One output field with the machine values its name encodes.
///
/// This is the antidote to string formatting as API: a caller filters this
/// table for `kind == "pred" && target == "y" && ridge == Some(0.5)` instead
/// of constructing `"pred_y__r0.5@h500"` by hand — which would require
/// reimplementing `num_label`'s float rendering. Produced by the same code
/// that renders the names, so the two cannot drift.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FieldMeta {
    pub field: String,
    /// pred / resid / sigma / zscore / ic / r2 / hit_rate / abs_resid_q /
    /// autocorr / drift / n_eff / coef / lam_selected / selected /
    /// pred_selected / pred_averaged — or an `ew_cov` statistic name.
    pub kind: String,
    pub target: Option<String>,
    /// The instance's decay, as configured (present even when the suffix is
    /// empty because there is a single instance).
    pub half_life: Option<f64>,
    pub lam: Option<f64>,
    pub ridge: Option<f64>,
    pub feature_set: Option<String>,
    /// Lasso path point.
    pub penalty: Option<f64>,
    /// Quantile level (`abs_resid_q*` fields).
    pub quantile: Option<f64>,
    /// Lag, in learned rows, of an `ew_cov` `lag_corr_*` field
    /// (docs/ENHANCEMENTS.md E56); `None` for every other field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag: Option<usize>,
    /// Columns an `ew_cov` statistic is over.
    pub columns: Option<Vec<String>>,
    /// The polars dtype the field is materialized with, as its string form
    /// (`f64`, `bool`, `str`, `list[f64]`). Set from `src`, so it is the same
    /// table `assemble` fills the buffers from, so a declared output struct and
    /// the values that land in it cannot disagree (docs/IMPROVEMENTS.md C1 — a
    /// name-prefix guess once declared `drift_*` as `f64` while the bank
    /// produced `bool`, and polars refused the struct).
    pub dtype: String,
    /// Which assembled buffer, and where in it, this field's values come from.
    /// Private and not serialized: it is how `assemble` walks this schema
    /// instead of rebuilding the same nested loops with its own `format!`
    /// calls (docs/SIMPLIFICATION.md S1).
    #[serde(skip)]
    pub(super) src: Source,
}

/// The buffer a field's values are scattered into, with the index into it.
/// One variant per buffer `assemble` allocates.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) enum Source {
    /// Assigned before the field is pushed; never observed.
    #[default]
    Unset,
    Pred(usize),
    Resid(usize),
    Sigma(usize),
    ResidZ(usize),
    Drift(usize),
    /// `(which of ic/r2/hit_rate, index)`.
    Metric(usize, usize),
    /// `(which of lo/hi/coverage, index)`, laid out like `Metric`.
    Conformal(usize, usize),
    Quantile(usize),
    Autocorr(usize),
    NEff(usize),
    Coef(usize),
    /// `settled_frac`, per instance, in the `settled` buffer
    /// (docs/WARMUP-AND-CONVERGENCE.md §3).
    Settled(usize),
    /// `withheld_reason`, per instance: a code in the `reason` buffer,
    /// materialized as a dictionary-encoded string (a categorical).
    Reason(usize),
    /// `error_inflation_<slot>`, per slot, in the `inflation` buffer.
    Inflation(usize),
    /// `support_coef`, per instance, laid out like `Coef`.
    SupportCoef(usize),
    /// `scored_clock` and `learned_clock`, one pair per spec, in the clock
    /// column's own type (docs/PLAN.md task 152).
    ScoredClock,
    LearnedClock,
    LamSelected(usize),
    SelPred(usize),
    SelName(usize),
    AvgPred(usize),
    /// An `ew_cov` statistic, or a `kmeans` / `micro` distance, which rides
    /// in the `pred` buffer.
    Stat(usize),
    /// A `kmeans` assignment, or a `micro` count: the `pred` buffer holds a
    /// small non-negative integer as an f64 (NaN = null), materialized as
    /// `i32`.
    Cluster(usize),
    /// A `micro` id or label, or a `seqtest` count: monotone and never
    /// reused, so `i64`, the same way.
    Id(usize),
    /// A `micro` flag: `1.0` / `0.0` in the `pred` buffer (NaN = null),
    /// materialized as `Boolean`.
    Flag(usize),
    /// An `ew_class` prediction: the class's position in `classes` as an
    /// f64 in the `pred` buffer (NaN = null), materialized as the class name.
    Label(usize),
}

impl FieldMeta {
    fn new(field: String, kind: &str) -> Self {
        Self {
            field,
            kind: kind.to_string(),
            target: None,
            half_life: None,
            lam: None,
            ridge: None,
            feature_set: None,
            penalty: None,
            quantile: None,
            lag: None,
            columns: None,
            dtype: String::new(),
            src: Source::Unset,
        }
    }
    fn src(mut self, src: Source) -> Self {
        // The clock fields take the clock column's own type, known only
        // when a chunk arrives (task 152).
        let is_clock = matches!(src, Source::ScoredClock | Source::LearnedClock);
        self.src = src;
        self.dtype = if is_clock {
            "clock".to_string()
        } else {
            self.dtype().to_string()
        };
        self
    }

    /// The dtype `assemble` materializes this field with.
    pub fn dtype(&self) -> DataType {
        match self.src {
            Source::Drift(_) => DataType::Boolean,
            Source::SelName(_) => DataType::String,
            Source::Coef(_) | Source::SupportCoef(_) => DataType::List(Box::new(DataType::Float64)),
            Source::Reason(_) => DataType::from_frozen_categories(
                polars::prelude::FrozenCategories::new(crate::stream::WITHHELD_REASONS)
                    .expect("three distinct names"),
            ),
            Source::Cluster(_) => DataType::Int32,
            Source::Id(_) => DataType::Int64,
            Source::Flag(_) => DataType::Boolean,
            Source::Label(_) => DataType::String,
            Source::Unset => unreachable!("every field is given a source in output_index"),
            _ => DataType::Float64,
        }
    }
    fn decay(mut self, d: &online_core::Decay) -> Self {
        match d {
            online_core::Decay::Halflife(h) => self.half_life = Some(*h),
            online_core::Decay::Lam(l) => self.lam = Some(*l),
        }
        self
    }
    fn target(mut self, t: &str) -> Self {
        self.target = Some(t.to_string());
        self
    }
    fn combo(mut self, c: &crate::stream::Combo) -> Self {
        self.ridge = c.ridge;
        self.feature_set = c.feature_set.clone();
        self.penalty = c.lambda;
        self
    }
}

/// Every coefficient a spec reports, in `coef` list order, with the name
/// [`CoefField`] gives it. Empty for `ew_cov` and `seqtest`, which have
/// none, and for `micro`, whose `coef` is one `[id, label, n, radius, c_1 ..
/// c_p]` row per potential summary -- as many as there are, so no position
/// has a name.
///
/// The layout is the models' (`online_core`): per instance, one `coef` list
/// holding `(target, combo)` slots in the order the `pred` fields declare
/// them, each slot the full term vector -- `intercept` first when the spec
/// has one, then every feature, zeros for features a feature set leaves out
/// (`EwRidgeModel::solve` scatters each combo's solution into `k_total`
/// columns) -- or `level`, `trend` for `holt`. Rendered here, beside the
/// field names, from the same combos and suffixes, so the two cannot drift.
///
/// `kmeans` reports its centres here: `k` slots named `cluster{j}` in place
/// of the targets, each the centre's coordinate per feature, so
/// `coef_cluster0_x1` is centre 0's `x1` and `coef_index` lays the list out
/// as `(cluster, feature)`. `ew_class` does the same with its class means:
/// one slot per class, named by the class, so `coef_a_x1` is class `a`'s
/// mean of `x1`.
pub fn coef_fields(spec: &Spec) -> Vec<CoefField> {
    if matches!(
        spec.model,
        crate::ModelKind::EwCov { .. }
            | crate::ModelKind::Micro { .. }
            | crate::ModelKind::SeqTest { .. }
            | crate::ModelKind::Marginal { .. }
            | crate::ModelKind::Rcov { .. }
            | crate::ModelKind::CorrChange { .. }
            | crate::ModelKind::Bocpd { .. }
    ) {
        return Vec::new();
    }
    // deco's `coef` is the correlation values themselves, one slot each,
    // named for the block or the pair of blocks they belong to.
    if matches!(spec.model, crate::ModelKind::Deco { .. }) {
        let names = crate::stream::deco_block_names(spec);
        let labels = online_core::Deco::labels(&names);
        let values: Vec<&String> = labels
            .iter()
            .filter(|l| l.as_str() == "rho" || l.starts_with("rho_"))
            .collect();
        let mut out = Vec::new();
        for (suffix, d) in spec.decays().expect("validated") {
            for (position, l) in values.iter().enumerate() {
                let slot = l.strip_prefix("rho_").unwrap_or("rho");
                out.push(CoefField {
                    field: format!("coef{suffix}"),
                    position,
                    name: format!("coef_{slot}{suffix}"),
                    target: slot.to_string(),
                    half_life: match d {
                        online_core::Decay::Halflife(h) => Some(h),
                        online_core::Decay::Lam(_) => None,
                    },
                    lam: match d {
                        online_core::Decay::Lam(l) => Some(l),
                        online_core::Decay::Halflife(_) => None,
                    },
                    ridge: None,
                    feature_set: None,
                    penalty: None,
                    term: "rho".to_string(),
                });
            }
        }
        return out;
    }
    let slots: Vec<String> = match &spec.model {
        crate::ModelKind::KMeans { k, .. } => (0..*k).map(|j| format!("cluster{j}")).collect(),
        // One slot per hidden state, named as `kmeans` names its centres.
        crate::ModelKind::Hmm { k, .. } => (0..*k).map(|j| format!("state{j}")).collect(),
        crate::ModelKind::EwClass { classes, .. } => classes.clone(),
        _ => spec.targets.to_vec(),
    };
    let terms: Vec<String> = if matches!(spec.model, crate::ModelKind::Holt { .. }) {
        vec!["level".into(), "trend".into()]
    } else if matches!(
        spec.model,
        crate::ModelKind::KMeans { .. }
            | crate::ModelKind::EwClass { .. }
            | crate::ModelKind::Hmm { .. }
    ) {
        spec.features.clone()
    } else {
        let mut t = Vec::with_capacity(spec.features.len() + 1);
        if spec.fit_intercept {
            t.push("intercept".to_string());
        }
        t.extend(spec.features.iter().cloned());
        t
    };
    let combos = crate::stream::combos(spec);
    let decays = spec.decays().expect("validated");
    let mut out = Vec::new();
    for (suffix, d) in &decays {
        let (half_life, lam) = match d {
            online_core::Decay::Halflife(h) => (Some(*h), None),
            online_core::Decay::Lam(l) => (None, Some(*l)),
        };
        let mut position = 0;
        for t in &slots {
            for c in &combos {
                for term in &terms {
                    out.push(CoefField {
                        field: format!("coef{suffix}"),
                        position,
                        name: format!("coef_{t}_{term}{}{suffix}", c.label),
                        target: t.clone(),
                        half_life,
                        lam,
                        ridge: c.ridge,
                        feature_set: c.feature_set.clone(),
                        penalty: c.lambda,
                        term: term.clone(),
                    });
                    position += 1;
                }
            }
        }
    }
    out
}

/// The refusal for two outputs that render to one field name, naming the
/// inputs that collided -- a target, a feature, a grid label -- since the
/// rendered names are the user's handle on every output, and a duplicate
/// inside one struct would otherwise surface much later as a polars error.
/// `Spec::validate` refuses every way the grammar has of making one that it
/// can see (a feature-set name given twice got through until review
/// 2026-09-12, S7; a target `z_y` beside `y` under `emit_resid_z` until task
/// 144, which renamed the field); the names that remain are the columns'
/// own, as `ew_cov`'s `corr_a_b_c` over `a_b, c` and over `a, b_c`.
pub fn duplicate_field(spec: &Spec) -> Option<String> {
    let index = output_index(spec);
    let mut seen: HashMap<&str, &FieldMeta> = HashMap::with_capacity(index.len());
    for f in &index {
        let Some(first) = seen.insert(f.field.as_str(), f) else {
            continue;
        };
        let describe = |m: &FieldMeta| {
            let mut s = format!("{:?}", m.kind);
            if let Some(t) = &m.target {
                s.push_str(&format!(" of target {t:?}"));
            }
            if let Some(c) = &m.columns {
                s.push_str(&format!(" over columns {c:?}"));
            }
            if let Some(fs) = &m.feature_set {
                s.push_str(&format!(" in feature set {fs:?}"));
            }
            if let Some(r) = m.ridge {
                s.push_str(&format!(" at ridge {r}"));
            }
            if let Some(h) = m.half_life {
                s.push_str(&format!(" at half_life {h}"));
            }
            s
        };
        return Some(format!(
            "spec {:?}: two outputs render to the same field name {:?}: {} and {}; rename a \
             target, feature or grid label to tell them apart",
            spec.name,
            f.field,
            describe(first),
            describe(f)
        ));
    }
    None
}

/// Output field names for a spec, in struct order (used by Python for dtypes).
pub fn output_fields(spec: &Spec) -> Vec<String> {
    output_index(spec).into_iter().map(|m| m.field).collect()
}

/// Every output field with its metadata, in struct order.
///
/// The readiness fields (docs/WARMUP-AND-CONVERGENCE.md §3) ride on every
/// model that writes a row: `settled_frac` and `withheld_reason` follow
/// each instance's `n_eff`, and `support_coef` follows `coef` where the
/// model has one -- inserted here, once, rather than in each model's own
/// layout. The state-only models (`marginal`, `rcov`) write no row and
/// get none.
pub fn output_index(spec: &Spec) -> Vec<FieldMeta> {
    let base = output_index_base(spec);
    if matches!(
        spec.model,
        crate::ModelKind::Marginal { .. } | crate::ModelKind::Rcov { .. }
    ) {
        return with_clocks(spec, base);
    }
    let mut fields = Vec::with_capacity(base.len() + 3 * spec.decays().map_or(1, |d| d.len()));
    for f in base {
        // The instance the field belongs to, as its `n_eff` or `coef`
        // carries it: the same suffix and the same decay (none, for a model
        // that does not decay -- `seqtest`'s fields carry no half-life).
        let (src, suffix, half_life, lam) = (f.src.clone(), f.field.clone(), f.half_life, f.lam);
        fields.push(f);
        let like = |mut m: FieldMeta| {
            m.half_life = half_life;
            m.lam = lam;
            m
        };
        match src {
            Source::NEff(mi) => {
                let suffix = suffix.strip_prefix("weight_sum").unwrap_or("");
                fields.push(like(
                    FieldMeta::new(format!("settled_frac{suffix}"), "settled_frac")
                        .src(Source::Settled(mi)),
                ));
                fields.push(like(
                    FieldMeta::new(format!("withheld_reason{suffix}"), "withheld_reason")
                        .src(Source::Reason(mi)),
                ));
            }
            Source::Coef(mi) if spec.has_support_coef() => {
                let suffix = suffix.strip_prefix("coef").unwrap_or("");
                fields.push(like(
                    FieldMeta::new(format!("support_coef{suffix}"), "support_coef")
                        .src(Source::SupportCoef(mi)),
                ));
            }
            _ => {}
        }
    }
    with_clocks(spec, fields)
}

/// The clock fields, last, under `emit_clocks` (docs/PLAN.md task 152).
fn with_clocks(spec: &Spec, mut fields: Vec<FieldMeta>) -> Vec<FieldMeta> {
    if spec.emit_clocks {
        fields.push(FieldMeta::new("scored_clock".into(), "scored_clock").src(Source::ScoredClock));
        fields.push(
            FieldMeta::new("learned_clock".into(), "learned_clock").src(Source::LearnedClock),
        );
    }
    fields
}

/// [`output_index`] before the readiness fields: each model's own layout.
fn output_index_base(spec: &Spec) -> Vec<FieldMeta> {
    let decays = spec.decays().expect("validated");
    // deco is not a regression either: its slots are the row's own
    // equicorrelation estimate, the level before the row and the row's
    // log-density under it -- one of each unblocked, one `u_*` and one
    // `rho_*` per value and a single `loglik` with blocks.
    if matches!(spec.model, crate::ModelKind::Deco { .. }) {
        let names = crate::stream::deco_block_names(spec);
        let labels = online_core::Deco::labels(&names);
        let n_slots = labels.len();
        // The columns a slot is over: its block, its pair of blocks, or all
        // of them for `loglik` and the unblocked scalars.
        let blocks = crate::stream::deco_cfg(spec).expect("validated");
        let all = spec.features.clone();
        let cols_of = |slot: usize| -> Vec<String> {
            if names.is_empty() {
                return all.clone();
            }
            let k = names.len();
            let m = k + k * (k - 1) / 2;
            if slot >= 2 * m {
                return all.clone();
            }
            let v = slot % m;
            let of = |b: usize| -> Vec<String> {
                blocks.blocks[b].iter().map(|&i| all[i].clone()).collect()
            };
            if v < k {
                return of(v);
            }
            let mut p = k;
            for i in 0..k {
                for j in (i + 1)..k {
                    if p == v {
                        let mut cols = of(i);
                        cols.extend(of(j));
                        return cols;
                    }
                    p += 1;
                }
            }
            all.clone()
        };
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                // The kind is the label without its block suffix, so
                // `u_tech` and `u_banks` are both `u` to a reader of the
                // index.
                let kind = l.split('_').next().unwrap_or(l).to_string();
                let mut m = FieldMeta::new(format!("{l}{suffix}"), &kind)
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols_of(slot));
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // ew_cov is not a regression: its slots are named statistics, not
    // pred/resid pairs, and it has no targets or coefficients.
    if let crate::ModelKind::EwCov {
        stats,
        mahal_quantiles,
        pca,
        lags,
        ..
    } = &spec.model
    {
        let names = stats
            .clone()
            .unwrap_or_else(|| vec!["mean".into(), "std".into(), "corr".into()]);
        let kinds: Vec<online_core::EwCovStat> = names
            .iter()
            .map(|s| match s.as_str() {
                "mean" => online_core::EwCovStat::Mean,
                "var" => online_core::EwCovStat::Var,
                "std" => online_core::EwCovStat::Std,
                "cov" => online_core::EwCovStat::Cov,
                "partial_corr" => online_core::EwCovStat::PartialCorr,
                "mahal" => online_core::EwCovStat::Mahal,
                "lag_corr" => online_core::EwCovStat::LagCorr,
                _ => online_core::EwCovStat::Corr,
            })
            .collect();
        let levels: Vec<f64> = mahal_quantiles.clone().unwrap_or_default();
        let r = pca.unwrap_or(0);
        let lags: Vec<usize> = lags.clone().unwrap_or_default();
        let labels = online_core::EwCovModel::labels(&spec.features, &kinds, &levels, r, &lags);
        // Statistic kind, the columns it is over and its quantile level, in
        // label order: the same walk `labels` makes (per stat: each column,
        // each i<j pair, or all of them; then the levels; then `k + 3` per
        // component, all over every column).
        let all = spec.features.clone();
        // Statistic kind, the columns it is over, its quantile level and its
        // lag, in label order.
        type StatMeta = (String, Vec<String>, Option<f64>, Option<usize>);
        let mut meta: Vec<StatMeta> = Vec::new();
        for (name, kind) in names.iter().zip(&kinds) {
            match kind {
                online_core::EwCovStat::Mean
                | online_core::EwCovStat::Var
                | online_core::EwCovStat::Std => {
                    for col in &spec.features {
                        meta.push((name.clone(), vec![col.clone()], None, None));
                    }
                }
                online_core::EwCovStat::Mahal => meta.push((name.clone(), all.clone(), None, None)),
                // One entry per (lag, ordered pair), the auto terms
                // included: both orientations, since the lagged matrix is
                // not symmetric.
                online_core::EwCovStat::LagCorr => {
                    for &l in &lags {
                        for a in &spec.features {
                            for b in &spec.features {
                                meta.push((
                                    name.clone(),
                                    vec![a.clone(), b.clone()],
                                    None,
                                    Some(l),
                                ));
                            }
                        }
                    }
                }
                _ => {
                    for i in 0..spec.features.len() {
                        for j in (i + 1)..spec.features.len() {
                            meta.push((
                                name.clone(),
                                vec![spec.features[i].clone(), spec.features[j].clone()],
                                None,
                                None,
                            ));
                        }
                    }
                }
            }
        }
        for &q in &levels {
            meta.push(("mahal_q".into(), all.clone(), Some(q), None));
        }
        for _ in 0..r {
            meta.push(("pc_var".into(), all.clone(), None, None));
            meta.push(("pc_share".into(), all.clone(), None, None));
            for col in &spec.features {
                meta.push(("pc_loading".into(), vec![col.clone()], None, None));
            }
            meta.push(("pc_score".into(), all.clone(), None, None));
        }
        debug_assert_eq!(meta.len(), labels.len());
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, (l, (kind, cols, q, lag))) in labels.iter().zip(&meta).enumerate() {
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + slot));
                m.columns = Some(cols.clone());
                m.quantile = *q;
                m.lag = *lag;
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // kmeans is not a regression either: per instance, the nearest centre
    // and two distances (read before the row is learned), `n_eff`, and the
    // centres as `coef`.
    if matches!(spec.model, crate::ModelKind::KMeans { .. }) {
        let n_slots = 3;
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            fields.push(over(
                FieldMeta::new(format!("cluster{suffix}"), "cluster")
                    .decay(d)
                    .src(Source::Cluster(mi * n_slots)),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist{suffix}"), "dist")
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + 1)),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist_second{suffix}"), "dist_second")
                    .decay(d)
                    .src(Source::Stat(mi * n_slots + 2)),
            ));
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // micro: per instance, the nearest cluster's label and distance, the
    // micro-cluster id the row goes to, an outlier flag, two counts (all
    // read before the row is learned), `n_eff`, and the potential summaries
    // as `coef`.
    if matches!(spec.model, crate::ModelKind::Micro { .. }) {
        let n_slots = 6;
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            let at = |slot: usize| mi * n_slots + slot;
            fields.push(over(
                FieldMeta::new(format!("cluster{suffix}"), "cluster")
                    .decay(d)
                    .src(Source::Id(at(0))),
            ));
            fields.push(over(
                FieldMeta::new(format!("dist{suffix}"), "dist")
                    .decay(d)
                    .src(Source::Stat(at(1))),
            ));
            fields.push(over(
                FieldMeta::new(format!("micro_id{suffix}"), "micro_id")
                    .decay(d)
                    .src(Source::Id(at(2))),
            ));
            fields.push(over(
                FieldMeta::new(format!("outlier{suffix}"), "outlier")
                    .decay(d)
                    .src(Source::Flag(at(3))),
            ));
            fields.push(over(
                FieldMeta::new(format!("n_clusters{suffix}"), "n_clusters")
                    .decay(d)
                    .src(Source::Cluster(at(4))),
            ));
            fields.push(over(
                FieldMeta::new(format!("n_micro{suffix}"), "n_micro")
                    .decay(d)
                    .src(Source::Cluster(at(5))),
            ));
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // bocpd's value is a posterior over run lengths: per instance, the
    // changepoint mass, the run-length mode and mean, a predictive mean per
    // column and the row's log score.
    if matches!(spec.model, crate::ModelKind::Bocpd { .. }) {
        let labels = online_core::Bocpd::labels(&spec.features);
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                let src = if l == "run_mode" {
                    Source::Id(at)
                } else {
                    Source::Stat(at)
                };
                let kind = l.split('_').next().unwrap_or(l);
                let mut m = FieldMeta::new(format!("{l}{suffix}"), kind)
                    .decay(d)
                    .src(src);
                m.columns = Some(spec.features.clone());
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // corrchange is a test, not a model of the data: per instance the
    // statistic, its critical value, the flag, the rows since the last one
    // and, on a flag, the rows since the change it dates. Nothing is
    // reported except where a statistic is due.
    if matches!(spec.model, crate::ModelKind::CorrChange { .. }) {
        let labels = online_core::CorrChange::labels();
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                let src = match l.as_str() {
                    "flag" => Source::Flag(at),
                    "since_flag" | "since_change" => Source::Id(at),
                    _ => Source::Stat(at),
                };
                let mut m = FieldMeta::new(format!("{l}{suffix}"), l).decay(d).src(src);
                m.columns = Some(spec.features.clone());
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
        }
        return fields;
    }
    // hmm's state is hidden: per instance, the filtered and predicted
    // posteriors, the state, the row's log-likelihood, `n_eff` and the
    // state means as `coef`.
    if let crate::ModelKind::Hmm { k, .. } = &spec.model {
        let labels = online_core::Hmm::labels(*k);
        let n_slots = labels.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            for (slot, l) in labels.iter().enumerate() {
                let at = mi * n_slots + slot;
                // `state` is a small count, so it rides in `pred` and comes
                // out as an `i32`.
                let src = if l == "state" {
                    Source::Cluster(at)
                } else {
                    Source::Stat(at)
                };
                let mut m =
                    FieldMeta::new(format!("{l}{suffix}"), l.split('_').next().unwrap_or(l))
                        .decay(d)
                        .src(src);
                m.columns = Some(spec.features.clone());
                fields.push(m);
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // ew_class predicts a label, not a number: per instance, the class the
    // row is assigned to and one posterior per class (read before the row
    // is learned), `n_eff`, and the class means as `coef`.
    if let crate::ModelKind::EwClass { classes, .. } = &spec.model {
        let n_slots = 1 + classes.len();
        let mut fields = Vec::new();
        for (mi, (suffix, d)) in decays.iter().enumerate() {
            let over = |mut f: FieldMeta| {
                f.columns = Some(spec.features.clone());
                f
            };
            fields.push(over(
                FieldMeta::new(format!("class{suffix}"), "class")
                    .decay(d)
                    .target(&spec.targets[0])
                    .src(Source::Label(mi * n_slots)),
            ));
            for (c, class) in classes.iter().enumerate() {
                fields.push(over(
                    FieldMeta::new(format!("p_{class}{suffix}"), "p")
                        .decay(d)
                        .target(&spec.targets[0])
                        .src(Source::Stat(mi * n_slots + 1 + c)),
                ));
            }
            fields.push(
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi)),
            );
            fields.push(
                FieldMeta::new(format!("coef{suffix}"), "coef")
                    .decay(d)
                    .src(Source::Coef(mi)),
            );
        }
        return fields;
    }
    // seqtest predicts nothing: per target, the two log e-values and the two
    // sign counts they are staked on (read before the row is learned), then
    // `n_eff`. One undecayed instance, so no suffix and no decay on the
    // fields; no `coef`. A comparison names its fields by the two sides.
    if let crate::ModelKind::SeqTest { .. } = &spec.model {
        let n_slots = online_core::SEQTEST_SLOTS;
        let names: [&str; 4] = if spec.model.compares().is_some() {
            ["log_e_a", "log_e_b", "wins_a", "wins_b"]
        } else {
            ["log_e_pos", "log_e_neg", "n_pos", "n_neg"]
        };
        let mut fields = Vec::new();
        for (t_i, t) in spec.targets.iter().enumerate() {
            let at = |slot: usize| t_i * n_slots + slot;
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[0]), names[0])
                    .target(t)
                    .src(Source::Stat(at(0))),
            );
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[1]), names[1])
                    .target(t)
                    .src(Source::Stat(at(1))),
            );
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[2]), names[2])
                    .target(t)
                    .src(Source::Id(at(2))),
            );
            fields.push(
                FieldMeta::new(format!("{}_{t}", names[3]), names[3])
                    .target(t)
                    .src(Source::Id(at(3))),
            );
        }
        fields.push(FieldMeta::new("weight_sum".into(), "weight_sum").src(Source::NEff(0)));
        return fields;
    }
    // rcov and marginal emit nothing per row but `n_eff`, one per instance:
    // rcov's value is the block it emits at the group's close, marginal's
    // are the pairs `Bank::marginal` reads from the state.
    if matches!(
        spec.model,
        crate::ModelKind::Marginal { .. } | crate::ModelKind::Rcov { .. }
    ) {
        return decays
            .iter()
            .enumerate()
            .map(|(mi, (suffix, d))| {
                FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                    .decay(d)
                    .src(Source::NEff(mi))
            })
            .collect();
    }
    let combos = crate::stream::combos(spec);
    let (nc, m, n_models) = (combos.len(), spec.m(), decays.len());
    let mut fields = Vec::new();
    for (mi, (suffix, d)) in decays.iter().enumerate() {
        // `dst` is the flat (instance, target, combo) index every per-slot
        // buffer in `assemble` is laid out by. Computing it here, once, is
        // what lets `assemble` be a walk over this vector instead of the same
        // nested loops written a second time.
        let mk = |kind: &str, t: &str, c: &crate::stream::Combo, src: Source| {
            FieldMeta::new(format!("{kind}_{t}{}{suffix}", c.label), kind)
                .decay(d)
                .target(t)
                .combo(c)
                .src(src)
        };
        let dst = |t_i: usize, c_i: usize| mi * m * nc + t_i * nc + c_i;
        for (t_i, t) in spec.targets.iter().enumerate() {
            for (c_i, c) in combos.iter().enumerate() {
                fields.push(mk("pred", t, c, Source::Pred(dst(t_i, c_i))));
                fields.push(mk("resid", t, c, Source::Resid(dst(t_i, c_i))));
            }
        }
        if spec.emit_error_inflation {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk(
                        "error_inflation",
                        t,
                        c,
                        Source::Inflation(dst(t_i, c_i)),
                    ));
                }
            }
        }
        if spec.emit_sigma {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("sigma", t, c, Source::Sigma(dst(t_i, c_i))));
                }
            }
        }
        if spec.emit_zscore {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("zscore", t, c, Source::ResidZ(dst(t_i, c_i))));
                }
            }
        }
        if spec.conformal.is_some() {
            // Not `pred_lo`: `pred_` is the prefix that marks a prediction,
            // for `eval.unpack` and for the README's grammar alike.
            for (k, name) in ["lo", "hi", "coverage"].into_iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        fields.push(mk(name, t, c, Source::Conformal(k, dst(t_i, c_i))));
                    }
                }
            }
        }
        if spec.emit_metrics {
            for (k, name) in ["ic", "r2", "hit_rate"].into_iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        fields.push(mk(name, t, c, Source::Metric(k, dst(t_i, c_i))));
                    }
                }
            }
        }
        if let Some(levels) = &spec.resid_quantiles {
            for (li, q) in levels.iter().enumerate() {
                for (t_i, t) in spec.targets.iter().enumerate() {
                    for (c_i, c) in combos.iter().enumerate() {
                        let name = format!(
                            "abs_resid_q{}_{t}{}{suffix}",
                            crate::spec::num_label(*q),
                            c.label
                        );
                        let idx = (li * n_models + mi) * m * nc + t_i * nc + c_i;
                        let mut f = FieldMeta::new(name, "abs_resid_q")
                            .decay(d)
                            .target(t)
                            .combo(c)
                            .src(Source::Quantile(idx));
                        f.quantile = Some(*q);
                        fields.push(f);
                    }
                }
            }
        }
        if spec.emit_autocorr {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("autocorr", t, c, Source::Autocorr(dst(t_i, c_i))));
                }
            }
        }
        if spec.emit_drift {
            for (t_i, t) in spec.targets.iter().enumerate() {
                for (c_i, c) in combos.iter().enumerate() {
                    fields.push(mk("drift", t, c, Source::Drift(dst(t_i, c_i))));
                }
            }
        }
        fields.push(
            FieldMeta::new(format!("weight_sum{suffix}"), "weight_sum")
                .decay(d)
                .src(Source::NEff(mi)),
        );
        fields.push(
            FieldMeta::new(format!("coef{suffix}"), "coef")
                .decay(d)
                .src(Source::Coef(mi)),
        );
        if matches!(spec.model, crate::ModelKind::Lasso { .. }) {
            for (t_i, t) in spec.targets.iter().enumerate() {
                fields.push(
                    FieldMeta::new(format!("penalty_selected_{t}{suffix}"), "penalty_selected")
                        .decay(d)
                        .target(t)
                        .src(Source::LamSelected(mi * m + t_i)),
                );
            }
        }
    }
    if spec.emit_selected {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(
                FieldMeta::new(format!("pred_{t}__selected"), "pred_selected")
                    .target(t)
                    .src(Source::SelPred(t_i)),
            );
            fields.push(
                FieldMeta::new(format!("selected_{t}"), "selected")
                    .target(t)
                    .src(Source::SelName(t_i)),
            );
        }
    }
    if spec.emit_averaged {
        for (t_i, t) in spec.targets.iter().enumerate() {
            fields.push(
                FieldMeta::new(format!("pred_{t}__averaged"), "pred_averaged")
                    .target(t)
                    .src(Source::AvgPred(t_i)),
            );
        }
    }
    fields
}

/// Labels for every prediction slot of one target, across half-life instances
/// and combos, in the order the slots appear.
pub(super) fn slot_labels(spec: &Spec) -> Vec<String> {
    let decays = spec.decays().expect("validated");
    let combos = combo_labels(spec);
    let mut out = Vec::new();
    for (suffix, _) in &decays {
        for c in &combos {
            let label = format!("{c}{suffix}");
            out.push(if label.is_empty() {
                "default".to_string()
            } else {
                label.trim_start_matches("__").to_string()
            });
        }
    }
    out
}
