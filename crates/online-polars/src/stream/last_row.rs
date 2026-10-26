//! The last row a stream learned from, kept with its state ([`LastRow`]),
//! moved out of `stream.rs`, which had passed the size a source file is
//! held to (task 278, batch 17).

use super::*;

/// One row of a [`ChunkOut`], as a value: the output-struct fields of the
/// last row a stream learned from, kept with its state so that a bank loaded
/// from a file reports the diagnostics as they stood -- `pred`, `resid`,
/// `sigma`, the metrics, the residual quantiles, `n_eff`, `coef`, ... --
/// without the output frame (docs/PLAN.md task 34). Every field is the
/// chunk's buffer with `n_rows = 1`, so the row goes out through
/// [`LastRow::take`] and back in through [`LastRow::to_chunk`] by the one
/// layout `assemble` reads, and a value it holds is the value the output
/// row had, to the bit.
///
/// Every float vector here is annotated for the JSON export
/// ([`online_core::humanfloat`]): these are *diagnostics*, and `NaN` is an
/// ordinary value among them -- an unsupervised model has no `pred` or
/// `resid`, and `sigma` is `NaN` until the warm-up ends -- so this is the
/// struct that loses the most to a naive encoding. The msgpack the state
/// file uses is unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LastRow {
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub pred: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub resid: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub sigma: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub zscore: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub autocorr: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub metrics: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub conformal: Vec<f64>,
    /// Task 221's values; empty, and not written, without them.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        with = "online_core::humanfloat::vec_f64_or_tag"
    )]
    pub checks: Vec<f64>,
    /// Task 221 (g)'s feature health; empty, and not written, without it.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        with = "online_core::humanfloat::vec_f64_or_tag"
    )]
    pub health: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub resid_q: Vec<f64>,
    pub drift: Vec<bool>,
    /// One each under `emit_clocks`, else empty (task 152).
    #[serde(default)]
    pub scored_clock: Vec<Option<ClockValue>>,
    #[serde(default)]
    pub learned_clock: Vec<Option<ClockValue>>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub n_eff: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub lam_selected: Vec<f64>,
    /// A slot no solve has fit is NaN (review round 4, CC1), as is the mean
    /// of an `ew_class` class no row has carried, so the export tags them.
    #[serde(with = "online_core::humanfloat::vec_opt_vec_f64_or_tag")]
    pub coef: Vec<Option<Vec<f64>>>,
    #[serde(default, with = "online_core::humanfloat::vec_f64_or_tag")]
    pub settled: Vec<f64>,
    #[serde(default)]
    pub reason: Vec<u8>,
    #[serde(default, with = "online_core::humanfloat::vec_f64_or_tag")]
    pub inflation: Vec<f64>,
    /// The intercept's share is NaN by definition, so the export tags it.
    #[serde(default, with = "online_core::humanfloat::vec_opt_vec_f64_or_tag")]
    pub support_coef: Vec<Option<Vec<f64>>>,
    /// Under `emit_se_coef` (task 116): NaN where nothing is estimated.
    #[serde(default, with = "online_core::humanfloat::vec_opt_vec_f64_or_tag")]
    pub se_coef: Vec<Option<Vec<f64>>>,
    /// Under `emit_robust_se` (task 221 (c)), each instance's, `None` on a
    /// row that was not a `coef` row and where none is switched on.
    #[serde(default, with = "online_core::humanfloat::vec_opt_vec_f64_or_tag")]
    pub se_coef_hc0: Vec<Option<Vec<f64>>>,
    #[serde(default, with = "online_core::humanfloat::vec_opt_vec_f64_or_tag")]
    pub se_coef_hac: Vec<Option<Vec<f64>>>,
}

/// Row `ri` of a buffer of `n_rows` rows: in every `ChunkOut` buffer the row
/// is the innermost index, whatever the outer product (`ChunkOut::at`, and
/// the model-major blocks of `metrics`, `conformal` and `resid_q`), so the
/// row's values are the elements `g * n_rows + ri`.
fn take_row<T: Copy>(dst: &mut Vec<T>, src: &[T], n_rows: usize, ri: usize) {
    dst.clear();
    dst.extend(src.iter().skip(ri).step_by(n_rows.max(1)));
}

/// Row `ri` of a `coef`-like buffer, one list per instance, cloned into the
/// lists `dst` already holds, so a list of the same length as the last is
/// copied into its memory rather than allocated (task 270, G-1).
fn take_list_row(dst: &mut Vec<Option<Vec<f64>>>, src: &[Vec<Option<Vec<f64>>>], ri: usize) {
    dst.truncate(src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        d.clone_from(&s[ri]);
    }
    let have = dst.len();
    dst.extend(src[have..].iter().map(|s| s[ri].clone()));
}

impl LastRow {
    /// Overwrite with row `ri` of `out`, reusing the vectors.
    pub fn take(&mut self, out: &ChunkOut, ri: usize) {
        let n = out.processed.len();
        debug_assert!(ri < n && out.processed[ri]);
        take_row(&mut self.pred, &out.pred, n, ri);
        take_row(&mut self.resid, &out.resid, n, ri);
        take_row(&mut self.sigma, &out.sigma, n, ri);
        take_row(&mut self.zscore, &out.zscore, n, ri);
        take_row(&mut self.autocorr, &out.autocorr, n, ri);
        take_row(&mut self.metrics, &out.metrics, n, ri);
        take_row(&mut self.conformal, &out.conformal, n, ri);
        take_row(&mut self.checks, &out.checks, n, ri);
        take_row(&mut self.health, &out.health, n, ri);
        take_row(&mut self.resid_q, &out.resid_q, n, ri);
        take_row(&mut self.drift, &out.drift, n, ri);
        take_row(&mut self.n_eff, &out.n_eff, n, ri);
        take_row(&mut self.scored_clock, &out.scored_clock, n, ri);
        take_row(&mut self.learned_clock, &out.learned_clock, n, ri);
        take_row(&mut self.lam_selected, &out.lam_selected, n, ri);
        take_list_row(&mut self.coef, &out.coef, ri);
        take_row(&mut self.settled, &out.settled, n, ri);
        take_row(&mut self.reason, &out.reason, n, ri);
        take_row(&mut self.inflation, &out.inflation, n, ri);
        take_list_row(&mut self.support_coef, &out.support_coef, ri);
        take_list_row(&mut self.se_coef, &out.se_coef, ri);
        take_list_row(&mut self.se_coef_hc0, &out.se_coef_hc0, ri);
        take_list_row(&mut self.se_coef_hac, &out.se_coef_hac, ri);
    }

    /// A one-row chunk at absolute row `row`, marked processed, for
    /// `assemble`. `Err` when the row is not the shape `spec` writes with
    /// `n_models` instances of `n_slots` slots -- a state file that does not
    /// belong to its spec.
    pub fn to_chunk(
        &self,
        spec: &Spec,
        n_models: usize,
        n_slots: usize,
        row: usize,
    ) -> Result<ChunkOut, String> {
        let mut out = ChunkOut::new(spec, n_models, n_slots, 1);
        let same = [
            out.pred.len() == self.pred.len(),
            out.resid.len() == self.resid.len(),
            out.sigma.len() == self.sigma.len(),
            out.zscore.len() == self.zscore.len(),
            out.autocorr.len() == self.autocorr.len(),
            out.metrics.len() == self.metrics.len(),
            out.conformal.len() == self.conformal.len(),
            out.checks.len() == self.checks.len(),
            out.health.len() == self.health.len(),
            out.resid_q.len() == self.resid_q.len(),
            out.drift.len() == self.drift.len(),
            out.n_eff.len() == self.n_eff.len(),
            out.lam_selected.len() == self.lam_selected.len(),
            out.coef.len() == self.coef.len(),
            out.settled.len() == self.settled.len(),
            out.reason.len() == self.reason.len(),
            out.inflation.len() == self.inflation.len(),
            out.support_coef.len() == self.support_coef.len(),
            out.se_coef.len() == self.se_coef.len(),
            out.se_coef_hc0.len() == self.se_coef_hc0.len(),
            out.se_coef_hac.len() == self.se_coef_hac.len(),
            out.scored_clock.len() == self.scored_clock.len(),
            out.learned_clock.len() == self.learned_clock.len(),
        ];
        if same.contains(&false) {
            return Err(format!(
                "saved last row of spec {:?} has the wrong shape for its spec",
                spec.name
            ));
        }
        out.rows.push(row);
        out.processed[0] = true;
        out.pred.clone_from(&self.pred);
        out.resid.clone_from(&self.resid);
        out.sigma.clone_from(&self.sigma);
        out.zscore.clone_from(&self.zscore);
        out.autocorr.clone_from(&self.autocorr);
        out.metrics.clone_from(&self.metrics);
        out.conformal.clone_from(&self.conformal);
        out.checks.clone_from(&self.checks);
        out.health.clone_from(&self.health);
        out.resid_q.clone_from(&self.resid_q);
        out.drift.clone_from(&self.drift);
        out.n_eff.clone_from(&self.n_eff);
        out.lam_selected.clone_from(&self.lam_selected);
        out.scored_clock.clone_from(&self.scored_clock);
        out.learned_clock.clone_from(&self.learned_clock);
        for (dst, src) in out.coef.iter_mut().zip(&self.coef) {
            dst[0].clone_from(src);
        }
        out.settled.clone_from(&self.settled);
        out.reason.clone_from(&self.reason);
        out.inflation.clone_from(&self.inflation);
        for (dst, src) in out.support_coef.iter_mut().zip(&self.support_coef) {
            dst[0].clone_from(src);
        }
        for (dst, src) in out.se_coef.iter_mut().zip(&self.se_coef) {
            dst[0].clone_from(src);
        }
        for (dst, src) in out.se_coef_hc0.iter_mut().zip(&self.se_coef_hc0) {
            dst[0].clone_from(src);
        }
        for (dst, src) in out.se_coef_hac.iter_mut().zip(&self.se_coef_hac) {
            dst[0].clone_from(src);
        }
        Ok(out)
    }
}
