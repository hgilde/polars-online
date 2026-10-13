//! A spec's output struct under construction: one [`Sink`] per spec and
//! chunk, which takes each stream's [`ChunkOut`] buffers in and gives the
//! struct array out.
//!
//! A stream writes its outputs into a run's buffers, a run being the rows
//! that fit the cache (`ChunkOut::run_rows`: about a thousand for a
//! 230-statistic `ew_cov`, a hundred thousand for a narrow model), and the
//! sink copies each run into the spec's output columns. Two ways in:
//!
//! - A spec that one stream serves in the chunk -- no group, or a single
//!   group -- hands each run to its sink as the run ends (`process`). The
//!   copy runs on another thread while the stream learns the next run, and
//!   the next run but one gets its buffers in the memory the copied run
//!   freed, so the call holds its output and two runs of buffers.
//! - A spec of many groups keeps every stream's runs until the streams are
//!   done, since their rows interleave in the output, and its sink takes
//!   them all at once, one field per job (`assemble`). It holds the runs and
//!   the output at once, at the end.
//!
//! The output is the same either way, to the bit: each field takes each run
//! by the same rule, whichever way the runs arrive. The first way is what
//! keeps a wide eager call near the size of its output: a 230-statistic
//! `ew_cov` over 200,000 rows returns 393 MiB, and peaked 797 MiB above
//! where it started when every run waited for the end (task 255 (a)).

use super::*;

/// One output field under construction, as its source dictates.
enum Acc {
    /// Every field that is one `f64` a row in a [`ChunkOut`] buffer.
    Flat(F64Column),
    /// A flat field whose column is allocated as its first run is copied
    /// in, so that its prefill is still in cache for the copy.
    Later,
    /// `withheld_reason`: a code a row, 0 for none.
    Codes(Vec<u8>),
    Drift(Vec<Option<bool>>),
    /// `coef` and the fields laid out like it: the rows that carry one, as
    /// `(row, values)`, taken from the run's buffers rather than copied.
    Lists(Vec<(usize, Vec<f64>)>),
    Clocks(Vec<Option<ClockValue>>),
    /// `emit_selected` and `emit_averaged`, which read every slot of a row
    /// and so are kept in the sink rather than per field.
    Derived,
}

/// The shape numbers every field's offset into a run is computed from.
#[derive(Clone, Copy)]
struct Shape {
    n_models: usize,
    nc: usize,
    m: usize,
    per_model: usize,
    n_levels: usize,
    k: usize,
}

impl Shape {
    /// Where field `src`'s `nr` values start in a run of `nr` rows, the
    /// buffer they are in, and whether every processed row is valid whatever
    /// its value (`n_eff`, which is reported as it is). `None` for a field
    /// that is not one `f64` a row.
    fn flat<'c>(&self, src: &Source, ch: &'c ChunkOut) -> Option<(&'c [f64], bool)> {
        let nr = ch.rows.len();
        let block = ch.n_slots * nr;
        // Where field `i`'s run starts in a per-instance buffer.
        let at = |i: usize| ChunkOut::at(ch.n_slots, nr, i / self.per_model, i % self.per_model, 0);
        let (buf, start, all): (&[f64], usize, bool) = match *src {
            Source::Pred(i)
            | Source::Stat(i)
            | Source::Id(i)
            | Source::Flag(i)
            | Source::Label(i) => (&ch.pred, at(i), false),
            Source::Resid(i) => (&ch.resid, at(i), false),
            Source::Sigma(i) => (&ch.sigma, at(i), false),
            Source::ResidZ(i) => (&ch.zscore, at(i), false),
            Source::Autocorr(i) => (&ch.autocorr, at(i), false),
            // Model-major: instance mi owns 3 contiguous blocks.
            Source::Metric(k, i) => {
                let (mi, slot) = (i / self.per_model, i % self.per_model);
                (&ch.metrics, mi * 3 * block + k * block + slot * nr, false)
            }
            Source::Conformal(k, i) => {
                let (mi, slot) = (i / self.per_model, i % self.per_model);
                (&ch.conformal, mi * 3 * block + k * block + slot * nr, false)
            }
            Source::Health(which, mi, fi) => (
                &ch.health,
                (mi * 2 * self.k + which * self.k + fi) * nr,
                false,
            ),
            Source::Check(k, i) => {
                let (mi, slot) = (i / self.per_model, i % self.per_model);
                (
                    &ch.checks,
                    mi * ch.n_checks * block + k * block + slot * nr,
                    false,
                )
            }
            Source::Quantile(i) => {
                // `(li * n_models + mi) * m * nc + slot`, as `output_index`
                // numbers the quantile fields.
                let per = self.m * self.nc;
                let slot = i % per;
                let (li, mi) = ((i / per) / self.n_models, (i / per) % self.n_models);
                (
                    &ch.resid_q,
                    mi * self.n_levels * block + li * block + slot * nr,
                    false,
                )
            }
            Source::NEff(mi) => (&ch.n_eff, mi * nr, true),
            Source::Settled(mi) => (&ch.settled, mi * nr, false),
            Source::Inflation(i) => (&ch.inflation, at(i), false),
            Source::LamSelected(i) => (&ch.lam_selected, i * nr, false),
            _ => return None,
        };
        Some((&buf[start..][..nr], all))
    }
}

/// One value per processed row of `ch` into `col`: `vals` is the field's
/// values for the run in row order -- every `ChunkOut` buffer is
/// slot-major, so that is one slice. Non-finite values stay null unless
/// `all`.
///
/// A run whose rows are one unbroken range -- a single group, or a group
/// that arrives in blocks -- is copied as a select over three slices with no
/// index indirection, its validity packed on the way (`F64Column::run`); the
/// general path scatters through `rows`. Same values either way: a
/// processed, finite value lands, anything else leaves the NaN prefill
/// (docs/PERFORMANCE.md §13).
fn scatter(col: &mut F64Column, ch: &ChunkOut, all: bool, vals: &[f64]) {
    debug_assert_eq!(vals.len(), ch.rows.len());
    match ch.contiguous() {
        Some(base) if !all => col.run(base, vals, &ch.processed),
        _ => {
            for (ri, &i) in ch.rows.iter().enumerate() {
                if !ch.processed[ri] {
                    continue;
                }
                if all {
                    col.set(i, vals[ri]);
                } else {
                    col.set_if_finite(i, vals[ri]);
                }
            }
        }
    }
}

/// The `coef`-like buffer a list field reads, one per instance.
fn lists_of<'c>(src: &Source, ch: &'c mut ChunkOut) -> &'c mut [Vec<Option<Vec<f64>>>] {
    match src {
        Source::Coef(_) => &mut ch.coef,
        Source::SupportCoef(_) => &mut ch.support_coef,
        Source::SeCoef(_) => &mut ch.se_coef,
        Source::SeCoefHc0(_) => &mut ch.se_coef_hc0,
        Source::SeCoefHac(_) => &mut ch.se_coef_hac,
        _ => unreachable!("only a list field reads a coef-like buffer"),
    }
}

/// One spec's output struct under construction, over a chunk of `n` rows.
pub(super) struct Sink<'a> {
    spec: &'a Spec,
    d: &'a SpecDerived,
    n: usize,
    clock_dtype: Option<&'a crate::stream::ClockDtype>,
    shape: Shape,
    fields: Vec<Acc>,
    sel_pred: Vec<Vec<Option<f64>>>,
    /// The selected slot's label, as its index in `slot_labels`.
    sel_name: Vec<Vec<Option<usize>>>,
    avg_pred: Vec<Vec<Option<f64>>>,
}

impl<'a> Sink<'a> {
    /// Every field null, over `n` rows. With `prefill`, the output columns
    /// are allocated and filled here, in parallel from `PAR_MIN_ROWS` rows,
    /// for a sink that takes its runs one at a time; without, each as its
    /// field is first written, for one that takes them all at once.
    pub(super) fn new(
        spec: &'a Spec,
        d: &'a SpecDerived,
        n: usize,
        clock_dtype: Option<&'a crate::stream::ClockDtype>,
        prefill: bool,
    ) -> Self {
        let shape = Shape {
            n_models: d.n_models,
            nc: d.nc,
            m: d.m,
            per_model: d.per_model,
            n_levels: spec.resid_quantiles.as_ref().map_or(0, Vec::len),
            k: spec.k(),
        };
        let fields =
            map_maybe_par::<_, _, Vec<Acc>>(&d.schema, n >= PAR_MIN_ROWS, |f| match f.src {
                Source::Reason(_) => Acc::Codes(vec![0u8; n]),
                Source::Drift(_) => Acc::Drift(vec![None; n]),
                Source::Coef(_)
                | Source::SupportCoef(_)
                | Source::SeCoef(_)
                | Source::SeCoefHc0(_)
                | Source::SeCoefHac(_) => Acc::Lists(Vec::new()),
                Source::ScoredClock | Source::LearnedClock => Acc::Clocks(vec![None; n]),
                Source::SelPred(_) | Source::SelName(_) | Source::AvgPred(_) => Acc::Derived,
                Source::Unset => unreachable!("every field is given a source in output_index"),
                _ if prefill => Acc::Flat(F64Column::new(n)),
                _ => Acc::Later,
            });
        let m = d.m;
        Self {
            spec,
            d,
            n,
            clock_dtype,
            shape,
            fields,
            sel_pred: vec![vec![None; n]; if spec.emit_selected { m } else { 0 }],
            sel_name: vec![vec![None; n]; if spec.emit_selected { m } else { 0 }],
            avg_pred: vec![vec![None; n]; if spec.emit_averaged { m } else { 0 }],
        }
    }

    /// Runs in, every field: each run's `coef`-like buffers are taken, the
    /// rest is read. One job per field from `PAR_MIN_ROWS` rows between them
    /// (docs/PERFORMANCE.md P10): one spec with a grid of instances used to
    /// assemble on a single thread.
    pub(super) fn absorb(&mut self, chunks: &mut [ChunkOut]) {
        #[cfg(test)]
        tests::absorbed(&self.spec.name);
        self.prepare(chunks);
        let (shape, n) = (self.shape, self.n);
        let chunks: &[ChunkOut] = chunks;
        let job = |(acc, f): (&mut Acc, &FieldMeta)| {
            for ch in chunks {
                absorb_field(acc, &f.src, ch, shape, n);
            }
        };
        let rows: usize = chunks.iter().map(|ch| ch.rows.len()).sum();
        if rows >= PAR_MIN_ROWS {
            self.fields
                .par_iter_mut()
                .zip(self.d.schema.par_iter())
                .for_each(job);
        } else {
            self.fields
                .iter_mut()
                .zip(self.d.schema.iter())
                .for_each(job);
        }
    }

    /// `emit_selected` and `emit_averaged` over one run's rows.
    fn absorb_derived(&mut self, ch: &ChunkOut) {
        let (n_models, nc, m) = (self.d.n_models, self.d.nc, self.d.m);
        let nr = ch.rows.len();
        if self.spec.emit_averaged {
            // Exponentially weighted forecaster: weight each slot by
            // exp(-eta * (sigma² / sigma²_best - 1)), normalized -- each slot's EW
            // squared error as a ratio to the best slot's, so `eta` means the same
            // in any target's units. The difference it replaced made `eta = 1` an
            // equal-weight mean for a return and the argmin for a price (review
            // 2026-09-12, S23). `sigma` is that error's square root, already
            // tracked for E12, so this costs one pass.
            let eta = self.spec.average_eta_or_default();
            for (ri, &i) in ch.rows.iter().enumerate() {
                if !ch.processed[ri] {
                    continue;
                }
                let sig =
                    |mi: usize, slot: usize| ch.sigma[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                let prd =
                    |mi: usize, slot: usize| ch.pred[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                for (t_i, avg_t) in self.avg_pred.iter_mut().enumerate() {
                    // The best loss is the denominator: the best slot weighs
                    // exp(0) = 1, and no weight overflows. At a best of 0 the
                    // ratio's limit is the argmin -- every slot at 0, no other.
                    let mut best = f64::INFINITY;
                    for mi in 0..n_models {
                        for c_i in 0..nc {
                            let s = sig(mi, t_i * nc + c_i);
                            if s.is_finite() && prd(mi, t_i * nc + c_i).is_finite() {
                                best = best.min(s * s);
                            }
                        }
                    }
                    if !best.is_finite() {
                        continue;
                    }
                    let (mut num, mut den) = (0.0, 0.0);
                    for mi in 0..n_models {
                        for c_i in 0..nc {
                            let slot = t_i * nc + c_i;
                            let (s, p) = (sig(mi, slot), prd(mi, slot));
                            if s.is_finite() && p.is_finite() {
                                // A slot past the best weighs `exp(-eta *
                                // excess)`, and the best weighs 1 at any
                                // `eta`: at `inf`, `-inf * 0` made it NaN
                                // (review 2026-09-12, S27).
                                let excess = if best > 0.0 {
                                    s * s / best - 1.0
                                } else if s * s == 0.0 {
                                    0.0
                                } else {
                                    f64::INFINITY
                                };
                                let wgt = if excess > 0.0 {
                                    (-eta * excess).exp()
                                } else {
                                    1.0
                                };
                                num += wgt * p;
                                den += wgt;
                            }
                        }
                    }
                    if den > 0.0 {
                        avg_t[i] = Some(num / den);
                    }
                }
            }
        }
        if self.spec.emit_selected {
            for (ri, &i) in ch.rows.iter().enumerate() {
                if !ch.processed[ri] {
                    continue;
                }
                let sig =
                    |mi: usize, slot: usize| ch.sigma[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                let prd =
                    |mi: usize, slot: usize| ch.pred[ChunkOut::at(ch.n_slots, nr, mi, slot, ri)];
                for t_i in 0..m {
                    let mut best: Option<(f64, usize, usize)> = None;
                    for mi in 0..n_models {
                        for c_i in 0..nc {
                            let s = sig(mi, t_i * nc + c_i);
                            if s.is_finite() && best.is_none_or(|(b, _, _)| s < b) {
                                best = Some((s, mi, c_i));
                            }
                        }
                    }
                    if let Some((_, mi, c_i)) = best {
                        let p = prd(mi, t_i * nc + c_i);
                        if p.is_finite() {
                            self.sel_pred[t_i][i] = Some(p);
                            self.sel_name[t_i][i] = Some(mi * nc + c_i);
                        }
                    }
                }
            }
        }
    }

    /// The struct array: every field finished, in schema order, one job
    /// per field from `PAR_MIN_ROWS` rows.
    pub(super) fn finish(mut self) -> PolarsResult<StructArray> {
        let fields = std::mem::take(&mut self.fields);
        let schema = &self.d.schema;
        let job = |(acc, f): (Acc, &FieldMeta)| self.finish_field(acc, f);
        let arrays: Vec<Box<dyn ArrowArray>> = if self.n >= PAR_MIN_ROWS {
            fields
                .into_par_iter()
                .zip(schema.par_iter())
                .map(job)
                .collect()
        } else {
            fields.into_iter().zip(schema.iter()).map(job).collect()
        };
        Ok(self.struct_of(arrays))
    }

    /// The `coef`-like lists and the selected and averaged fields of every
    /// run: what [`Self::absorb`] does on the calling thread.
    fn prepare(&mut self, chunks: &mut [ChunkOut]) {
        for ch in chunks.iter() {
            self.absorb_derived(ch);
        }
        // The lists are taken first, on this thread: each takes from its own
        // buffer, but a job holding `&mut` to every run would hold them all.
        for (acc, f) in self.fields.iter_mut().zip(&self.d.schema) {
            if let Acc::Lists(list) = acc {
                for ch in chunks.iter_mut() {
                    take_lists(list, &f.src, ch);
                }
            }
        }
    }

    /// One field finished as its Arrow array.
    fn finish_field(&self, acc: Acc, f: &FieldMeta) -> Box<dyn ArrowArray> {
        let (spec, n, labels) = (self.spec, self.n, &self.d.slot_labels);
        // A field no run reached is all null.
        let acc = match acc {
            Acc::Later => Acc::Flat(F64Column::new(n)),
            acc => acc,
        };
        match (acc, &f.src) {
            (Acc::Flat(col), Source::Id(_)) => col.finish_i64_array_boxed(),
            (Acc::Flat(col), Source::Flag(_)) => col.finish_bool_array_boxed(),
            (Acc::Flat(col), Source::Label(_)) => {
                let classes: &[String] = match &spec.model {
                    ModelKind::EwClass { classes, .. } => classes,
                    _ => unreachable!("only ew_class emits a label"),
                };
                col.finish_label_array_boxed(classes)
            }
            (Acc::Flat(col), _) => col.finish_array_boxed(),
            (Acc::Codes(codes), _) => Box::new(crate::column::code_array(
                &codes,
                &crate::stream::WITHHELD_REASONS,
            )),
            (Acc::Drift(drift), _) => opt_bool_array(&drift),
            (Acc::Lists(list), _) => {
                // Finite-or-null inside the list too: an `ew_class` class
                // no row has carried yet has NaN means, and a null says so.
                let mut rows: Vec<Option<&Vec<f64>>> = vec![None; n];
                for (row, c) in &list {
                    rows[*row] = Some(c);
                }
                coef_list_array(&rows)
            }
            (Acc::Clocks(values), _) => {
                let numeric = crate::stream::ClockDtype::Numeric;
                self.clock_dtype.unwrap_or(&numeric).array(&values)
            }
            (Acc::Derived, Source::SelPred(i)) => opt_f64_array(&self.sel_pred[*i]),
            (Acc::Derived, Source::SelName(i)) => Box::new(crate::column::dict_array(
                self.sel_name[*i]
                    .iter()
                    .map(|v| v.and_then(|s| labels.iter().position(|x| *x == labels[s])))
                    .map(|k| k.map(|k| k as u32)),
                &labels.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            (Acc::Derived, Source::AvgPred(i)) => opt_f64_array(&self.avg_pred[*i]),
            (Acc::Derived | Acc::Later, _) => {
                unreachable!("only a selected or averaged field is derived")
            }
        }
    }

    /// The finished fields as the spec's struct array.
    ///
    /// The schema is written once, in `output_index`, which also records
    /// where each field's values live. Emission is a walk over it: no second
    /// copy of the ordering, no `format!` here at all, and no way for the
    /// declared schema and the realized struct to disagree
    /// (docs/SIMPLIFICATION.md S1 -- that divergence is exactly what defect
    /// E23 was). The output path is Arrow end to end: the struct array is
    /// what the bank returns, and giving it the spec's name is the adapter's
    /// job, not this function's (docs/PLAN.md task 86).
    fn struct_of(&self, arrays: Vec<Box<dyn ArrowArray>>) -> StructArray {
        let arrow_fields: Vec<ArrowField> = self
            .d
            .schema
            .iter()
            .zip(&arrays)
            .map(|(f, a)| {
                let field = ArrowField::new(f.field.as_str().into(), a.dtype().clone(), true);
                // The reason, a class and a selected slot are each an `Enum`
                // over their names: the metadata is what tells polars so
                // (`column::enum_metadata`).
                if matches!(f.src, Source::Reason(_)) {
                    field.with_metadata(crate::column::enum_metadata(
                        &crate::stream::WITHHELD_REASONS,
                    ))
                } else if let Some(names) = &f.categories {
                    let names: Vec<&str> = names.iter().map(String::as_str).collect();
                    field.with_metadata(crate::column::enum_metadata(&names))
                } else {
                    field
                }
            })
            .collect();
        StructArray::new(ArrowDataType::Struct(arrow_fields), self.n, arrays, None)
    }
}

/// One run into one field that is not a list.
fn absorb_field(acc: &mut Acc, src: &Source, ch: &ChunkOut, shape: Shape, n: usize) {
    if matches!(acc, Acc::Later) {
        *acc = Acc::Flat(F64Column::new(n));
    }
    match acc {
        Acc::Flat(col) => {
            let (vals, all) = shape
                .flat(src, ch)
                .expect("a flat field reads one f64 a row");
            scatter(col, ch, all, vals);
        }
        Acc::Codes(codes) => {
            let Source::Reason(mi) = *src else {
                unreachable!("only withheld_reason is a code")
            };
            let nr = ch.rows.len();
            for (ri, &row) in ch.rows.iter().enumerate() {
                if ch.processed[ri] {
                    codes[row] = ch.reason[mi * nr + ri];
                }
            }
        }
        Acc::Drift(drift) => {
            let Source::Drift(i) = *src else {
                unreachable!("only drift is a flag a slot")
            };
            let nr = ch.rows.len();
            let at = ChunkOut::at(ch.n_slots, nr, i / shape.per_model, i % shape.per_model, 0);
            for (ri, &row) in ch.rows.iter().enumerate() {
                if ch.processed[ri] {
                    drift[row] = Some(ch.drift[at + ri]);
                }
            }
        }
        Acc::Clocks(values) => {
            let v = match src {
                Source::ScoredClock => &ch.scored_clock,
                _ => &ch.learned_clock,
            };
            for (ri, &row) in ch.rows.iter().enumerate() {
                if ch.processed[ri] {
                    values[row] = v[ri];
                }
            }
        }
        // Taken from the run by `take_lists`, and kept in the sink.
        Acc::Lists(_) | Acc::Derived => {}
        Acc::Later => unreachable!("allocated above"),
    }
}

/// The rows of one run that carry a `coef`-like list, moved out of the run.
fn take_lists(list: &mut Vec<(usize, Vec<f64>)>, src: &Source, ch: &mut ChunkOut) {
    let mi = match *src {
        Source::Coef(mi)
        | Source::SupportCoef(mi)
        | Source::SeCoef(mi)
        | Source::SeCoefHc0(mi)
        | Source::SeCoefHac(mi) => mi,
        _ => unreachable!("only a list field takes lists"),
    };
    let rows = std::mem::take(&mut ch.rows);
    let processed = std::mem::take(&mut ch.processed);
    for (ri, &row) in rows.iter().enumerate() {
        if processed[ri]
            && let Some(c) = lists_of(src, ch)[mi][ri].take()
        {
            list.push((row, c));
        }
    }
    ch.rows = rows;
    ch.processed = processed;
}

/// One spec's struct array from every run of a chunk at once: one job per
/// field from `PAR_MIN_ROWS` rows, which allocates the field's column,
/// copies every run into it and finishes it while it is still in cache.
pub(super) fn assemble(
    spec: &Spec,
    d: &SpecDerived,
    n: usize,
    chunks: &mut [ChunkOut],
    clock_dtype: Option<&crate::stream::ClockDtype>,
) -> PolarsResult<StructArray> {
    let mut sink = Sink::new(spec, d, n, clock_dtype, false);
    sink.prepare(chunks);
    let fields = std::mem::take(&mut sink.fields);
    let (sink, chunks): (&Sink<'_>, &[ChunkOut]) = (&sink, chunks);
    let job = |(mut acc, f): (Acc, &FieldMeta)| {
        for ch in chunks {
            absorb_field(&mut acc, &f.src, ch, sink.shape, n);
        }
        sink.finish_field(acc, f)
    };
    let arrays: Vec<Box<dyn ArrowArray>> = if n >= PAR_MIN_ROWS {
        fields
            .into_par_iter()
            .zip(d.schema.par_iter())
            .map(job)
            .collect()
    } else {
        fields.into_iter().zip(d.schema.iter()).map(job).collect()
    };
    Ok(sink.struct_of(arrays))
}

#[cfg(test)]
#[path = "assembly_tests.rs"]
pub(super) mod tests;
