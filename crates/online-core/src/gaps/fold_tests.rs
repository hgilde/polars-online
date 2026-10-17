//! The targets folded into a blocked Gram's merge (docs/PLAN.md task 257):
//! each target's centred cross-moments and feature means, held to an
//! exactly rounded oracle of their definition at every block size, thread
//! count and `target_gaps`, with gaps, zero weights, decays that halve or
//! take the whole history, and a reset of one target's own; to the bit at
//! every thread count and whatever other targets share the block; and a
//! state from before the fold, whose targets were current through the held
//! rows, going on from it. A target far above its spread, where the block's
//! own mean and its residue must both be carried; what a blocked fit still
//! takes row by row, to the unblocked fit's bits; a row the block decays to
//! nothing; and a damaged held block, dropped.

use super::*;
use crate::oracle;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// One row as [`Acc::learn`] takes it: the feature row `z`, the targets,
/// the decay and the weight.
#[derive(Clone)]
struct Row {
    z: Vec<f64>,
    y: Vec<Option<f64>>,
    lam: f64,
    w: f64,
}

/// What a stream holds besides gaps and weights.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// No decay; every target with gaps of its own; zero weights, the
    /// first row's among them.
    Plain,
    /// Decays of `1/2` and one of 0 (a gap that takes the whole history),
    /// on top of [`Kind::Plain`].
    Decays,
    /// Target 0 present only on rows of weight `1e-310` at first, then
    /// absent, then present on a row of weight 1: its own history is in the
    /// subnormal range of that row and forgotten there ([`decay_into`]),
    /// where the Gram's, over every row, is not.
    Reset,
}

/// `n` rows of `k` features at `level`, the first a constant 1, `t`
/// targets. The decays are 1,
/// `1/2` and 0 and the weights dyadic, so every weight and every discount
/// the oracle forms is exact.
fn stream(kind: Kind, k: usize, t: usize, n: usize, level: f64, seed: u64) -> Vec<Row> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            // Feature 0 is a constant, as `ewridge`'s intercept slot is.
            let z: Vec<f64> = (0..k)
                .map(|f| if f == 0 { 1.0 } else { level + lcg(&mut s) })
                .collect();
            let fit = z
                .iter()
                .enumerate()
                .map(|(f, v)| (v - level) / (f + 1) as f64)
                .sum::<f64>();
            let mut w = match i % 7 {
                _ if i == 0 => 0.0,
                3 => 0.0,
                1 => 2.5,
                5 => 0.5,
                _ => 1.0,
            };
            let mut y: Vec<Option<f64>> = (0..t)
                .map(|j| {
                    let v = 0.5 * level + fit * (1.0 + j as f64) + 0.2 * lcg(&mut s);
                    (i % (5 + 2 * j) != 2).then_some(v)
                })
                .collect();
            let mut lam = 1.0;
            if matches!(kind, Kind::Decays) {
                if i % 13 == 6 {
                    lam = 0.5;
                }
                if i == n / 2 {
                    lam = 0.0;
                }
            }
            if matches!(kind, Kind::Reset) {
                if i < 6 {
                    w = 1e-310;
                } else if i < n / 2 {
                    y[0] = None;
                } else if i == n / 2 {
                    y[0] = Some(0.5 * level + fit);
                    w = 1.0;
                }
            }
            Row { z, y, lam, w }
        })
        .collect()
}

/// Per target, its weight, its feature means and its centred
/// cross-moments over `rows`, from their definition: each row it learns
/// from weighs its weight times every decay after it -- its own, which
/// [`decay_into`] sets to 0 where it forgets the target's history -- and
/// the sums are exactly rounded ([`oracle::sum`], [`oracle::dot`],
/// [`oracle::dot3`]). With the bounds a sum over the rows in any order is
/// held to: the magnitudes `Σ v|z − m|/W` and `Σ v|z − m||y − ȳ|/W`, and
/// the target's spread `Σ v|y − ȳ|/W`, which a mean's rounding at its
/// level multiplies.
struct Defined {
    w: f64,
    m: Vec<f64>,
    c: Vec<f64>,
    m_size: Vec<f64>,
    c_size: Vec<f64>,
    y_size: f64,
}

fn defined(rows: &[Row], j: usize, drop: Option<usize>) -> Option<Defined> {
    let k = rows[0].z.len();
    // (v, row) for each row the target learns from, discounted as it goes.
    let mut kept: Vec<(f64, usize)> = Vec::new();
    let mut wj = 0.0f64;
    for (r, row) in rows.iter().enumerate() {
        let present = row.y[j].is_some() && row.lam * wj + row.w > 0.0;
        let lam = if present {
            decay_into(row.lam, wj, row.w)
        } else {
            row.lam
        };
        for (v, _) in kept.iter_mut() {
            *v *= lam;
        }
        wj = lam * wj + if present { row.w } else { 0.0 };
        if present && row.w > 0.0 && Some(r) != drop {
            kept.push((row.w, r));
        }
    }
    kept.retain(|&(v, _)| v > 0.0);
    if kept.is_empty() {
        return None;
    }
    let w = oracle::sum(kept.iter().map(|&(v, _)| v));
    let y = |r: usize| rows[r].y[j].unwrap();
    let ybar = oracle::dot(kept.iter().map(|&(v, r)| (v, y(r)))) / w;
    let mut out = Defined {
        w,
        m: vec![0.0; k],
        c: vec![0.0; k],
        m_size: vec![0.0; k],
        c_size: vec![0.0; k],
        y_size: kept
            .iter()
            .map(|&(v, r)| v * (y(r) - ybar).abs())
            .sum::<f64>()
            / w,
    };
    for f in 0..k {
        let z = |r: usize| rows[r].z[f];
        let m = oracle::dot(kept.iter().map(|&(v, r)| (v, z(r)))) / w;
        // Σ v (z − m)(y − ȳ) = Σ v z y − m Σ v y − ȳ Σ v z + m ȳ Σ v, as
        // one exactly rounded sum of triple products.
        let triples = kept.iter().flat_map(|&(v, r)| {
            [
                (v, z(r), y(r)),
                (-m, v, y(r)),
                (-ybar, v, z(r)),
                (m, ybar, v),
            ]
        });
        out.m[f] = m;
        out.c[f] = oracle::dot3(triples) / w;
        out.m_size[f] = kept.iter().map(|&(v, r)| v * (z(r) - m).abs()).sum::<f64>() / w;
        out.c_size[f] = kept
            .iter()
            .map(|&(v, r)| v * ((z(r) - m) * (y(r) - ybar)).abs())
            .sum::<f64>()
            / w;
    }
    Some(out)
}

/// How far past its bound each entry of target `j`'s moments in `acc` is
/// from the definition (`≤ 1` is within): `γ_{8R+32}` of the magnitudes,
/// `R` the rows, a few roundings of the value itself, and the data's own
/// resolution at the feature's level: a block's merge takes its share of
/// the weight as `U/W'`, `W'` the weight the rows carried, which is the
/// block's `U` to rounding, so a mean moves a rounding of its level short
/// of the block's (as the Gram's own does), and the cross-moment's between
/// term carries it times the target's spread: `4u·|m|·Σ v|y − ȳ|/W`.
fn worst(acc: &Acc, rows: &[Row], j: usize, drop: Option<usize>) -> f64 {
    const U: f64 = f64::EPSILON / 2.0;
    let Some(d) = defined(rows, j, drop) else {
        return 0.0;
    };
    let mu = (8 * rows.len() + 32) as f64 * U;
    let gamma = mu / (1.0 - mu);
    let mut worst = 0.0f64;
    assert!(
        drop.is_some() || (acc.wj[j] - d.w).abs() <= gamma * d.w,
        "the weight: {} against {} (drop {drop:?})",
        acc.wj[j],
        d.w
    );
    let k = d.m.len();
    for f in 0..k {
        let m = acc.cross.mj[j][f] + acc.cross.mj_lo[j][f];
        let m_bound = gamma * d.m_size[f] + 2.0 * U * d.m[f].abs() + 1e-300;
        let c_bound = gamma * d.c_size[f]
            + 4.0 * U * d.c[f].abs()
            + 4.0 * U * d.m[f].abs() * d.y_size
            + 1e-300;
        // A NaN is past every bound: `f64::max` would drop it.
        for r in [
            (m - d.m[f]).abs() / m_bound,
            (acc.cross.c[j][f] - d.c[f]).abs() / c_bound,
        ] {
            worst = if r.is_nan() {
                f64::INFINITY
            } else {
                worst.max(r)
            };
        }
    }
    worst
}

/// The accumulators after `rows`, blocked by `block` (0: row by row), on
/// `threads` threads, every held block merged into a copy.
fn run(rows: &[Row], block: usize, threads: usize, gaps: TargetGaps) -> Acc {
    let (k, t) = (rows[0].z.len(), rows[0].y.len());
    let mut acc = Acc::new(t, k, block, false);
    acc.set_threads(threads);
    for r in rows {
        acc.learn(&r.z, &r.y, r.lam, r.w, gaps);
    }
    acc.flushed().into_owned()
}

/// A named change to a `T`: a damage, or a part changed alone.
pub(super) type Damage<'a, T> = (&'a str, &'a dyn Fn(&mut T));

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("a test pool")
}

fn bits(acc: &Acc) -> Vec<u64> {
    acc.cross
        .c
        .iter()
        .chain(&acc.cross.mj)
        .chain(&acc.cross.mj_lo)
        .flatten()
        .chain(&acc.cross.my)
        .chain(&acc.wj)
        .map(|v| v.to_bits())
        .collect()
}

const GAPS: [TargetGaps; 2] = [TargetGaps::OwnRows, TargetGaps::Pairwise];

/// Every entry of every target's cross-moments and feature means within
/// its bound of the definition: blocks of 1 to 257 rows and row by row,
/// one to eight threads, both `target_gaps`, features at 0 and at `1e3`,
/// and each kind of stream. And the bound is not vacuous: the definition
/// with one of the target's rows left out lies outside it.
#[test]
fn the_folded_moments_are_the_definitions_within_their_bound() {
    let pool = pool(8);
    for kind in [Kind::Plain, Kind::Decays, Kind::Reset] {
        for level in [0.0, 1e3] {
            let rows = stream(kind, 5, 3, 140, level, 7 + level as u64);
            for gaps in GAPS {
                for block in [0usize, 1, 2, 3, 16, 64, 257] {
                    for threads in [1usize, 3, 8] {
                        let acc = pool.install(|| run(&rows, block, threads, gaps));
                        for j in 0..3 {
                            let what = format!(
                                "{kind:?}, level {level}, {gaps:?}, block {block}, \
                                 {threads} threads, target {j}"
                            );
                            let w = worst(&acc, &rows, j, None);
                            assert!(w <= 1.0, "{what}: {w} of the bound");
                        }
                    }
                }
                // One row of target 1 left out of the definition: outside.
                let acc = run(&rows, 16, 1, gaps);
                let r = rows
                    .iter()
                    .rposition(|r| r.y[1].is_some() && r.w > 0.0)
                    .unwrap();
                let w = worst(&acc, &rows, 1, Some(r));
                assert!(
                    w > 1.0,
                    "{kind:?}, {gaps:?}: a dropped row is within the bound ({w})"
                );
                // Nor is a cross-moment scaled by `1 + 1e-12`, where the
                // features' level leaves the resolution to see it.
                if level == 0.0 {
                    let mut off = acc.clone();
                    off.cross.c[1].iter_mut().for_each(|c| *c *= 1.0 + 1e-12);
                    let w = worst(&off, &rows, 1, None);
                    assert!(
                        w > 1.0,
                        "{kind:?}, {gaps:?}: a scale of 1e-12 is within ({w})"
                    );
                }
            }
        }
    }
}

/// The fold sees the stream it is for: blocks hold targets' values, some
/// with gaps inside the block (their own feature means), and under
/// `Kind::Reset` target 0's own history is forgotten while the Gram's is
/// not.
#[test]
fn the_cases_reach_the_paths_they_are_for() {
    let rows = stream(Kind::Reset, 3, 3, 140, 0.0, 9);
    let mut acc = Acc::new(3, 3, 16, false);
    let mut held = 0;
    let mut reset = false;
    for (i, r) in rows.iter().enumerate() {
        acc.learn(&r.z, &r.y, r.lam, r.w, TargetGaps::Pairwise);
        held = held.max(acc.held.y.iter().map(Vec::len).max().unwrap_or(0));
        if i == rows.len() / 2 {
            // The row that forgets target 0's history starts its held
            // values again, with no history behind them.
            reset = acc.held.y[0].len() == 1 && acc.held.hist[0] == 0.0;
            assert!(acc.grams.grams[0].n_eff() > 1.0);
        }
    }
    assert!(reset, "target 0's history was not forgotten on its own");
    assert_eq!(held, 15, "a block of 16 holds 15 rows between merges");
    assert!(
        acc.held.y[1].iter().any(|v| v.is_nan()),
        "a gap inside a block"
    );
}

/// The moments are the same bits at every thread count, wide enough that
/// the fold's sums are cut into bands of columns, and the same whichever
/// other targets share the block: a target's sums are its own, row by row.
#[test]
fn every_thread_count_and_every_neighbour_gives_the_same_bits() {
    let pool = pool(8);
    let rows = stream(Kind::Decays, 600, 2, 40, 1e3, 11);
    for gaps in GAPS {
        let one = pool.install(|| run(&rows, 16, 1, gaps));
        for threads in [2usize, 3, 8] {
            let many = pool.install(|| run(&rows, 16, threads, gaps));
            assert!(bits(&many) == bits(&one), "{gaps:?}, {threads} threads");
        }
    }
    // Target 0 alone, and beside target 1, under `pairwise`: one Gram over
    // every row either way.
    let alone: Vec<Row> = rows
        .iter()
        .map(|r| Row {
            y: r.y[..1].to_vec(),
            ..r.clone()
        })
        .collect();
    let a = run(&alone, 16, 1, TargetGaps::Pairwise);
    let b = run(&rows, 16, 1, TargetGaps::Pairwise);
    let of0 = |acc: &Acc| -> Vec<u64> {
        acc.cross.c[0]
            .iter()
            .chain(&acc.cross.mj[0])
            .map(|v| v.to_bits())
            .collect()
    };
    assert!(of0(&a) == of0(&b), "a neighbour moved target 0's bits");
}

/// Under `own_rows` a target's rows are its Gram's, so its feature means,
/// merged by the Gram's own steps, are its Gram's to the bit.
#[test]
fn under_own_rows_a_targets_means_are_its_grams() {
    let rows = stream(Kind::Decays, 4, 3, 140, 1e3, 13);
    for block in [1usize, 5, 16] {
        let acc = run(&rows, block, 1, TargetGaps::OwnRows);
        assert!(acc.grams.grams.len() > 1, "the targets part");
        for j in 0..3 {
            let g = &acc.grams.grams[acc.grams.of[j]];
            assert_eq!(
                acc.cross.mj[j].as_slice(),
                g.means(),
                "block {block}, target {j}"
            );
            assert_eq!(
                acc.cross.mj_lo[j].as_slice(),
                g.means_lo(),
                "block {block}, target {j}"
            );
        }
    }
}

/// A state written before the fold holds a blocked Gram's rows with its
/// targets' moments already through them, and no held values: the rows
/// after it are folded alone, a suffix of the block, and the moments stay
/// within their bound of the definition.
#[test]
fn a_state_from_before_the_fold_goes_on_from_its_held_rows() {
    let rows = stream(Kind::Decays, 4, 2, 120, 1e3, 17);
    for gaps in GAPS {
        for before in [5usize, 21, 60] {
            let mut acc = Acc::new(2, 4, 16, false);
            for r in &rows[..before] {
                acc.learn_per_row(&r.z, &r.y, r.lam, r.w, gaps);
            }
            assert!(acc.grams.grams.iter().any(EwCov::has_pending));
            assert!(acc.held.is_empty());
            for r in &rows[before..] {
                acc.learn(&r.z, &r.y, r.lam, r.w, gaps);
            }
            let acc = acc.flushed();
            for j in 0..2 {
                let w = worst(&acc, &rows, j, None);
                assert!(w <= 1.0, "{gaps:?}, from row {before}, target {j}: {w}");
            }
        }
    }
}

/// Mid-block, the held values are written and read back: a round trip of
/// the accumulators at every offset goes on to the same bits as the ones
/// that never stopped, and a state between blocks writes no field for them.
#[test]
fn a_state_saved_mid_block_goes_on_to_the_bit() {
    let rows = stream(Kind::Decays, 3, 2, 90, 1e3, 19);
    for gaps in GAPS {
        let whole = run(&rows, 8, 1, gaps);
        for cut in [1usize, 3, 8, 13, 45, 89] {
            let mut acc = Acc::new(2, 3, 8, false);
            for r in &rows[..cut] {
                acc.learn(&r.z, &r.y, r.lam, r.w, gaps);
            }
            let bytes = rmp_serde::to_vec_named(&acc).unwrap();
            let compact = rmp_serde::to_vec(&acc).unwrap();
            let has_field = |b: &[u8]| b.windows(4).any(|w| w == b"held");
            assert_eq!(
                has_field(&bytes),
                !acc.held.is_empty(),
                "{gaps:?}, cut {cut}"
            );
            for b in [bytes, compact] {
                let mut back: Acc = rmp_serde::from_slice(&b).unwrap();
                assert!(back == acc, "{gaps:?}, cut {cut}: the round trip");
                for r in &rows[cut..] {
                    back.learn(&r.z, &r.y, r.lam, r.w, gaps);
                }
                assert!(
                    bits(&back.flushed()) == bits(&whole),
                    "{gaps:?}, saved after row {cut}"
                );
            }
        }
    }
}

/// `n` rows of `k` features near 0 (the first a constant 1) and `t`
/// targets at a level of `1e8`, far above both their spread and the
/// features': a block's own mean `ȳ_B` then rounds at `1e8`'s resolution,
/// and the residue `δy` it leaves is a rounding step of that level, which
/// the fold must carry. And every 16th row, so the first of every block of
/// 16, 32 or 64, is an outlier `1e6` above the others at a weight of
/// `2^-40`: the block's
/// mean is summed on values shifted by its first, so the shift is far
/// from the mean, and a mean the shift did not come back from would leave
/// the scatter `1e6` off centre. Each target has gaps, never on an outlier.
fn far_stream(k: usize, t: usize, n: usize, seed: u64) -> Vec<Row> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let z: Vec<f64> = (0..k)
                .map(|f| if f == 0 { 1.0 } else { lcg(&mut s) })
                .collect();
            let fit = z
                .iter()
                .enumerate()
                .map(|(f, v)| v / (f + 1) as f64)
                .sum::<f64>();
            let outlier = i % 16 == 0;
            let w = match i % 7 {
                _ if outlier => 2f64.powi(-40),
                3 => 0.0,
                1 => 2.5,
                5 => 0.5,
                _ => 1.0,
            };
            let y = (0..t)
                .map(|j| {
                    let v = 1e8 + fit * (1.0 + j as f64) + 0.2 * lcg(&mut s);
                    let v = if outlier { v + 1e6 } else { v };
                    (outlier || i % (5 + 2 * j) != 2).then_some(v)
                })
                .collect();
            Row { z, y, lam: 1.0, w }
        })
        .collect()
}

/// A target far above its spread keeps what its block's mean leaves out,
/// and its block's scatter is taken about its own mean, not about the
/// value the sum was shifted by: within the bound of the definition at
/// every block size. Measured (2026-10-14): the fold at under 0.04 of the
/// bound; a block mean not brought back from the shift, or a residue
/// dropped or of the wrong sign, at 45 to 2.8e5 times it. Under `pairwise`
/// the targets' gaps put their own feature means apart from the Gram's (`ε
/// ≠ δ`), so the residue's own term, `U·ε·δy`, is read as well.
#[test]
fn a_target_far_from_its_spread_is_centred_on_its_own_block_mean() {
    let rows = far_stream(4, 3, 160, 23);
    for gaps in GAPS {
        for block in [16usize, 32, 64] {
            let acc = run(&rows, block, 1, gaps);
            for j in 0..3 {
                let w = worst(&acc, &rows, j, None);
                assert!(
                    w <= 1.0,
                    "{gaps:?}, block {block}, target {j}: {w} of the bound"
                );
            }
        }
    }
}

/// What a blocked fit still takes row by row -- each target's weight, its
/// own mean `ȳ_j`, its moments and its row count, and the weight, mean and
/// row count over every row -- is the unblocked fit's to the bit: the same
/// steps on the same numbers, the block holding only the cross-moments and
/// feature means back. At every block size, both `target_gaps` and each
/// kind of stream, zero weights among the rows, the first row's among
/// them; and each target's count of rows of weight is the rows it was
/// present on with weight.
#[test]
fn what_a_block_takes_row_by_row_is_the_unblocked_fits_to_the_bit() {
    fn per_row(acc: &Acc) -> Vec<u64> {
        let (c, tm) = (&acc.cross, &acc.tm);
        let floats = [c.w]
            .iter()
            .chain(&acc.wj)
            .chain(&c.m)
            .chain(&c.m_lo)
            .chain(&c.my)
            .chain(&c.my_lo)
            .chain(tm.means())
            .chain(tm.means_lo())
            .chain(tm.vars())
            .chain(tm.q())
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        floats
            .into_iter()
            .chain(c.nj.iter().copied())
            .chain([c.n])
            .collect()
    }
    for kind in [Kind::Plain, Kind::Decays, Kind::Reset] {
        for level in [0.0, 1e3] {
            let rows = stream(kind, 4, 3, 140, level, 29);
            for gaps in GAPS {
                let mut unblocked = Acc::new(3, 4, 0, false);
                let mut blocked: Vec<Acc> = [1usize, 3, 16]
                    .iter()
                    .map(|&b| Acc::new(3, 4, b, false))
                    .collect();
                for (i, r) in rows.iter().enumerate() {
                    unblocked.learn(&r.z, &r.y, r.lam, r.w, gaps);
                    let want = per_row(&unblocked);
                    for acc in &mut blocked {
                        acc.learn(&r.z, &r.y, r.lam, r.w, gaps);
                        assert!(
                            per_row(acc) == want,
                            "{kind:?}, level {level}, {gaps:?}, block {}, row {i}",
                            acc.grams.grams[0].block_rows()
                        );
                    }
                }
                // The rows of weight each target was present on, counted.
                let counted: Vec<u64> = (0..3)
                    .map(|j| {
                        rows.iter()
                            .filter(|r| r.y[j].is_some() && r.w > 0.0)
                            .count() as u64
                    })
                    .collect();
                assert_eq!(unblocked.cross.nj, counted, "{kind:?}, {gaps:?}");
            }
        }
    }
}

/// A row of no weight leaves a target's own mean `ȳ_j` where it was, to the
/// bit, under a blocked Gram as without one (CLAUDE.md hard rule 9): its
/// step is not taken, where a step of 0 would round the pair afresh. After
/// `0.1` at weight 1 and `1.0` at weight `0.5` the pair is `0.4 −
/// 5.6e-17`, whose low part is a whole rounding step of `0.4`, and adding 0
/// to it gives `0.39999999999999997 + 0`: the same number, other bits.
#[test]
fn a_row_of_no_weight_leaves_a_targets_mean_where_it_was() {
    use crate::comp::{add, dev};
    for gaps in GAPS {
        for block in [0usize, 4] {
            let mut acc = Acc::new(1, 2, block, false);
            let z = [1.0, 0.5];
            acc.learn(&z, &[Some(0.1)], 1.0, 1.0, gaps);
            acc.learn(&z, &[Some(1.0)], 1.0, 0.5, gaps);
            let before = (acc.cross.my[0], acc.cross.my_lo[0]);
            // The case is the one it is for: a step of 0 moves these bits.
            let (mut hi, mut lo) = before;
            let step = 0.0 * dev(3.0, hi, lo);
            add(&mut hi, &mut lo, step);
            assert_ne!(
                (hi.to_bits(), lo.to_bits()),
                (before.0.to_bits(), before.1.to_bits()),
                "a step of 0 leaves this pair alone; no case"
            );
            acc.learn(&z, &[Some(3.0)], 1.0, 0.0, gaps);
            assert_eq!(
                (acc.cross.my[0].to_bits(), acc.cross.my_lo[0].to_bits()),
                (before.0.to_bits(), before.1.to_bits()),
                "{gaps:?}, block {block}"
            );
        }
    }
}

/// A row whose weight the rest of its block decays to nothing is not a row
/// of the block's for a target, as it is not for the Gram (`u = 0`): a
/// target present on no other row of the block keeps its cross-moments and
/// feature means where they were, as if it had been absent there. Its row
/// weighs `5e-324`, the least double, and a decay of `1/2` after it in the
/// same block rounds that to 0.
#[test]
fn a_row_the_block_decays_to_nothing_is_not_the_targets() {
    for gaps in GAPS {
        // Target 1 on the first block's rows, then only on the second
        // block's first row, the one of weight `5e-324`.
        let rows = |present: bool| -> Vec<Row> {
            (0..8)
                .map(|i| {
                    let x = i as f64;
                    let z = vec![1.0, 0.3 * x - 0.1 * x * x];
                    let w = if i == 4 { 5e-324 } else { 1.0 };
                    let lam = if i == 5 { 0.5 } else { 1.0 };
                    let y1 = (i < 4 || (i == 4 && present)).then_some(0.2 * x + 1.0);
                    Row {
                        z,
                        y: vec![Some(x), y1],
                        lam,
                        w,
                    }
                })
                .collect()
        };
        let (with, without) = (run(&rows(true), 4, 1, gaps), run(&rows(false), 4, 1, gaps));
        let of1 = |acc: &Acc| -> Vec<u64> {
            acc.cross.c[1]
                .iter()
                .chain(&acc.cross.mj[1])
                .chain(&acc.cross.mj_lo[1])
                .map(|v| v.to_bits())
                .collect()
        };
        assert!(with.cross.c[1].iter().all(|c| c.is_finite()), "{gaps:?}");
        assert!(of1(&with) == of1(&without), "{gaps:?}");
        assert_eq!(
            with.cross.nj[1], 5,
            "{gaps:?}: the row of weight 5e-324 is a row"
        );
    }
}

/// A held block a damaged state could carry -- a target holding more rows
/// than its Gram, cross-moments of another width, a weight after the block
/// of 0, infinite or NaN -- is dropped, its target's moments left where
/// they were: never read past the block, and never divided by a weight
/// that is not one. `Acc::has_shape` refuses the first two on load; the
/// weight is not a shape.
#[test]
fn a_damaged_held_block_is_dropped_whole() {
    let rows = stream(Kind::Plain, 3, 2, 12, 0.0, 31);
    for gaps in GAPS {
        let mut acc = Acc::new(2, 3, 16, false);
        for r in &rows {
            acc.learn(&r.z, &r.y, r.lam, r.w, gaps);
        }
        assert!(
            acc.held.y.iter().all(|v| v.len() > 2),
            "{gaps:?}: nothing held"
        );
        let damages: [Damage<'_, Acc>; 5] = [
            ("more rows than the Gram's", &|a| {
                let more = a.grams.grams[a.grams.of[1]].pending_len() + 1;
                a.held.y[1].resize(more, 0.5);
            }),
            ("cross-moments a slot short", &|a| {
                a.cross.c[1].pop();
            }),
            ("a weight of 0", &|a| a.held.w_end[1] = 0.0),
            ("an infinite weight", &|a| a.held.w_end[1] = f64::INFINITY),
            ("a weight of NaN", &|a| a.held.w_end[1] = f64::NAN),
        ];
        for (what, damage) in damages {
            let mut a = acc.clone();
            damage(&mut a);
            let (c, mj) = (a.cross.c[1].clone(), a.cross.mj[1].clone());
            a.flush();
            assert!(a.held.is_empty(), "{gaps:?}, {what}: still held");
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.cross.c[1]), bits(&c), "{gaps:?}, {what}");
            assert_eq!(bits(&a.cross.mj[1]), bits(&mj), "{gaps:?}, {what}");
            // Target 0's block is whole, and folded.
            assert_ne!(a.cross.mj[0], acc.cross.mj[0], "{gaps:?}, {what}: target 0");
        }
    }
}

/// A target on a light share of a block under `pairwise` takes its block
/// mean from its own rows, not as the whole block's sum less the rows it
/// misses (review round 8, A-1; docs/PLAN.md task 269): that difference
/// carried the whole block's rounding, amplified by the block's weight over
/// the target's, 300 to 506,290 ulps for a target on one row of a 256-row
/// block where the row-by-row path is exact. Held here to the bound of a
/// sum over the target's own rows, against the exact oracle: target 1 on
/// one row of the block at weight `2^-10` (or `2^-20`), target 2 on two,
/// beside target 0 on every row, the features at a level ten times their
/// spread. Measured (2026-10-16): on `19ffdd5` the light targets at 11 to
/// 26 times the bound at `2^-10` and 21,000 to 31,000 times at `2^-20`;
/// from their own rows, 0 (one row: exact) and at most 0.0023 of it.
#[test]
fn a_light_target_takes_its_block_mean_from_its_own_rows() {
    let block = 256;
    for (light, level, spread) in [(2f64.powi(-10), 1e3, 1e2), (2f64.powi(-20), 1e4, 1e3)] {
        let mut s = 77u64;
        let rows: Vec<Row> = (0..block)
            .map(|i| {
                let z = vec![
                    1.0,
                    level + spread * lcg(&mut s),
                    -level + spread * lcg(&mut s),
                ];
                let fit = (z[1] + z[2]) / spread;
                let one = i == block - 5;
                // Target 2's rows sit first and third of their fours (216
                // and 246), target 1's last (251): the fused pass skips a
                // four by all four of its coefficients, each position held.
                let two = i == block - 40 || i == block - 10;
                let w = if one || two { light } else { 1.0 };
                let y = vec![
                    Some(fit + 0.1 * lcg(&mut s)),
                    one.then(|| fit + 0.1 * lcg(&mut s)),
                    two.then(|| 2.0 * fit + 0.1 * lcg(&mut s)),
                ];
                Row { z, y, lam: 1.0, w }
            })
            .collect();
        for gaps in GAPS {
            for b in [0usize, block] {
                let acc = run(&rows, b, 1, gaps);
                for j in 0..3 {
                    let w = worst(&acc, &rows, j, None);
                    assert!(
                        w <= 1.0,
                        "light {light:e}, {gaps:?}, block {b}, target {j}: {w} of the bound"
                    );
                }
            }
        }
        // The case is the one it is for: one Gram over every row, whose
        // block the light targets miss most of.
        let mut acc = Acc::new(3, 3, block, false);
        for r in &rows[..block - 1] {
            acc.learn(&r.z, &r.y, r.lam, r.w, TargetGaps::Pairwise);
        }
        assert_eq!(acc.grams.grams.len(), 1);
        let present = |j: usize| acc.held.y[j].iter().filter(|v| !v.is_nan()).count();
        assert_eq!((present(1), present(2)), (1, 2), "the light rows held");
        assert!(
            acc.held.y[1].iter().any(|v| v.is_nan()),
            "target 1 misses rows"
        );
        assert!(acc.held.y[2].len() > 30 && acc.held.y[2].iter().any(|v| v.is_nan()));
    }
}

/// A weight past `1.34e154`, whose square overflows, or below `1.5e-154`,
/// whose square leaves the normal range, merges a block as a row at a time
/// does (review round 8, A-2; docs/PLAN.md task 269): the between term's
/// `H·U/W'²` read `inf/inf` or `0/0` there, a NaN the moments never shed.
/// Every weight scaled by `2^515` (about `1.1e155`) and by `2^-560` (about
/// `2.6e-169`), exactly, so the oracle's discounts stay exact: within the
/// bound of the definition, row by row and in blocks, under both
/// `target_gaps`.
#[test]
fn a_weight_whose_square_leaves_the_doubles_merges_a_block() {
    // The six rows of `Kind::Reset` at weight `1e-310` alone, where `1/W'`
    // is infinite at every merge and nothing after them washes a wrong
    // `S'/W'` out (task 269, the mutants after review round 8). A subnormal
    // weight keeps about 44 bits, so no path meets the oracle's bound here
    // (row by row reads 430 times it); each block is held to the rows
    // instead, closely enough that a share read wrong, an error of the
    // moments' own size, cannot pass.
    let rows = stream(Kind::Reset, 4, 3, 140, 1e3, 11);
    let rows = &rows[..6];
    assert!(rows.iter().all(|r| r.w == 1e-310));
    for gaps in GAPS {
        let seq = run(rows, 0, 1, gaps);
        for block in [1usize, 2, 3] {
            let acc = run(rows, block, 1, gaps);
            for j in 0..3 {
                assert!(defined(rows, j, None).is_some(), "target {j} has rows");
                for (what, a, b) in [
                    ("c", &seq.cross.c[j], &acc.cross.c[j]),
                    ("m", &seq.cross.mj[j], &acc.cross.mj[j]),
                ] {
                    let size = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    assert!(size > 0.0, "{what} of target {j} is not all 0");
                    for (x, y) in a.iter().zip(b.iter()) {
                        assert!(
                            (x - y).abs() <= 1e-9 * size,
                            "subnormal rows, {gaps:?}, block {block}, target {j}, {what}: \
                             {y:e} against the rows' {x:e}"
                        );
                    }
                }
            }
        }
    }
    for scale in [2f64.powi(515), 2f64.powi(-560)] {
        let rows: Vec<Row> = stream(Kind::Decays, 4, 3, 60, 1e3, 37)
            .into_iter()
            .map(|r| Row {
                w: r.w * scale,
                ..r
            })
            .collect();
        for gaps in GAPS {
            for block in [0usize, 3, 16] {
                let acc = run(&rows, block, 1, gaps);
                for j in 0..3 {
                    let w = worst(&acc, &rows, j, None);
                    assert!(
                        w <= 1.0,
                        "scale {scale:e}, {gaps:?}, block {block}, target {j}: {w} of the bound"
                    );
                }
            }
        }
    }
}
