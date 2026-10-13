//! The targets folded into a blocked Gram's merge (docs/PLAN.md task 257):
//! each target's centred cross-moments and feature means, held to an
//! exactly rounded oracle of their definition at every block size, thread
//! count and `target_gaps`, with gaps, zero weights, decays that halve or
//! take the whole history, and a reset of one target's own; to the bit at
//! every thread count and whatever other targets share the block; and a
//! state from before the fold, whose targets were current through the held
//! rows, going on from it.

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
        worst = worst
            .max((m - d.m[f]).abs() / m_bound)
            .max((acc.cross.c[j][f] - d.c[f]).abs() / c_bound);
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
