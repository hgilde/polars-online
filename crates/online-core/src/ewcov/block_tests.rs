use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
}

/// `(x, lam, w)` rows exercising everything a block has to survive:
/// unequal weights with zeros, decay, and a feature offset far from zero
/// so that a naive uncentred product would lose the variance.
fn stream(
    n: usize,
    k: usize,
    seed: u64,
    offset: f64,
    lam: f64,
    wmode: u8,
) -> Vec<(Vec<f64>, f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let x: Vec<f64> = (0..k).map(|_| offset + lcg(&mut s)).collect();
            let w = match wmode {
                0 => 1.0,
                1 => 0.2 + 1.5 * (lcg(&mut s) + 1.0) / 2.0,
                _ => {
                    if i % 7 == 0 {
                        0.0
                    } else {
                        1.0
                    }
                }
            };
            (x, lam, w)
        })
        .collect()
}

fn run(rows: &[(Vec<f64>, f64, f64)], k: usize, block: usize) -> EwCov {
    let mut c = EwCov::new(k);
    if block > 0 {
        c.set_block_rows(block);
    }
    for (x, lam, w) in rows {
        c.update(x, *lam, *w);
    }
    c.flush();
    c
}

/// The mean and co-moments are a reassociation, so they agree closely
/// rather than exactly. The co-moments are compared **against the
/// matrix's own scale**, not entry by entry: an off-diagonal co-moment
/// of two independent features is a small number that happens to be
/// near zero, and a per-entry relative tolerance turns its rounding into
/// a failure while saying nothing about the merge (at `k = 2,000` the
/// worst per-entry ratio is `1.6e-5` on a merge that is right to `5e-13`
/// of the largest variance).
///
/// The tolerance is the **data's own resolution**: a feature at `1e7`
/// is stored to `1e7·ε ≈ 2e-9`, so a centred value of order `σ` is only
/// known to `1e7·ε/σ` and the two paths, which round it in different
/// orders, legitimately differ by that much (`2.5e-9` of the variance at
/// `k = 1`, block 2, offset `1e7`, where the floor of `1e-9` failed).
/// The design measured `5e-13` of the largest variance at an offset of
/// `1e3` and `7e-10` at `1e7`, so a wrong term -- of order `1` -- is a
/// million times the bound at either offset.
fn assert_moments_close(seq: &EwCov, blk: &EwCov, what: &str) {
    let k = seq.k;
    let scale = (0..k)
        .map(|i| seq.c[i * k + i].abs())
        .fold(0.0f64, f64::max)
        .max(1e-300);
    let magnitude = seq.m.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    let resolution = magnitude * f64::EPSILON / scale.sqrt();
    let tol = 1e-9f64.max(64.0 * resolution);
    for i in 0..k {
        let (a, b) = (seq.m[i], blk.m[i]);
        assert!(
            (a - b).abs() <= 1e-12 * a.abs().max(1.0),
            "mean[{i}] {a} vs {b}, {what}"
        );
    }
    for i in 0..k * k {
        let (a, b) = (seq.c[i], blk.c[i]);
        assert!(
            (a - b).abs() <= tol * a.abs().max(scale),
            "c[{i}] {a} vs {b}: {:.2e} of the scale {scale:.3e}, tol {tol:.1e}, {what}",
            (a - b).abs() / scale
        );
    }
}

/// The four scalars must be **bit-identical**, not close.
///
/// They are what `n_eff` and `n_kish` are read from, and those are
/// emitted on every row, so anything less than equality would change
/// shipped output the moment blocking was switched on. It holds because
/// both paths run the one copy of `step_factors` / `commit_scalars`.
fn assert_scalars_equal(seq: &EwCov, blk: &EwCov, what: &str) {
    assert_eq!(seq.w_sum, blk.w_sum, "w_sum, {what}");
    assert_eq!(seq.q_sum, blk.q_sum, "q_sum, {what}");
    assert_eq!(seq.prior_scale, blk.prior_scale, "prior_scale, {what}");
    assert_eq!(
        seq.precision_scale, blk.precision_scale,
        "precision_scale, {what}"
    );
}

/// A row skipped on an empty blocked accumulator ages the prior's scale
/// as the plain one does: `skip` takes the decay, not the buffer, at
/// `w_sum = 0` (review 2026-09-26, C1: the mutant that divides instead
/// of multiplying takes the buffer, which returns before the scalars).
#[test]
fn a_skipped_row_on_an_empty_blocked_gram_ages_the_prior_as_the_plain_one() {
    let mut plain = EwCov::new(2);
    let mut blocked = EwCov::new(2);
    blocked.set_block_rows(4);
    plain.skip(&[1.0, 2.0], 0.9);
    blocked.skip(&[1.0, 2.0], 0.9);
    assert_eq!(plain.prior_scale(), blocked.prior_scale());
    assert_eq!(blocked.prior_scale(), 0.9);
}

#[test]
fn the_scalars_are_bit_identical_whichever_path_ran() {
    for &k in &[1usize, 3, 8, 37] {
        for &lam in &[1.0f64, 0.995] {
            for wmode in 0u8..3 {
                for &block in &[2usize, 7, 64] {
                    let rows = stream(200, k, 11, 1e3, lam, wmode);
                    let seq = run(&rows, k, 0);
                    let blk = run(&rows, k, block);
                    let what = format!("k={k} lam={lam} wmode={wmode} block={block}");
                    assert_scalars_equal(&seq, &blk, &what);
                }
            }
        }
    }
}

#[test]
fn the_moments_match_the_rank_one_recursion() {
    for &k in &[1usize, 3, 8, 37] {
        for &lam in &[1.0f64, 0.995] {
            for wmode in 0u8..3 {
                for &block in &[2usize, 7, 64] {
                    for &offset in &[0.0f64, 1e3, 1e7] {
                        let rows = stream(200, k, 5, offset, lam, wmode);
                        let seq = run(&rows, k, 0);
                        let blk = run(&rows, k, block);
                        let what =
                            format!("k={k} lam={lam} wmode={wmode} block={block} offset={offset}");
                        assert_moments_close(&seq, &blk, &what);
                    }
                }
            }
        }
    }
}

/// The product is `faer`'s, and a GEMM has tile edges the tests above
/// never reach at `k <= 37`: a block shorter than `k`, a block longer
/// than `k`, and a `k` that is not a multiple of any vector width.
#[test]
fn a_wide_block_matches_too() {
    let k = 131;
    let rows = stream(300, k, 17, 1e3, 0.998, 1);
    let seq = run(&rows, k, 0);
    for &block in &[64usize, 256] {
        let blk = run(&rows, k, block);
        let what = format!("k={k} block={block}");
        assert_scalars_equal(&seq, &blk, &what);
        assert_moments_close(&seq, &blk, &what);
    }
}

/// The merge's numerical question, put as a test. A long uncapped gap
/// leaves the history *nearly* gone -- `W_A` of `1e-30`, not zero -- with
/// `m_A` still where the old data was. A merge centred on `m_A` then
/// forms the block's scatter as a difference of two numbers of order
/// `U·(m_B − m_A)²`, and at an offset of `1e7` that leaves an error of
/// **17% of the variance** in this test (49% in the design probe, and
/// 24% centred on the block's first weighted row, which here is the one
/// *before* the gap). The rank-1 recursion is exact, because its mean
/// jumps to the row. Centred on the block's own mean the merge measures
/// `8.4e-10` of the variance, which is the data's own resolution at
/// `1e7`, and `3e-14` at `1e3`. Verified to fail, at the 17%, with the
/// centring switched to `m_A`.
#[test]
fn a_history_all_but_gone_leaves_nothing_to_cancel() {
    let k = 2;
    for &(gap_lam, offset) in &[(1e-200f64, 1e7f64), (1e-30, 1e7), (1e-30, 1e3), (1e-8, 1e3)] {
        let mut rows = stream(50, k, 3, 0.0, 1.0, 0);
        let far = stream(8, k, 4, offset, 1.0, 0);
        for (i, (x, _, w)) in far.into_iter().enumerate() {
            let lam = if i == 0 { gap_lam } else { 1.0 };
            rows.push((x, lam, w));
        }
        // A block of 8 over 58 rows puts two old rows and the gap in
        // one block, which is the arrangement that fooled the row origin.
        let seq = run(&rows, k, 0);
        let blk = run(&rows, k, 8);
        let what = format!("gap_lam={gap_lam:e} offset={offset:e}");
        assert_scalars_equal(&seq, &blk, &what);
        let scale = seq.c[0].max(seq.c[3]);
        for i in 0..k * k {
            let (a, b) = (seq.c[i], blk.c[i]);
            assert!(
                (a - b).abs() <= 1e-8 * scale,
                "c[{i}] {a} vs {b}: {:.2e} of the variance, {what}",
                (a - b).abs() / scale
            );
        }
        for i in 0..k {
            assert!(
                (seq.m[i] - blk.m[i]).abs() <= 4.0 * f64::EPSILON * seq.m[i].abs().max(1.0),
                "mean[{i}], {what}"
            );
        }
    }
}

/// `EwCov` has no notion of a chunk: rows arrive one call at a time, and
/// a block boundary is a function of the row sequence alone. So this
/// cannot fail, and is here as a pin of that fact and a place to write
/// down the rule it implies for the caller.
///
/// **A flush forced at a chunk boundary would move the last digits** --
/// a flush every row is the rank-1 association, a flush every sixteen
/// is a merge -- which is exactly the chunk-dependence CLAUDE.md rule 3
/// forbids. So the flush may only ever be triggered by things that are
/// functions of the row sequence: the block filling, and a read on a
/// learned-row schedule (`solve_every`). It may never be triggered by a
/// chunk ending. The test with teeth is the bank's chunk-invariance
/// test with `gram_block_rows` on, which straddles block boundaries
/// with chunk boundaries (docs/PLAN.md task 71).
#[test]
fn handing_the_rows_over_differently_changes_nothing() {
    let k = 4;
    let rows = stream(150, k, 23, 1e3, 0.997, 1);
    let whole = run(&rows, k, 16);
    for pause in [1usize, 5, 16, 17, 31, 64] {
        let mut c = EwCov::new(k);
        c.set_block_rows(16);
        for chunk in rows.chunks(pause) {
            for (x, lam, w) in chunk {
                c.update(x, *lam, *w);
            }
        }
        c.flush();
        assert_eq!(c.w_sum, whole.w_sum, "w_sum at pause={pause}");
        assert_eq!(
            c.c, whole.c,
            "c differs when the rows arrive {pause} at a time"
        );
    }
}

/// The block fills on the row that makes it `block_rows` long, and on
/// no other: one short holds, the exact count merges.
#[test]
fn the_block_fills_on_exactly_the_last_row() {
    let k = 3;
    let rows = stream(40, k, 19, 1e3, 0.99, 0);
    let mut c = EwCov::new(k);
    c.set_block_rows(8);
    for (i, (x, lam, w)) in rows.iter().enumerate() {
        c.update(x, *lam, *w);
        let held = (i + 1) % 8 != 0;
        assert_eq!(
            c.has_pending(),
            held,
            "after row {i}: {} rows held",
            c.pending.len()
        );
        if held {
            assert_eq!(c.pending.len(), (i + 1) % 8);
        }
    }
}

/// Flushing early must be indistinguishable from not needing to: a read
/// mid-block is allowed and does not change the sequence's answer.
#[test]
fn an_early_flush_is_not_a_different_stream() {
    let k = 3;
    let rows = stream(90, k, 31, 0.0, 0.99, 2);
    let never = run(&rows, k, 32);
    let mut every = EwCov::new(k);
    every.set_block_rows(32);
    for (x, lam, w) in &rows {
        every.update(x, *lam, *w);
        every.flush(); // degenerates to the rank-1 path
    }
    assert_scalars_equal(&never, &every, "flush every row");
    assert_moments_close(&never, &every, "flush every row");
}

/// A read with rows still held is the one mistake the `ewridge` wiring
/// can make, and it would be silent -- stale moments, not a crash -- so
/// every matrix read asserts in debug builds. This is the test that
/// checks the assertion is there.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "block still pending")]
fn reading_the_matrix_mid_block_is_caught() {
    let k = 2;
    let mut c = EwCov::new(k);
    c.set_block_rows(4);
    c.update(&[1.0, 2.0], 1.0, 1.0);
    let _ = c.cov(0, 0);
}

/// The scalar reads are the ones a per-row emitter needs, and they are
/// advanced by the shipped path as each row is held, so they are never
/// stale and are not guarded.
#[test]
fn the_scalars_are_readable_mid_block() {
    let k = 2;
    let rows = stream(5, k, 29, 1e3, 0.99, 1);
    let mut seq = EwCov::new(k);
    let mut blk = EwCov::new(k);
    blk.set_block_rows(8);
    for (x, lam, w) in &rows {
        seq.update(x, *lam, *w);
        blk.update(x, *lam, *w);
        assert!(blk.has_pending());
        assert_eq!(blk.n_eff(), seq.n_eff());
        assert_eq!(blk.q_sum(), seq.q_sum());
        assert_eq!(blk.prior_scale(), seq.prior_scale());
    }
}

/// A clock gap past `gap_cap` is capped there, and with the documented
/// defaults that cap is a `lam` of `2^-10000` -- **exactly zero**
/// (CLAUDE.md rule 9 and docs/PLAN.md task 67). It is reachable from any
/// stream with one long pause, and it lands *inside* a block, so the
/// merge has to survive it. None of the other tests here use a `lam`
/// below 0.99.
///
/// It does *not* empty the accumulator, which is what this test was
/// called before the branch was checked for reachability: `lam = 0`
/// makes the row's own weight the whole of `w_sum`, so `step_factors`
/// accepts it and the merge divides by a positive number.
#[test]
fn a_capped_gap_inside_a_block_matches_the_sequential_path() {
    for &block in &[2usize, 8, 64] {
        for at in [0usize, 1, 5, 30] {
            let k = 3;
            let mut rows = stream(60, k, 3, 1e3, 1.0, 1);
            rows[at].1 = 0.0; // the capped gap
            let seq = run(&rows, k, 0);
            let blk = run(&rows, k, block);
            let what = format!("block={block} gap_at={at}");
            assert_scalars_equal(&seq, &blk, &what);
            assert_moments_close(&seq, &blk, &what);
        }
    }
}

/// Every row in a block carrying zero weight: the history only ages, and
/// `E_w` of a history that has merely aged is what it was. This reaches
/// the no-weight arm of the merge, which no other test does -- the
/// `wmode = 2` streams never put two zeros next to each other.
#[test]
fn a_block_of_nothing_but_zero_weight_rows_only_ages() {
    let k = 3;
    let mut rows = stream(40, k, 9, 1e3, 0.99, 0);
    for r in rows.iter_mut().skip(8).take(6) {
        r.2 = 0.0; // six consecutive zero-weight rows, inside one block
    }
    let seq = run(&rows, k, 0);
    let blk = run(&rows, k, 4);
    assert_scalars_equal(&seq, &blk, "a zero-weight block");
    assert_moments_close(&seq, &blk, "a zero-weight block");
}

/// A zero-weight row is legal as the **first** row of a stream, where it
/// means "advance the clock, learn nothing" (CLAUDE.md rule 9). It must
/// not open a block: it is the decay alone, which holds nothing and
/// ages the prior's scale as the clock advances (task 159, R1; it once
/// did not age that row either).
#[test]
fn a_zero_weight_first_row_opens_no_block() {
    let k = 2;
    let mut c = EwCov::new(k);
    c.set_block_rows(4);
    c.update(&[1e3, 1e3], 0.5, 0.0);
    assert!(
        !c.has_pending(),
        "a row the shipped path ignores must not be held"
    );
    assert_eq!(c.w_sum, 0.0);
    assert_eq!(c.prior_scale, 0.5, "the row ages the prior's scale");

    let rows = stream(30, k, 13, 1e3, 0.995, 0);
    let mut with_lead = vec![(vec![1e3; k], 0.5, 0.0)];
    with_lead.extend(rows.iter().cloned());
    let a = run(&with_lead, k, 4);
    let b = run(&rows, k, 4);
    assert_eq!(a.w_sum, b.w_sum, "the leading no-op changed the count");
    assert_eq!(a.c, b.c, "the leading no-op changed the moments");
}

/// The pending rows travel in the state, so a bank saved mid-block and
/// resumed must continue as though it never stopped -- including the
/// block boundary, which is a function of the row sequence. Both
/// msgpack encodings: the bank writes the named one, and the compact one
/// is where a `skip_serializing_if` field in the wrong place shows
/// (`crates/online-core/tests/state_encoding.rs`).
#[test]
fn a_state_saved_mid_block_resumes_identically() {
    let k = 3;
    let rows = stream(80, k, 41, 1e3, 0.996, 1);
    let whole = run(&rows, k, 16);
    for cut in [1usize, 7, 16, 23, 40] {
        let mut c = EwCov::new(k);
        c.set_block_rows(16);
        for (x, lam, w) in &rows[..cut] {
            c.update(x, *lam, *w);
        }
        let named = rmp_serde::to_vec_named(&c).unwrap();
        let compact = rmp_serde::to_vec(&c).unwrap();
        for (bytes, enc) in [(named, "named"), (compact, "compact")] {
            let mut back: EwCov = rmp_serde::from_slice(&bytes)
                .unwrap_or_else(|e| panic!("{enc} encoding at cut={cut}: {e}"));
            assert_eq!(back, c, "{enc} round-trip changed the state at cut={cut}");
            assert_eq!(
                back.has_pending(),
                c.has_pending(),
                "pending lost at cut={cut} ({enc})"
            );
            for (x, lam, w) in &rows[cut..] {
                back.update(x, *lam, *w);
            }
            back.flush();
            assert_eq!(
                back.w_sum, whole.w_sum,
                "w_sum after resume at {cut} ({enc})"
            );
            assert_eq!(back.c, whole.c, "moments after resume at {cut} ({enc})");
        }
    }
}

/// The JSON export (task 69) is for reading, so the held block must come
/// out as named, finite fields a reader can navigate, and go back in.
#[test]
fn a_held_block_is_legible_in_json() {
    let k = 2;
    let rows = stream(3, k, 43, 1e3, 0.99, 1);
    let mut c = EwCov::new(k);
    c.set_block_rows(8);
    for (x, lam, w) in &rows {
        c.update(x, *lam, *w);
    }
    let v = serde_json::to_value(&c).unwrap();
    let p = &v["pending"];
    assert_eq!(p["block_rows"], 8);
    assert_eq!(p["lam"].as_array().unwrap().len(), 3);
    assert_eq!(p["w"].as_array().unwrap().len(), 3);
    assert_eq!(p["x"].as_array().unwrap().len(), 3 * k, "row-major n*k");
    assert!(p["w_open"].is_number());
    let back: EwCov = serde_json::from_value(v).unwrap();
    assert_eq!(back, c);

    // And blocking off writes what it always did: no `pending` at all.
    let plain = serde_json::to_value(EwCov::new(k)).unwrap();
    assert!(plain.get("pending").is_none(), "{plain}");
}

/// A block of one, and a block the stream never fills: the two ends of
/// the parameter. Neither may lose a row.
#[test]
fn a_block_of_one_and_a_block_longer_than_the_stream() {
    let k = 3;
    let rows = stream(25, k, 55, 1e3, 0.995, 1);
    let seq = run(&rows, k, 0);
    for &block in &[1usize, 1000] {
        let blk = run(&rows, k, block);
        let what = format!("block={block}");
        assert_scalars_equal(&seq, &blk, &what);
        assert_moments_close(&seq, &blk, &what);
    }
}

/// Changing the block size mid-stream flushes what is held first, so no
/// row is carried across a size change and none is dropped.
#[test]
fn changing_the_block_size_mid_stream_keeps_every_row() {
    let k = 3;
    let rows = stream(60, k, 67, 1e3, 0.995, 1);
    let seq = run(&rows, k, 0);
    let mut c = EwCov::new(k);
    c.set_block_rows(8);
    for (i, (x, lam, w)) in rows.iter().enumerate() {
        if i == 21 {
            c.set_block_rows(3);
        }
        if i == 44 {
            c.set_block_rows(0); // and off entirely
        }
        c.update(x, *lam, *w);
    }
    c.flush();
    assert_scalars_equal(&seq, &c, "resized mid-stream");
    assert_moments_close(&seq, &c, "resized mid-stream");
}

/// `decay` ages `w_sum`, which a held block has snapshotted as `w_open`.
/// Ageing it underneath the block leaves the merge computing `W_A` from a
/// weight that no longer exists, so `decay` merges first. Verified to
/// fail without that flush.
#[test]
fn ageing_the_accumulator_mid_block_does_not_strand_the_held_rows() {
    let k = 3;
    let rows = stream(40, k, 71, 1e3, 1.0, 0);
    // sequential: the same interruption, with nothing held to strand
    let mut seq = EwCov::new(k);
    for (i, (x, lam, w)) in rows.iter().enumerate() {
        if i == 11 {
            seq.decay(0.5);
        }
        seq.update(x, *lam, *w);
    }
    let mut blk = EwCov::new(k);
    blk.set_block_rows(8);
    for (i, (x, lam, w)) in rows.iter().enumerate() {
        if i == 11 {
            blk.decay(0.5); // lands mid-block, with three rows held
        }
        blk.update(x, *lam, *w);
    }
    blk.flush();
    assert_scalars_equal(&seq, &blk, "a mid-block decay");
    assert_moments_close(&seq, &blk, "a mid-block decay");
}

/// `set_moments` replaces the mean, the co-moments and the weight at
/// once. Rows still held at that point were centred on a state that is
/// about to vanish, so they must not survive the call to be merged into
/// the caller's state later; `set_moments` flushes them first. That the
/// flush merges rather than drops is not observable here -- the caller's
/// values overwrite the result either way -- which is the point: what
/// protects a caller that read the moments *before* flushing is the
/// assertion on the reads, not this. `EwRidge::blend_toward_long_run` is
/// the caller that makes it reachable as soon as `ewridge` turns
/// blocking on. Verified to fail without the flush.
#[test]
fn replacing_the_moments_mid_block_leaves_no_row_behind() {
    let k = 2;
    let rows = stream(24, k, 83, 1e3, 0.99, 1);
    let mut blk = EwCov::new(k);
    blk.set_block_rows(8);
    // The reference sees the same rows and the same replacement, with
    // nothing held: `set_moments` leaves `prior_scale` and
    // `precision_scale` alone, so the history before it still counts.
    let mut seq = EwCov::new(k);
    for (x, lam, w) in rows.iter().take(5) {
        seq.update(x, *lam, *w);
        blk.update(x, *lam, *w);
    }
    assert!(
        blk.has_pending(),
        "the test needs rows held to mean anything"
    );
    let (m, c) = ([7.0, 9.0], [1.0, 0.0, 0.0, 1.0]);
    seq.set_moments(&m, &c, 3.0, Some(3.0));
    blk.set_moments(&m, &c, 3.0, Some(3.0));
    assert!(!blk.has_pending(), "set_moments must not leave rows held");
    assert_eq!(blk.w_sum, 3.0, "and the caller's weight is what stands");
    assert_eq!(blk.means(), &m);
    assert_eq!(blk.comoments(), &c);
    assert_scalars_equal(&seq, &blk, "at set_moments");
    // ...and rows after it merge against the caller's state, not the old
    // one: the same as the rank-1 path from that state on.
    for (x, lam, w) in rows.iter().skip(5) {
        seq.update(x, *lam, *w);
        blk.update(x, *lam, *w);
    }
    blk.flush();
    assert_scalars_equal(&seq, &blk, "after set_moments");
    assert_moments_close(&seq, &blk, "after set_moments");
}

/// Blocking off must be the code that shipped, to the bit.
#[test]
fn block_rows_zero_is_the_untouched_path() {
    let k = 5;
    let rows = stream(120, k, 77, 1e3, 0.995, 1);
    let a = run(&rows, k, 0);
    let mut b = EwCov::new(k);
    b.set_block_rows(8);
    b.set_block_rows(0); // asked for, then turned off again
    for (x, lam, w) in &rows {
        b.update(x, *lam, *w);
    }
    assert_eq!(a.w_sum, b.w_sum);
    assert_eq!(a.m, b.m);
    assert_eq!(a.c, b.c, "turning blocking off must restore the exact path");
}

/// A reader that must not move the block boundary -- the bank's
/// `gram()` mid-stream -- gets the merged moments from a copy: what it
/// reads is what the shipped recursion says, and the original goes on
/// holding its rows, so the stream after the read is the stream there
/// would have been without it. With nothing pending it is a borrow.
#[test]
fn a_read_through_flushed_leaves_the_block_where_it_was() {
    use std::borrow::Cow;
    let k = 4;
    let rows = stream(45, k, 91, 1e3, 0.99, 1);
    let seq = run(&rows, k, 0);
    let mut blk = EwCov::new(k);
    blk.set_block_rows(16);
    for (x, lam, w) in &rows {
        blk.update(x, *lam, *w);
    }
    assert!(blk.has_pending(), "45 rows in blocks of 16 leave 13 held");
    let read = blk.flushed();
    assert!(matches!(read, Cow::Owned(_)));
    assert_moments_close(&seq, &read, "flushed()");
    assert_scalars_equal(&seq, &read, "flushed()");
    assert!(blk.has_pending(), "the read merged the model's own block");
    // The stream continues as if nothing had been read.
    let more = stream(20, k, 92, 1e3, 0.99, 1);
    let mut untouched = EwCov::new(k);
    untouched.set_block_rows(16);
    for (x, lam, w) in rows.iter().chain(&more) {
        untouched.update(x, *lam, *w);
    }
    for (x, lam, w) in &more {
        blk.update(x, *lam, *w);
    }
    untouched.flush();
    blk.flush();
    assert_eq!(untouched.c, blk.c, "the read changed the stream after it");
    assert_eq!(untouched.m, blk.m);
    assert!(matches!(blk.flushed(), Cow::Borrowed(_)));
    assert!(matches!(seq.flushed(), Cow::Borrowed(_)));
}

/// A block whose own mean rounds: `1` and `1 + 3ε` at equal weight have
/// the mean `1 + 1.5ε`, which rounds to `1 + 2ε`, so the scatter about
/// it is `5ε²` where the scatter about the true mean is `4.5ε²`. The
/// residue `δ = −ε/2` takes the difference back out, and the block's
/// variance is the definition's `(1.5ε)²` to the bit, as the per-row
/// recursion's is (task 158).
#[test]
fn a_block_whose_mean_rounds_takes_the_residue_back_out() {
    let e = f64::EPSILON;
    let rows = [(vec![1.0], 1.0, 1.0), (vec![1.0 + 3.0 * e], 1.0, 1.0)];
    let blk = run(&rows, 1, 2);
    let seq = run(&rows, 1, 0);
    assert_eq!(1.0 + 1.5 * e, 1.0 + 2.0 * e, "the block's mean rounds");
    assert_eq!(blk.c[0], (1.5 * e) * (1.5 * e));
    assert_eq!(seq.c[0], blk.c[0]);
}

/// A row of weight 0 opening a block, at `1e100` against rows at `1e7`,
/// takes no part in where the block is centred: the block is centred on
/// its first row of weight, and its moments are the per-row recursion's
/// to the data's resolution (centred on the zero row, the block's mean
/// would read 0 and its variance cancel `1e14` against `1`) (task 158).
#[test]
fn a_zero_weight_row_opening_a_block_is_not_its_centre() {
    let k = 2;
    let mut rows = stream(4, k, 31, 1e7, 1.0, 0);
    rows.push((vec![1e100, 1e100], 1.0, 0.0));
    rows.extend(stream(3, k, 37, 1e7, 1.0, 0));
    let blk = run(&rows, k, 4);
    let seq = run(&rows, k, 0);
    assert_moments_close(&seq, &blk, "a zero-weight row first in the second block");
    assert_scalars_equal(&seq, &blk, "a zero-weight row first in the second block");
}

/// A row of weight 0 whose decay takes the whole history -- `lam · W`
/// underflows to 0 while `lam` and `W` are not 0 -- is the decay alone:
/// the weight goes to 0 (task 115 (c)), and the next row starts over
/// (task 158).
#[test]
fn a_decay_that_underflows_the_weight_empties_the_accumulator() {
    let mut c = EwCov::new(1);
    c.update(&[3.0], 1.0, 0.25);
    let lam = f64::from_bits(1); // 2^-1074
    assert_eq!(lam * 0.25, 0.0);
    assert!(lam / 0.25 > 0.0);
    c.update(&[5.0], lam, 0.0);
    assert_eq!(c.n_eff(), 0.0);
    c.update(&[7.0], 1.0, 1.0);
    assert_eq!((c.n_eff(), c.mean(0)), (1.0, 7.0));
}
