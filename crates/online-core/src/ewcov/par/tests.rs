//! `gram_threads` (docs/PLAN.md task 225): the pieces cover the triangle
//! once, they are the single product to the bit where faer's sequential
//! product is the same calls, and every thread count gives the same bits.

use super::*;
use crate::EwCov;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A merge's inputs: `C` symmetric, `D` (`n × k`), and the per-row
/// corrections, all generated.
struct Case {
    k: usize,
    n: usize,
    c: Vec<f64>,
    d: Vec<f64>,
    bi: Vec<f64>,
    wi: Vec<f64>,
    big: Vec<f64>,
    delta: Vec<f64>,
}

impl Case {
    fn new(k: usize, n: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..=i {
                let v = lcg(&mut s);
                c[i * k + j] = v;
                c[j * k + i] = v;
            }
        }
        let mut v = |len: usize| -> Vec<f64> { (0..len).map(|_| lcg(&mut s)).collect() };
        Self {
            k,
            n,
            d: v(n * k),
            bi: v(k),
            wi: v(k),
            big: v(k),
            delta: v(k),
            c,
        }
    }

    fn merge(&self) -> Merge<'_> {
        Merge {
            scale: 0.75,
            alpha: 1.0 / 3.0,
            bi: &self.bi,
            wi: &self.wi,
            big: &self.big,
            delta: &self.delta,
        }
    }

    /// `merge_lower_cut` on a copy of `C`.
    fn cut(&self, threads: usize, cut: (usize, usize)) -> Vec<f64> {
        let mut c = self.c.clone();
        merge_lower_cut(&mut c, self.k, &self.d, self.n, &self.merge(), threads, cut);
        c
    }

    /// The merge as `EwCov::flush` made it before task 225: four passes
    /// over `C`, the product one faer call on the whole triangle.
    #[cfg_attr(target_arch = "x86_64", allow(dead_code))]
    fn whole(&self) -> Vec<f64> {
        let (k, m) = (self.k, self.merge());
        let mut c = self.c.clone();
        for i in 0..k {
            for j in 0..=i {
                c[i * k + j] *= m.scale;
            }
        }
        let d = MatRef::from_row_major_slice(&self.d, self.n, k);
        tri(
            MatMut::from_row_major_slice_mut(&mut c, k, k),
            BlockStructure::TriangularLower,
            Accum::Add,
            d.transpose(),
            BlockStructure::Rectangular,
            d,
            BlockStructure::Rectangular,
            m.alpha,
            Par::Seq,
        );
        for i in 0..k {
            for j in 0..=i {
                c[i * k + j] += m.bi[i] * m.big[j] - m.wi[i] * m.delta[j];
            }
        }
        for i in 0..k {
            for j in 0..i {
                c[j * k + i] = c[i * k + j];
            }
        }
        c
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("a test pool")
}

/// A fine cut, so that a matrix a debug build multiplies quickly is still
/// many pieces of both kinds.
const FINE: (usize, usize) = (16, 8);

#[test]
fn every_entry_is_in_one_piece_and_mirrored_once() {
    for (k, cut) in [
        (1, FINE),
        (16, FINE),
        (17, FINE),
        (100, FINE),
        (257, (LEAF, TILE)),
        (1000, (LEAF, TILE)),
    ] {
        let mut hits = vec![0u32; k * k];
        let mut owner = vec![0.0f64; k * k];
        let found = pieces(MatMut::from_row_major_slice_mut(&mut owner, k, k), cut);
        let n_pieces = found.len();
        for p in found {
            // Each piece marks its lower entries by their own index, and its
            // mirror tile by the same, so a mark that lands in the wrong
            // place shows.
            match p {
                Piece::Diag(blk, o) => {
                    for i in 0..blk.nrows() {
                        for j in 0..blk.nrows() {
                            hits[(o + i) * k + o + j] += 1;
                        }
                    }
                }
                Piece::Tile(low, up, r0, c0) => {
                    assert_eq!((low.nrows(), low.ncols()), (up.ncols(), up.nrows()));
                    for i in 0..low.nrows() {
                        for j in 0..low.ncols() {
                            assert!(r0 + i > c0 + j, "a tile entry above the diagonal");
                            hits[(r0 + i) * k + c0 + j] += 1;
                            hits[(c0 + j) * k + r0 + i] += 1;
                        }
                    }
                }
            }
        }
        assert!(
            hits.iter().all(|&h| h == 1),
            "k = {k}: an entry covered other than once"
        );
        // The cut is live: past one leaf there is more than one piece.
        assert_eq!(n_pieces == 1, k <= cut.0, "k = {k}: {n_pieces} pieces");
    }
}

/// faer's sequential triangular product, on this architecture, is the
/// calls the pieces make -- its own diagonal blocks, and its rectangles
/// cut into tiles no narrower than 65, which its kernel sums as it sums
/// the whole rectangle -- so the merge is the four passes' to the bit, at
/// the shipped cut and a finer one, with a block longer than the kernel's
/// 512-row panel among them. Not on x86-64, where faer hands the whole
/// triangle to one kernel (`private_gemm_x86`) whose blocking the pieces do
/// not reproduce; the thread-count test below holds there too.
///
/// A cut into slivers parted from it (a one-row tile is a matrix-vector
/// product; a draft that cut them parted at `k = 17`), which is why `tiles` never cuts
/// one.
#[cfg(not(target_arch = "x86_64"))]
#[test]
fn the_pieces_are_the_single_product_to_the_bit() {
    const MID: (usize, usize) = (128, 65);
    for (k, n, cut) in [
        (1, 3, (LEAF, TILE)),
        (2, 1, (LEAF, TILE)),
        (256, 2, (LEAF, TILE)),
        (257, 3, (LEAF, TILE)),
        (300, 1, (LEAF, TILE)),
        (300, 600, (LEAF, TILE)),
        (520, 4, (LEAF, TILE)),
        (129, 5, MID),
        (200, 2, MID),
        (263, 7, MID),
    ] {
        let case = Case::new(k, n, 3 + k as u64);
        assert_eq!(
            bits(&case.cut(1, cut)),
            bits(&case.whole()),
            "k = {k}, n = {n}"
        );
    }
}

#[test]
fn every_thread_count_gives_the_same_bits() {
    let pool = pool(8);
    for (k, n, cut) in [(100, 7, FINE), (61, 2, FINE), (300, 3, (LEAF, TILE))] {
        let case = Case::new(k, n, 11 + k as u64);
        let one = bits(&case.cut(1, cut));
        for threads in [2, 3, 8, 64] {
            let many = pool.install(|| case.cut(threads, cut));
            assert_eq!(bits(&many), one, "k = {k}, {threads} threads");
        }
    }
}

/// The work does reach other threads: the test above would pass on one.
#[test]
fn spread_runs_on_several_threads_and_on_one_when_asked() {
    let pool = pool(4);
    let ids = |threads: usize| {
        let seen = Mutex::new(std::collections::HashSet::new());
        pool.install(|| {
            spread((0..32).collect::<Vec<_>>(), threads, |_| {
                std::thread::sleep(std::time::Duration::from_millis(2));
                seen.lock().unwrap().insert(std::thread::current().id());
            })
        });
        seen.into_inner().unwrap()
    };
    assert!(ids(4).len() >= 2, "four threads asked, one used");
    assert_eq!(ids(1).len(), 1);
    // In order on one thread: each item exactly once.
    let order = Mutex::new(Vec::new());
    spread((0..9).collect::<Vec<_>>(), 1, |i| {
        order.lock().unwrap().push(i)
    });
    assert_eq!(order.into_inner().unwrap(), (0..9).collect::<Vec<_>>());
    let all = Mutex::new(Vec::new());
    pool.install(|| {
        spread((0..9).collect::<Vec<_>>(), 4, |i| {
            all.lock().unwrap().push(i)
        })
    });
    let mut all = all.into_inner().unwrap();
    all.sort_unstable();
    assert_eq!(all, (0..9).collect::<Vec<_>>());
}

/// The accumulator's moments over a stream -- a zero-weight first row
/// (hard rule 9), zero weights inside, a decay that takes the whole history
/// (`lam = 0`) -- blocked and per row, at several thread counts, wide
/// enough that the merge is many pieces and the per-row update is split.
/// `lam` is a literal, so no platform's libm is on the stream.
fn stream(k: usize, block: usize, threads: usize) -> EwCov {
    let mut s = 5u64;
    let mut c = EwCov::new(k);
    c.set_block_rows(block);
    c.set_threads(threads);
    for r in 0..40 {
        let x: Vec<f64> = (0..k).map(|_| 1e3 + lcg(&mut s)).collect();
        let w = match r {
            0 | 9 => 0.0,
            _ => 0.5 + lcg(&mut s).abs(),
        };
        let lam = if r == 20 { 0.0 } else { 0.97 };
        c.update(&x, lam, w);
    }
    c
}

fn moment_bits(c: &EwCov) -> Vec<u64> {
    let mut out = bits(c.comoments());
    out.extend(bits(c.means()));
    out.push(c.n_eff().to_bits());
    out.push(c.n_kish().unwrap_or(f64::NAN).to_bits());
    out
}

#[test]
fn an_accumulator_has_the_same_bits_at_every_thread_count() {
    let pool = pool(8);
    // Blocked: a merge of many pieces; the last block is read through
    // `flushed`, as the bank's `gram()` reads it.
    let k = LEAF + 44;
    let one = moment_bits(&stream(k, 7, 1).flushed());
    for threads in [2, 8] {
        let many = pool.install(|| stream(k, 7, threads));
        assert_eq!(many.threads(), threads);
        assert_eq!(
            moment_bits(&pool.install(|| many.flushed().into_owned())),
            one,
            "blocked, {threads} threads"
        );
    }
    // Per row: past the floor below which a row stays on one thread.
    let k = PAR_MIN_ENTRIES.isqrt() + 3;
    assert!(k * k >= PAR_MIN_ENTRIES);
    let one = moment_bits(&stream(k, 0, 1));
    for threads in [2, 8] {
        let many = pool.install(|| stream(k, 0, threads));
        assert_eq!(moment_bits(&many), one, "per row, {threads} threads");
    }
}

/// Threads are configuration: a state carries none, and a restored
/// accumulator runs on one until its owner sets it again.
#[test]
fn a_restored_accumulator_runs_on_one_thread() {
    let mut c = EwCov::new(3);
    c.set_threads(8);
    c.update(&[1.0, 2.0, 3.0], 0.9, 1.0);
    let bytes = rmp_serde::to_vec(&c).unwrap();
    let back: EwCov = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(back.threads(), 1);
    // ... and is the same accumulator: equality ignores the count.
    assert_eq!(back, c);
    let mut plain = EwCov::new(3);
    plain.update(&[1.0, 2.0, 3.0], 0.9, 1.0);
    assert_eq!(rmp_serde::to_vec(&plain).unwrap(), bytes);
    // `0` is taken as one.
    c.set_threads(0);
    assert_eq!(c.threads(), 1);
}
