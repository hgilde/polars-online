//! Probe for docs/PLAN.md task 225: one block's Gram product, `S += a·DᵀD`
//! on the lower triangle, by several kernels, timed and compared to the bit.
//!
//!     cargo run --release -p online-core --example gram_kernel_probe -- [k...]
//!
//! Cost is wall time per Gram entry per row, `t / (k² · n)`, the unit the
//! PLAN quotes (28-38 ps on 0.13.0 through the bank, numpy 2.65 ps).

use faer::linalg::matmul::matmul as gemm;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri};
use faer::{Accum, MatMut, MatRef, Par};
use std::hint::black_box;
use std::sync::Mutex;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn tri_faer(c: &mut [f64], d: &[f64], n: usize, k: usize, alpha: f64, par: Par) {
    let dm = MatRef::from_row_major_slice(d, n, k);
    let cm = MatMut::from_row_major_slice_mut(c, k, k);
    tri(
        cm,
        BlockStructure::TriangularLower,
        Accum::Add,
        dm.transpose(),
        BlockStructure::Rectangular,
        dm,
        BlockStructure::Rectangular,
        alpha,
        par,
    );
}

fn full_faer(c: &mut [f64], d: &[f64], n: usize, k: usize, alpha: f64) {
    let dm = MatRef::from_row_major_slice(d, n, k);
    let cm = MatMut::from_row_major_slice_mut(c, k, k);
    gemm(cm, Accum::Add, dm.transpose(), dm, alpha, Par::Seq);
}

/// A plain register-blocked SYRK: 4×8 output tiles, rows summed in order,
/// `mul` then `add` (or `mul_add` when `fused`).
fn hand(c: &mut [f64], d: &[f64], n: usize, k: usize, alpha: f64, fused: bool) {
    const MI: usize = 4;
    const NJ: usize = 8;
    let mut i0 = 0;
    while i0 < k {
        let mi = MI.min(k - i0);
        let mut j0 = 0;
        while j0 < i0 + mi && j0 < k {
            let nj = NJ.min(k - j0);
            let mut acc = [[0.0f64; NJ]; MI];
            if mi == MI && nj == NJ {
                for r in 0..n {
                    let row = &d[r * k..(r + 1) * k];
                    let a: [f64; MI] = row[i0..i0 + MI].try_into().unwrap();
                    let b: [f64; NJ] = row[j0..j0 + NJ].try_into().unwrap();
                    for ii in 0..MI {
                        for jj in 0..NJ {
                            if fused {
                                acc[ii][jj] = a[ii].mul_add(b[jj], acc[ii][jj]);
                            } else {
                                acc[ii][jj] += a[ii] * b[jj];
                            }
                        }
                    }
                }
            } else {
                for r in 0..n {
                    let row = &d[r * k..(r + 1) * k];
                    for ii in 0..mi {
                        for jj in 0..nj {
                            acc[ii][jj] += row[i0 + ii] * row[j0 + jj];
                        }
                    }
                }
            }
            for ii in 0..mi {
                for jj in 0..nj {
                    if j0 + jj <= i0 + ii {
                        c[(i0 + ii) * k + j0 + jj] += alpha * acc[ii][jj];
                    }
                }
            }
            j0 += NJ;
        }
        i0 += MI;
    }
}

/// Row bands of `band` rows: band `b` is one rectangular product for the
/// columns left of its diagonal block and one triangular product for the
/// block, both faer's, sequential. The bands are fixed by `k` and `band`
/// alone; `workers` tasks pull them, largest first, from a queue.
fn banded(c: &mut [f64], d: &[f64], n: usize, k: usize, alpha: f64, band: usize, workers: usize) {
    let dm = MatRef::from_row_major_slice(d, n, k);
    let mut queue: Vec<(usize, &mut [f64])> = c.chunks_mut(band * k).enumerate().collect();
    let one = |b: usize, rows: &mut [f64]| {
        let r0 = b * band;
        let h = rows.len() / k;
        let cm = MatMut::from_row_major_slice_mut(rows, h, k);
        let (left, rest) = cm.split_at_col_mut(r0);
        let diag = rest.subcols_mut(0, h);
        let lhs = dm.subcols(r0, h).transpose();
        if r0 > 0 {
            gemm(left, Accum::Add, lhs, dm.subcols(0, r0), alpha, Par::Seq);
        }
        tri(
            diag,
            BlockStructure::TriangularLower,
            Accum::Add,
            lhs,
            BlockStructure::Rectangular,
            dm.subcols(r0, h),
            BlockStructure::Rectangular,
            alpha,
            Par::Seq,
        );
    };
    if workers <= 1 {
        for (b, rows) in queue {
            one(b, rows);
        }
        return;
    }
    let queue = Mutex::new(std::mem::take(&mut queue));
    rayon::scope(|s| {
        for _ in 0..workers {
            s.spawn(|_| {
                loop {
                    let next = queue.lock().unwrap().pop();
                    match next {
                        Some((b, rows)) => one(b, rows),
                        None => break,
                    }
                }
            });
        }
    });
}

enum Task<'a> {
    Tri(MatMut<'a, f64>, usize),
    Rect(MatMut<'a, f64>, usize, usize),
}

/// faer's own recursion (halve, the bottom-left block rectangular, recurse
/// on the two diagonal blocks), stopped at `leaf`, with each rectangle cut
/// into tiles of `w` to `2w - 1` on a side. Fixed by `k`, `leaf` and `w`.
fn plan<'a>(dst: MatMut<'a, f64>, o: usize, leaf: usize, w: usize, out: &mut Vec<Task<'a>>) {
    let s = dst.nrows();
    if s <= leaf {
        out.push(Task::Tri(dst, o));
        return;
    }
    let h = s / 2;
    let (tl, _tr, bl, br) = dst.split_at_mut(h, h);
    tiles(bl, o + h, o, w, out);
    plan(tl, o, leaf, w, out);
    plan(br, o + h, leaf, w, out);
}

fn tiles<'a>(dst: MatMut<'a, f64>, r0: usize, c0: usize, w: usize, out: &mut Vec<Task<'a>>) {
    // As `ewcov/par.rs` cuts: `len / w` near-equal parts, none narrower
    // than `w`, so no tile is a sliver that takes another kernel.
    let (m, n) = (dst.nrows(), dst.ncols());
    let parts = |len: usize| (len / w).max(1);
    if parts(m) > 1 {
        let first = m.div_ceil(parts(m));
        let (a, b) = dst.split_at_row_mut(first);
        tiles(a, r0, c0, w, out);
        tiles(b, r0 + first, c0, w, out);
    } else if parts(n) > 1 {
        let first = n.div_ceil(parts(n));
        let (a, b) = dst.split_at_col_mut(first);
        tiles(a, r0, c0, w, out);
        tiles(b, r0, c0 + first, w, out);
    } else {
        out.push(Task::Rect(dst, r0, c0));
    }
}

fn run_task(t: Task<'_>, dm: MatRef<'_, f64>, alpha: f64) {
    match t {
        Task::Tri(dst, o) => {
            let s = dst.nrows();
            tri(
                dst,
                BlockStructure::TriangularLower,
                Accum::Add,
                dm.subcols(o, s).transpose(),
                BlockStructure::Rectangular,
                dm.subcols(o, s),
                BlockStructure::Rectangular,
                alpha,
                Par::Seq,
            )
        }
        Task::Rect(dst, r0, c0) => {
            let (m, n) = (dst.nrows(), dst.ncols());
            gemm(
                dst,
                Accum::Add,
                dm.subcols(r0, m).transpose(),
                dm.subcols(c0, n),
                alpha,
                Par::Seq,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mimic(
    c: &mut [f64],
    d: &[f64],
    n: usize,
    k: usize,
    alpha: f64,
    leaf: usize,
    w: usize,
    workers: usize,
) {
    let dm = MatRef::from_row_major_slice(d, n, k);
    let cm = MatMut::from_row_major_slice_mut(c, k, k);
    let mut tasks = Vec::new();
    plan(cm, 0, leaf, w, &mut tasks);
    if workers <= 1 {
        for t in tasks {
            run_task(t, dm, alpha);
        }
        return;
    }
    let queue = Mutex::new(tasks);
    rayon::scope(|s| {
        for _ in 0..workers {
            s.spawn(|_| {
                loop {
                    let next = queue.lock().unwrap().pop();
                    match next {
                        Some(t) => run_task(t, dm, alpha),
                        None => break,
                    }
                }
            });
        }
    });
}

fn time<F: FnMut()>(mut f: F, min_s: f64) -> f64 {
    f();
    let mut reps = 0u32;
    let t = Instant::now();
    loop {
        f();
        reps += 1;
        let e = t.elapsed().as_secs_f64();
        if e > min_s && reps >= 3 {
            return e / reps as f64;
        }
    }
}

fn bits_eq(a: &[f64], b: &[f64], k: usize) -> (usize, f64) {
    let mut diff = 0;
    let mut worst = 0.0f64;
    for i in 0..k {
        for j in 0..=i {
            let (x, y) = (a[i * k + j], b[i * k + j]);
            if x.to_bits() != y.to_bits() {
                diff += 1;
                worst = worst.max(((x - y) / y.abs().max(1e-300)).abs());
            }
        }
    }
    (diff, worst)
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let ks = if args.is_empty() {
        vec![1000, 2000, 4000]
    } else {
        args
    };
    let n = 256usize;
    let alpha = 1.0 / 3.0;
    for &k in &ks {
        let mut s = 11u64 + k as u64;
        let d: Vec<f64> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let c0: Vec<f64> = (0..k * k).map(|_| lcg(&mut s)).collect();
        let er = (k * k * n) as f64;
        let ps = |t: f64| t / er * 1e12;
        let min_s = 1.0;

        let mut want = c0.clone();
        tri_faer(&mut want, &d, n, k, alpha, Par::Seq);

        let mut c = c0.clone();
        let t_tri = time(
            || {
                c.copy_from_slice(&c0);
                tri_faer(&mut c, &d, n, k, alpha, Par::Seq)
            },
            min_s,
        );
        let mut c = c0.clone();
        let t_copy = time(
            || {
                c.copy_from_slice(&c0);
                black_box(&c);
            },
            min_s,
        );
        println!("k={k} n={n}  (copy of C {:.2} ps, subtracted)", ps(t_copy));
        println!(
            "  faer triangular, seq          {:7.2} ps",
            ps(t_tri - t_copy)
        );
        let t_full = time(
            || {
                c.copy_from_slice(&c0);
                full_faer(&mut c, &d, n, k, alpha)
            },
            min_s,
        );
        println!(
            "  faer general matmul, seq      {:7.2} ps",
            ps(t_full - t_copy)
        );
        for fused in [false, true] {
            let t_h = time(
                || {
                    c.copy_from_slice(&c0);
                    hand(&mut c, &d, n, k, alpha, fused)
                },
                min_s,
            );
            let mut h = c0.clone();
            hand(&mut h, &d, n, k, alpha, fused);
            let (nd, worst) = bits_eq(&h, &want, k);
            println!(
                "  hand 4x8 SYRK {}       {:7.2} ps   bits off {nd} (worst rel {worst:.1e})",
                if fused { "mul_add " } else { "mul+add " },
                ps(t_h - t_copy)
            );
        }
        for par in [2usize, 4, 8, 14] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(par)
                .build()
                .unwrap();
            let t_p = pool.install(|| {
                time(
                    || {
                        c.copy_from_slice(&c0);
                        tri_faer(&mut c, &d, n, k, alpha, Par::rayon(par))
                    },
                    min_s,
                )
            });
            let mut p = c0.clone();
            pool.install(|| tri_faer(&mut p, &d, n, k, alpha, Par::rayon(par)));
            let (nd, worst) = bits_eq(&p, &want, k);
            println!(
                "  faer triangular, Par::rayon({par:>2}) {:7.2} ps  bits off {nd} (worst {worst:.1e})",
                ps(t_p - t_copy)
            );
        }
        for (leaf, w) in [
            (64usize, 64usize),
            (128, 128),
            (256, 256),
            (256, 128),
            (512, 256),
        ] {
            let mut b = c0.clone();
            mimic(&mut b, &d, n, k, alpha, leaf, w, 1);
            let (nd, worst) = bits_eq(&b, &want, k);
            let t_b = time(
                || {
                    c.copy_from_slice(&c0);
                    mimic(&mut c, &d, n, k, alpha, leaf, w, 1)
                },
                min_s,
            );
            println!(
                "  mimic leaf {leaf:>3} tile {w:>3},  1 thread  {:7.2} ps   bits vs faer seq off {nd} (worst {worst:.1e})",
                ps(t_b - t_copy)
            );
            for wk in [2usize, 4, 8, 14] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(wk)
                    .build()
                    .unwrap();
                let t_b = pool.install(|| {
                    time(
                        || {
                            c.copy_from_slice(&c0);
                            mimic(&mut c, &d, n, k, alpha, leaf, w, wk)
                        },
                        min_s,
                    )
                });
                let mut bw = c0.clone();
                pool.install(|| mimic(&mut bw, &d, n, k, alpha, leaf, w, wk));
                let (nd1, _) = bits_eq(&bw, &want, k);
                println!(
                    "  mimic leaf {leaf:>3} tile {w:>3}, {wk:>2} threads {:7.2} ps   bits vs faer seq off {nd1}",
                    ps(t_b - t_copy)
                );
            }
        }
        if std::env::var_os("BANDED").is_none() {
            continue;
        }
        for band in [64usize, 128, 256] {
            let mut b = c0.clone();
            banded(&mut b, &d, n, k, alpha, band, 1);
            let (nd, worst) = bits_eq(&b, &want, k);
            let t_b = time(
                || {
                    c.copy_from_slice(&c0);
                    banded(&mut c, &d, n, k, alpha, band, 1)
                },
                min_s,
            );
            println!(
                "  banded {band:>3}, 1 thread         {:7.2} ps   bits off {nd} (worst {worst:.1e})",
                ps(t_b - t_copy)
            );
            for w in [2usize, 4, 8, 14] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(w)
                    .build()
                    .unwrap();
                let t_b = pool.install(|| {
                    time(
                        || {
                            c.copy_from_slice(&c0);
                            banded(&mut c, &d, n, k, alpha, band, w)
                        },
                        min_s,
                    )
                });
                let mut bw = c0.clone();
                pool.install(|| banded(&mut bw, &d, n, k, alpha, band, w));
                let (nd1, _) = bits_eq(&bw, &b, k);
                println!(
                    "  banded {band:>3}, {w:>2} threads        {:7.2} ps   bits vs 1 thread off {nd1}",
                    ps(t_b - t_copy)
                );
            }
        }
    }
}
