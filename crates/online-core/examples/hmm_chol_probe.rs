//! Task 456's probe: `hmm`'s rows and faer's Cholesky at hmm's orders, timed
//! and digested, to be built twice (baseline x86-64 and `-C
//! target-cpu=x86-64-v3`) and compared.
//!
//!     cargo run --release -p online-core --example hmm_chol_probe -- [reps] [only]
//!
//! Prints one line per case: the best of `reps` timings (default 7) in ns per
//! call, and an FNV digest of every output's bits, which must be the same in
//! every build. `only` (e.g. `hmm` or `spd n=1`) keeps the cases whose name
//! starts with it. The orders straddle both switches of `solve.rs`'s small
//! paths: the factor's at 12 and 13, the solve's at 4 and 5.
//!
//! * `hmm d=4 k=2`: the bank's `hmm` case through the model alone.
//! * `spd n=N`: `SpdFactor::of` of an order-`N` matrix, the call `hmm` makes
//!   per state per row, and its `log_det` read.
//! * `llt n=N`: faer's `llt::factor::cholesky_in_place` alone on a copy.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::prelude::*;
use online_core::{Covariance, Decay, Hmm, HmmCfg, OnlineModel, SeedRule, SpdFactor};
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn fnv(h: &mut u64, v: f64) {
    for b in v.to_bits().to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn spd(k: usize) -> Vec<f64> {
    let mut s = 3u64;
    let n = k + 7;
    let x: Vec<f64> = (0..n * k).map(|_| 0.5 + lcg(&mut s)).collect();
    let xm = MatRef::from_row_major_slice(&x, n, k);
    let g = xm.transpose() * xm;
    (0..k * k)
        .map(|ij| g[(ij / k, ij % k)] / n as f64 + if ij / k == ij % k { 1e-3 } else { 0.0 })
        .collect()
}

fn best(reps: usize, mut f: impl FnMut() -> f64) -> (f64, u64) {
    let mut t_best = f64::INFINITY;
    let mut h = 0xcbf29ce484222325u64;
    for _ in 0..reps {
        let t = Instant::now();
        let v = f();
        let dt = t.elapsed().as_secs_f64();
        t_best = t_best.min(dt);
        fnv(&mut h, v);
    }
    (t_best, h)
}

fn hmm_case(reps: usize) {
    const N: usize = 200_000;
    let d = 4;
    let mut s = 42u64;
    let xs: Vec<Vec<f64>> = (0..N)
        .map(|i| {
            let blob = (i / 50) % 2;
            (0..d)
                .map(|_| lcg(&mut s) + if blob == 1 { 3.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let cfg = HmmCfg {
        n_features: d,
        k: 2,
        decay: Decay::Halflife(1000.0),
        covariance: Covariance::Full,
        precision_prior: 1.0,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 20,
        seed_rule: SeedRule::First,
        seed: 7,
        tvtp: None,
    };
    let (t, h) = best(reps, || {
        let mut m = Hmm::new(cfg.clone()).unwrap();
        let mut acc = 0.0;
        let t0 = Instant::now();
        for (i, x) in xs.iter().enumerate() {
            let st = m.step(x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            acc += st.n_eff + st.pred.iter().filter(|v| v.is_finite()).sum::<f64>();
            black_box(&st.pred);
        }
        black_box(t0);
        acc
    });
    println!(
        "{:<14} {:>10.1} ns/row   digest {h:016x}",
        "hmm d=4 k=2",
        t * 1e9 / N as f64
    );
}

fn spd_case(n: usize, reps: usize) {
    const CALLS: usize = 200_000;
    let a = spd(n);
    let (t, h) = best(reps, || {
        let mut acc = 0.0;
        for _ in 0..CALLS {
            let f = SpdFactor::of(black_box(&a), n).unwrap();
            acc += f.log_det();
            black_box(&f);
        }
        acc
    });
    println!(
        "{:<14} {:>10.1} ns/call  digest {h:016x}",
        format!("spd n={n}"),
        t * 1e9 / CALLS as f64
    );
}

fn llt_case(n: usize, reps: usize) {
    const CALLS: usize = 200_000;
    let a = spd(n);
    let base = Mat::from_fn(n, n, |i, j| if i < j { 0.0 } else { a[i * n + j] });
    let mut l = base.clone();
    let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq,
        Default::default(),
    ));
    let (t, h) = best(reps, || {
        let mut acc = 0.0;
        for _ in 0..CALLS {
            l.as_mut().copy_from(base.as_ref());
            llt::factor::cholesky_in_place(
                l.as_mut(),
                Default::default(),
                Par::Seq,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .unwrap();
            acc += l[(n - 1, n - 1)];
        }
        black_box(&l);
        acc
    });
    println!(
        "{:<14} {:>10.1} ns/call  digest {h:016x}",
        format!("llt n={n}"),
        t * 1e9 / CALLS as f64
    );
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(7);
    println!(
        "target-feature avx2: {}  fma: {}  (compile time)",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "fma")
    );
    let only = std::env::args().nth(2).unwrap_or_default();
    let wanted = |name: &str| name.starts_with(&only);
    if wanted("hmm") {
        hmm_case(reps);
    }
    for n in [2, 3, 4, 5, 6, 8, 12, 13, 16, 20, 32, 48, 64, 65, 96] {
        if wanted(&format!("spd n={n}")) {
            spd_case(n, reps);
        }
    }
    for n in [2, 4, 5, 8, 12, 13, 16, 32, 64] {
        if wanted(&format!("llt n={n}")) {
            llt_case(n, reps);
        }
    }
}
