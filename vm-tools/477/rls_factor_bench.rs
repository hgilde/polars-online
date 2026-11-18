//! args: features targets groups mode(never|coef) rows reps. UNCOMMITTED bench.
use online_core::*;
use std::time::Instant;
fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let k: usize = a[1].parse().unwrap();
    let t: usize = a[2].parse().unwrap();
    let g: usize = a[3].parse().unwrap();
    let coef = a[4] == "coef";
    let rows: usize = a[5].parse().unwrap();
    let reps: usize = a[6].parse().unwrap();
    let mut s = 12345u64;
    let pool = 4096usize;
    let x: Vec<f64> = (0..pool * k).map(|_| lcg(&mut s)).collect();
    let y: Vec<f64> = (0..pool * t).map(|_| lcg(&mut s)).collect();
    let mut best = f64::INFINITY;
    let mut digest = 0u64;
    for _ in 0..reps {
        let mut ms: Vec<Rls> = (0..g)
            .map(|_| {
                Rls::new(RlsCfg {
                    n_features: k,
                    n_targets: t,
                    fit_intercept: true,
                    decay: Decay::Halflife(1000.0),
                    delta: 1.0,
                    coef_prior: None,
                    min_weight: 5.0,
                })
                .unwrap()
            })
            .collect();
        let mut out = Step::default();
        let mut d = 0u64;
        let mut ys: Vec<Option<f64>> = vec![None; t];
        let t0 = Instant::now();
        for i in 0..rows {
            let gi = (i.wrapping_mul(2654435761)) % g;
            let p = i % pool;
            for (j, yy) in ys.iter_mut().enumerate() {
                *yy = Some(y[p * t + j]);
            }
            ms[gi].step_into(&x[p * k..(p + 1) * k], &ys, 1.0, 1.0, &mut out);
            for v in &out.pred {
                d = d.rotate_left(5) ^ v.to_bits();
            }
            if coef {
                for c in ms[gi].coefficients() {
                    for v in c {
                        d = d.rotate_left(5) ^ v.to_bits();
                    }
                }
            }
        }
        best = best.min(t0.elapsed().as_secs_f64());
        // state digest
        for m in &ms {
            let st = m.state();
            let bytes = rmp_serde::to_vec(&st).unwrap();
            for b in bytes {
                d = d.rotate_left(3) ^ b as u64;
            }
        }
        digest = d;
    }
    println!("min {:.1} ns/row digest {:x}", best * 1e9 / rows as f64, digest);
}
