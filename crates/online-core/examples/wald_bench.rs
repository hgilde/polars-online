//! Scratch (task 426b, never committed): TwinFit::update + wald per row, ns, and the wald values to a file.
//!   cargo run --release -p online-core --example wald_bench -- OUTFILE [k...]
use online_core::TwinFit;
use std::io::Write;
use std::time::Instant;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = &args[1];
    let ks: Vec<usize> = if args.len() > 2 {
        args[2..].iter().map(|a| a.parse().unwrap()).collect()
    } else {
        vec![4, 20, 60, 200]
    };
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let mut file = std::fs::File::create(out).unwrap();
    for &k in &ks {
        for &intercept in &[true, false] {
            let n = (400_000 / (k * k).max(16))
                .clamp(300, 20_000)
                .max(3 * k + 60);
            let mut best_u = f64::MAX;
            let mut best_w = f64::MAX;
            let mut vals: Vec<f64> = Vec::new();
            for rep in 0..reps {
                let mut s = 7u64;
                let mut t = TwinFit::new(k);
                let idx: Vec<usize> = (0..k).collect();
                let beta: Vec<f64> = (0..k).map(|j| 1.0 / (1.0 + j as f64)).collect();
                let lam = 0.5f64.powf(1.0 / (5.0 * k as f64).max(20.0));
                let mut x = vec![0.0; k];
                let (mut tu, mut tw) = (0.0, 0.0);
                vals.clear();
                for r in 0..n {
                    for v in x.iter_mut() {
                        *v = 3.0 + lcg(&mut s);
                    }
                    let mut y = 0.7;
                    for j in 0..k {
                        y += beta[j] * x[j];
                    }
                    y += 0.3 * lcg(&mut s) + if r > n / 2 { 0.4 * x[0] } else { 0.0 };
                    let t0 = Instant::now();
                    t.update(&x, y, lam, 1.0);
                    let t1 = Instant::now();
                    let w = t.wald(&idx, intercept, 1.0);
                    let t2 = Instant::now();
                    if r >= 3 * k {
                        tu += (t1 - t0).as_secs_f64();
                        tw += (t2 - t1).as_secs_f64();
                    }
                    vals.push(w.unwrap_or(f64::NAN));
                }
                let m = (n - 3 * k) as f64;
                best_u = best_u.min(tu / m);
                best_w = best_w.min(tw / m);
                let _ = rep;
            }
            for v in &vals {
                file.write_all(&v.to_bits().to_le_bytes()).unwrap();
            }
            println!(
                "k={k:4} icpt={intercept:5} rows={n:6}  update {:9.0} ns  wald {:9.0} ns",
                best_u * 1e9,
                best_w * 1e9
            );
        }
    }
}
