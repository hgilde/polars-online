use online_core::TwinFit;
use std::time::Instant;
fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
fn main() {
    let orders: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    for k in orders {
        let mut t = TwinFit::new(k);
        let mut s = 7u64;
        for i in 0..(3 * k + 40) {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s) + 1.0).collect();
            let y = x[0] - 0.5 * x[k - 1] + 0.3 * lcg(&mut s) + if i > 2 * k { x[1.min(k - 1)] } else { 0.0 };
            t.update(&x, y, 0.999, 1.0);
        }
        let idx: Vec<usize> = (0..k).collect();
        let mut sink = 0.0;
        let reps = (20_000_000 / (k * k * k + 200)).clamp(20, 20000);
        for _ in 0..reps / 4 + 1 {
            sink += t.wald(&idx, false, 1.0).unwrap_or(0.0);
        }
        let st = Instant::now();
        for _ in 0..reps {
            sink += t.wald(&idx, false, 1.0).unwrap_or(0.0);
        }
        let ns = st.elapsed().as_nanos() as f64 / reps as f64;
        println!("wald order {k:3}  {ns:.1} ns/call  (sink {sink:.3e})");
    }
}
