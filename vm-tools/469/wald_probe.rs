use online_core::TwinFit;
use std::time::Instant;
fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
/// usage: wald_probe <intercept 0|1> <order>...   (order = coefficients, the intercept counted)
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let intercept = args[0] == "1";
    for a in &args[1..] {
        let order: usize = a.parse().unwrap();
        let kx = order - usize::from(intercept);
        let mut t = TwinFit::new(kx.max(1));
        let mut s = 7u64;
        for i in 0..(3 * order + 40) {
            let x: Vec<f64> = (0..kx.max(1)).map(|_| lcg(&mut s) + 1.0).collect();
            let y = x[0] - 0.5 * x[kx.max(1) - 1] + 0.3 * lcg(&mut s) + if i > 2 * order { x[1.min(kx.max(1) - 1)] } else { 0.0 };
            t.update(&x, y, 0.999, 1.0);
        }
        let idx: Vec<usize> = (0..kx).collect();
        let mut sink = 0.0;
        let reps = (20_000_000 / (order * order * order + 200)).clamp(20, 20000);
        for _ in 0..reps / 4 + 1 {
            sink += t.wald(&idx, intercept, 1.0).unwrap_or(0.0);
        }
        let st = Instant::now();
        for _ in 0..reps {
            sink += t.wald(&idx, intercept, 1.0).unwrap_or(0.0);
        }
        let ns = st.elapsed().as_nanos() as f64 / reps as f64;
        println!("wald order {order:3}  {ns:.1} ns/call  (sink {sink:.6e})");
    }
}
