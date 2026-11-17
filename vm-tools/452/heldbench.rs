//! Native timing of model cases' measured blocks: HB_REPS fresh setups each, the fastest kept; prints the checksum's bits.
use online_bench::*;
use std::time::Instant;

fn main() {
    let reps: usize = std::env::var("HB_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    for id in std::env::args().skip(1) {
        let mut best = f64::MAX;
        let mut sum = 0.0;
        for _ in 0..reps {
            let mut run = model_setup(&id);
            let t = Instant::now();
            let b = polars_online_budget_model_rows(&mut run);
            let dt = t.elapsed().as_secs_f64();
            sum = b.checksum;
            best = best.min(dt);
        }
        println!("{id} {:.1} us {:016x}", best * 1e6, sum.to_bits());
    }
}
