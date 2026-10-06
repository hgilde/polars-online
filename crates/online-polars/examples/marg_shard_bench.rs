//! `marginal` stepped with its pairs split across a rayon pool
//! (`Marginal::step_sharded`, docs/PLAN.md task 126), against the unsplit
//! step: what `"auto"`'s threshold is calibrated from (docs/PERFORMANCE.md
//! §25).
//!
//! Arguments, all optional, as `key=value`: `features` (10,000), `targets`
//! (1), `rows` (2,000), `lags` as a comma list (none), `cross`, the cross
//! lags as a comma list or `none` (every lag), `bins` (0), `threads` (the
//! machine's), `shards` as a comma list (`1,2,4,8,14`), `runs` (3), `warm`,
//! the bins' warm-up rows (200), and `runner`: `pool`, `order` (every shard
//! on this thread) or `none` (no shard runs: the serial part alone). One
//! shard is the plain `step`. Each line is the best of `runs`, with a
//! checksum of every pair's numbers that must not move with the count.
use online_core::{
    BinCfg, BinRule, Decay, Marginal, MarginalCfg, MarginalShard, OnlineModel, ShardRunner, Shards,
};
use rayon::prelude::*;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn on_the_pool(shards: &mut [MarginalShard<'_>]) {
    shards.par_iter_mut().for_each(MarginalShard::run);
}

fn in_order(shards: &mut [MarginalShard<'_>]) {
    shards.iter_mut().for_each(MarginalShard::run);
}

/// Runs nothing: the rows' serial part alone, for measuring it.
fn nothing(_: &mut [MarginalShard<'_>]) {}

fn main() {
    let args: Vec<(String, String)> = std::env::args()
        .skip(1)
        .map(|a| {
            let (k, v) = a.split_once('=').expect("arguments are key=value");
            (k.to_string(), v.to_string())
        })
        .collect();
    let known = [
        "features", "targets", "rows", "lags", "cross", "bins", "threads", "shards", "runs",
        "runner", "warm",
    ];
    if let Some((k, _)) = args.iter().find(|(k, _)| !known.contains(&k.as_str())) {
        panic!("unknown argument {k:?}; expected one of {known:?}");
    }
    let get = |key: &str| args.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str());
    let num = |key: &str, default: usize| {
        get(key).map_or(default, |v| v.parse().expect("a whole number"))
    };
    let list = |v: &str| -> Vec<usize> {
        match v {
            "none" | "" => Vec::new(),
            _ => v.split(',').map(|l| l.parse().expect("a number")).collect(),
        }
    };
    let (p, t, n, n_bins, runs) = (
        num("features", 10_000),
        num("targets", 1),
        num("rows", 2_000),
        num("bins", 0),
        num("runs", 3),
    );
    let lags = get("lags").map_or_else(Vec::new, list);
    let cross_lags = get("cross").map(list);
    let counts = list(get("shards").unwrap_or("1,2,4,8,14"));
    let threads = num("threads", 0);
    // `pool` (the default), `order` (every shard on this thread), or `none`
    // (no shard runs: the serial part alone, and a checksum that differs).
    let runner: &ShardRunner<'_> = match get("runner") {
        None | Some("pool") => &on_the_pool,
        Some("order") => &in_order,
        Some("none") => &nothing,
        Some(other) => panic!("unknown runner {other:?}"),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a pool");
    let mut seed = 99u64;
    let mut level = vec![0.0; p];
    let rows: Vec<(Vec<f64>, Vec<Option<f64>>)> = (0..n)
        .map(|_| {
            for v in level.iter_mut() {
                *v = 0.9 * *v + lcg(&mut seed);
            }
            let y = (0..t)
                .map(|k| Some(level[k % p] + 0.3 * lcg(&mut seed)))
                .collect();
            (level.clone(), y)
        })
        .collect();
    let cfg = MarginalCfg {
        n_features: p,
        n_targets: t,
        decay: Decay::Halflife(500.0),
        min_weight: vec![3.0; t],
        lags: lags.clone(),
        serial_rule: None,
        cross_lags,
        bins: (n_bins > 0).then(|| {
            Box::new(BinCfg {
                n_bins,
                edges: None,
                rule: BinRule::Quantile,
                warm_rows: num("warm", 200),
                budget_mib: None,
            })
        }),
        feature_moments: online_core::FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    let mut base = None;
    for &count in &counts {
        let shards = Shards { count, run: runner };
        let mut best = f64::INFINITY;
        let mut last = None;
        for _ in 0..runs {
            let mut m = Marginal::new(cfg.clone()).expect("a valid cfg");
            let el = pool.install(|| {
                let t0 = Instant::now();
                for (i, (x, y)) in rows.iter().enumerate() {
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    if count <= 1 {
                        OnlineModel::step(&mut m, x, y, d, 1.0);
                    } else {
                        m.step_sharded(x, y, d, 1.0, &shards);
                    }
                }
                m.flush(&shards);
                t0.elapsed().as_secs_f64()
            });
            best = best.min(el);
            last = Some(m);
        }
        let m = last.expect("at least one run");
        let mut h = 0xcbf29ce484222325u64;
        for k in 0..t {
            for j in 0..p {
                let q = m.pair(k, j);
                // Every number the pair reports, the shard-written ones
                // included (review 2026-09-26, E3: `lagcorr_yx` and
                // `bin_var_y` were left out of "every pair's numbers").
                let scalars = [
                    q.n_eff,
                    q.n_kish,
                    q.mean_x,
                    q.var_x,
                    q.mean_y,
                    q.var_y,
                    q.cov,
                    q.corr,
                    q.beta,
                    q.t,
                    q.split_gain,
                    q.split_at,
                    q.split_gain_t,
                ];
                let lagged = q
                    .lagcorr_xx
                    .iter()
                    .chain(&q.lagcorr_yy)
                    .chain(&q.lagcorr_xy)
                    .chain(&q.lagcorr_yx);
                let binned = q
                    .bin_edges
                    .iter()
                    .chain(&q.bin_n)
                    .chain(&q.bin_mean_y)
                    .chain(&q.bin_var_y);
                for v in scalars.iter().chain(lagged).chain(binned) {
                    for b in v.to_bits().to_le_bytes() {
                        h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
                    }
                }
            }
        }
        let per_row = best / n as f64 * 1e6;
        let base_row = *base.get_or_insert(per_row);
        println!(
            "shards {count:>3}: {per_row:>9.2} us/row  {:>5.2}x  checksum {h:016x}",
            base_row / per_row
        );
    }
}
