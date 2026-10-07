//! `marginal` over a fixed stream, by default with none of its optional
//! parts.
//!
//! This exists to answer one question: does a stream that asks for no lags,
//! no bins and no window pay for the fact that they exist? Run it against a
//! build from before the feature under suspicion, in a separate worktree and
//! `CARGO_TARGET_DIR`, best of the runs each prints (docs/PERFORMANCE.md
//! §17). It found two real costs that way -- a call inside `learn`'s loop
//! and an `O(p)` scan outside its guard -- neither of which any test could
//! have seen.
//!
//! Arguments, all optional, as `key=value`: `targets` (1), `bins` learned
//! per feature (0, none), `features` (8), `absent`, the percentage of rows
//! each target but the first is absent on (0), `lags` as a comma list (none),
//! `cross`, the cross lags as a comma list or `none` (every lag), `rows`
//! (1,000,000), `runs` (25) and `moments`, `per_target` or `shared` (the
//! feature moments' layout, docs/PLAN.md task 125). `-- targets=9 bins=16` is E71's shape (docs/PLAN.md task 122), and
//! `-- targets=9 lags=1,2,5,10,20,50 cross=1` E70's (task 123). The last
//! line is a checksum of every pair's numbers, so two builds that should
//! agree to the bit can be seen to.
use online_core::*;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn main() {
    let args: Vec<(String, String)> = std::env::args()
        .skip(1)
        .map(|a| {
            let (k, v) = a.split_once('=').expect("arguments are key=value");
            (k.to_string(), v.to_string())
        })
        .collect();
    let known = [
        "targets", "bins", "features", "absent", "lags", "cross", "rows", "runs", "moments",
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
            _ => v.split(',').map(|l| l.parse().expect("a lag")).collect(),
        }
    };
    let (n_targets, n_bins, k, absent_pct, n_rows, runs) = (
        num("targets", 1),
        num("bins", 0),
        num("features", 8),
        num("absent", 0),
        num("rows", 1_000_000),
        num("runs", 25),
    );
    let lags = get("lags").map_or_else(Vec::new, list);
    let cross_lags = get("cross").map(list);
    let mut seed = 12345u64;
    let rows: Vec<(Vec<f64>, Vec<Option<f64>>)> = (0..n_rows)
        .map(|_| {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut seed)).collect();
            let y = (0..n_targets)
                .map(|t| {
                    let v = x[t % k] + 0.3 * lcg(&mut seed);
                    let absent = t > 0 && (lcg(&mut seed) + 1.0) * 50.0 < absent_pct as f64;
                    (!absent).then_some(v)
                })
                .collect();
            (x, y)
        })
        .collect();
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..runs {
        let mut m = Marginal::new(MarginalCfg {
            n_features: k,
            n_targets,
            decay: Decay::Halflife(500.0),
            min_weight: vec![3.0; n_targets],
            lags: lags.clone(),
            serial_rule: None,
            cross_lags: cross_lags.clone(),
            bins: (n_bins > 0).then(|| {
                Box::new(BinCfg {
                    n_bins,
                    edges: None,
                    rule: BinRule::Quantile,
                    warm_rows: 1_000,
                    budget_mib: None,
                })
            }),
            feature_moments: match get("moments") {
                None | Some("per_target") => FeatureMomentLayout::PerTarget,
                Some("shared") => FeatureMomentLayout::Shared,
                Some(other) => panic!("moments is per_target or shared, got {other:?}"),
            },
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let t0 = Instant::now();
        for (i, (x, y)) in rows.iter().enumerate() {
            OnlineModel::step(&mut m, x, y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let el = t0.elapsed().as_secs_f64();
        best = best.min(el);
        std::hint::black_box(m.pair(0, 0));
        last = Some(m);
    }
    println!("{:.1} rows/s ({:.1} ms)", n_rows as f64 / best, best * 1e3);
    // FNV-1a over the bits of every number every pair reports.
    let m = last.expect("at least one run");
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |v: f64| {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    };
    for t in 0..n_targets {
        for j in 0..k {
            let q = m.pair(t, j);
            for v in [q.n_eff, q.mean_x, q.mean_y, q.var_x, q.cov, q.corr] {
                eat(v);
            }
            for v in q.lag_corr_xx.iter().chain(&q.lag_corr_yy) {
                eat(*v);
            }
            for v in q.lag_corr_xy.iter().chain(&q.lag_corr_yx) {
                eat(*v);
            }
            for v in q.bin_edges.iter().chain(&q.bin_n).chain(&q.bin_mean_y) {
                eat(*v);
            }
            for v in q
                .bin_var_y
                .iter()
                .chain(&[q.split_gain, q.split_at, q.split_gain_t])
            {
                eat(*v);
            }
        }
    }
    println!("checksum {h:016x}");
}
