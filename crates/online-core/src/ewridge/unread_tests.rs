//! A due solve no row can read is not run (`unread`'s module docs;
//! docs/PLAN.md task 311): every number a reader of the fit is served
//! -- the predictions, the residual variance, the reads below `min_weight`
//! at every floor a bank uses, the error inflation, the row's leverage,
//! the condition's flag, the weights and the Grams -- to the bit against
//! the same model running every due solve, over streams built to reach
//! every path the argument names; and the solves it passes counted.

use super::unread::{SKIP_UNREAD, SOLVED, UNREAD};
use super::*;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn base(k: usize, m: usize) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(30.0),
        ridge: vec![1e-6],
        feature_sets: vec![],
        standardize: true,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: (k + 1) as f64,
        solve_every: 10.0,
        max_rows_between_solves: u32::MAX,
        solve_share: Some(crate::DEFAULT_SOLVE_SHARE),
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

struct Row {
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    d: f64,
    w: f64,
}

/// `n` rows over `k` features and `m` targets: weights uneven, some tiny
/// and one row in nine of weight 0 (the first row among them); a clock gap
/// of six half-lives every 150 rows and one of forty at row 400, each
/// taking the weight back under every floor so the warm-up repeats; the
/// second target null on its first 40 rows and one row in five, the third
/// on one row in three; a level on the first feature.
fn rows(n: usize, k: usize, m: usize, seed: u64) -> Vec<Row> {
    let mut s = seed;
    let beta: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
    (0..n)
        .map(|i| {
            let x: Vec<f64> = (0..k)
                .map(|j| lcg(&mut s) + if j == 0 { 5.0 } else { 0.0 })
                .collect();
            let fit: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum();
            let noise = 0.3 * lcg(&mut s);
            let y = (0..m)
                .map(|t| {
                    let present = match t {
                        0 => true,
                        1 => i >= 40 && i % 5 != 2,
                        _ => i % 3 != 1,
                    };
                    present.then_some(fit * (1.0 + t as f64) - t as f64 + noise)
                })
                .collect();
            let d = match i {
                0 => 0.0,
                400 => 1_200.0,
                _ if i % 150 == 75 => 180.0,
                _ => 1.0,
            };
            let w = match i % 9 {
                0 => 0.0,
                4 => 0.01,
                7 => 5.0,
                _ => 1.0 + 0.5 * lcg(&mut s),
            };
            Row { x, y, d, w }
        })
        .collect()
}

/// One configuration of the model and what the stream does around it.
struct Case {
    name: &'static str,
    cfg: EwRidgeCfg,
    /// The explicit cadence's first period on the default schedule.
    first_period: bool,
    output_only: bool,
    targets: Vec<f64>,
    keep_factor: bool,
    /// A session's blend toward the slow twin every this many rows.
    blend_every: Option<usize>,
    /// Whether the case passes any solve, as the argument says it must or
    /// must not.
    skips: bool,
}

fn case(name: &'static str, cfg: EwRidgeCfg) -> Case {
    Case {
        name,
        cfg,
        first_period: false,
        output_only: true,
        targets: vec![],
        keep_factor: false,
        blend_every: None,
        skips: true,
    }
}

fn cases() -> Vec<Case> {
    let mut out = vec![case("default share", base(6, 2))];

    let mut c = case("per-target floors, low min_weight", base(6, 3));
    c.cfg.min_weight = 3.0;
    c.cfg.solve_share = Some(0.3);
    c.targets = vec![3.0, 9.0, 5.0];
    out.push(c);

    let mut c = case("min_weight over k, held back (schema 63)", base(6, 2));
    c.cfg.min_weight = 20.0;
    c.cfg.solve_share = Some(0.5);
    c.output_only = false;
    out.push(c);

    let mut c = case("min_weight over k, output only", base(6, 2));
    c.cfg.min_weight = 20.0;
    out.push(c);

    let mut c = case("first period, clock", base(6, 2));
    c.cfg.solve_share = None;
    c.cfg.solve_every = 60.0;
    c.first_period = true;
    out.push(c);

    let mut c = case("first period, row cap", base(6, 2));
    c.cfg.solve_share = None;
    c.cfg.solve_every = 1e17;
    c.cfg.max_rows_between_solves = 40;
    c.first_period = true;
    out.push(c);

    // A row cap's solve runs, read or not: at a cap of 1 every one.
    let mut c = case("share with a row cap of 1", base(6, 2));
    c.cfg.max_rows_between_solves = 1;
    c.skips = false;
    out.push(c);
    let mut c = case("share with a row cap of 3", base(6, 2));
    c.cfg.max_rows_between_solves = 3;
    out.push(c);

    let mut c = case("blocked Gram", base(6, 2));
    c.cfg.gram_block_rows = 8;
    c.cfg.standardize = false;
    out.push(c);

    let mut c = case("blocked Gram, pairwise, first period", base(6, 2));
    c.cfg.gram_block_rows = 5;
    c.cfg.solve_share = None;
    c.cfg.solve_every = 90.0;
    c.cfg.target_gaps = TargetGaps::Pairwise;
    c.first_period = true;
    out.push(c);

    let mut c = case("session blend", base(6, 2));
    c.cfg.session_shrink = Some(0.5);
    c.cfg.long_half_life = Some(200.0);
    c.blend_every = Some(90);
    out.push(c);

    let mut c = case("kept factors, ridge 0, no intercept", base(6, 2));
    c.cfg.fit_intercept = false;
    c.cfg.ridge = vec![0.0];
    c.cfg.min_weight = 6.0;
    c.keep_factor = true;
    out.push(c);

    let mut c = case("ridge_scale, pairwise", base(6, 2));
    c.cfg.ridge_scale = true;
    c.cfg.standardize = false;
    c.cfg.ridge = vec![0.5];
    c.cfg.target_gaps = TargetGaps::Pairwise;
    out.push(c);

    let mut c = case("grid and feature sets", base(6, 2));
    c.cfg.ridge = vec![1e-6, 0.1];
    c.cfg.feature_sets = vec![("a".into(), vec![0, 1, 2]), ("b".into(), vec![3, 4, 5])];
    c.keep_factor = true;
    out.push(c);

    let mut c = case("standardized through the origin", base(6, 2));
    c.cfg.fit_intercept = false;
    c.cfg.min_weight = 6.0;
    out.push(c);

    // No skip: a window's weight is not bounded by the argument, and a
    // floor of 0 is met by every row.
    let mut c = case("window", base(6, 2));
    c.cfg.window = Some(120.0);
    c.skips = false;
    out.push(c);
    let mut c = case("min_weight 0", base(6, 2));
    c.cfg.min_weight = 0.0;
    c.skips = false;
    out.push(c);
    out
}

fn build(c: &Case) -> EwRidge {
    let mut m = EwRidge::new(c.cfg.clone()).unwrap();
    m.set_min_weight_output_only(c.output_only);
    m.set_keep_factor(c.keep_factor);
    if !c.targets.is_empty() {
        m.set_target_min_weight(c.targets.clone()).unwrap();
    }
    if c.first_period {
        m.set_first_period_by_share(true);
    }
    m
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A Gram as `gram_parts` serves it: the matrix, its weight, and each
/// target's cross-moments, weight and means.
fn gram_bits(p: &GramPart, k: usize) -> Vec<u64> {
    let mut v: Vec<f64> = (0..k)
        .flat_map(|i| (0..k).map(move |j| (i, j)))
        .map(|(i, j)| p.cov.raw(i, j))
        .collect();
    v.push(p.cov.n_eff());
    v.extend(p.cross_centred.iter().flatten());
    v.extend(&p.target_weights);
    v.extend(p.means_by_target.iter().flatten());
    bits(&v)
}

/// Everything a reader of the fit is served on one row, as bits.
#[derive(Debug, PartialEq)]
struct Served {
    floors: Vec<Vec<u64>>,
    infl: Vec<u64>,
    row_infl: Vec<u64>,
    ill: bool,
    pred: Vec<u64>,
    n_eff: u64,
    sig2: Vec<u64>,
    target_n_eff: Vec<u64>,
    /// The fit, where the weight is at the floor and so a row can read it.
    coef: Option<Option<Vec<Vec<u64>>>>,
    grams: Option<Vec<Vec<u64>>>,
}

/// The thread's skip mode for as long as it lives, set back to the model's
/// way (`true`) when it drops -- on a panic too, so a failing test does not
/// leave a later test on its thread running the reference mode.
struct SkipMode;

impl SkipMode {
    fn set(skip: bool) -> Self {
        SKIP_UNREAD.with(|s| s.set(skip));
        SkipMode
    }
}

impl Drop for SkipMode {
    fn drop(&mut self) {
        SKIP_UNREAD.with(|s| s.set(true));
    }
}

/// Run `c` over `data`, every due solve run or not, with a save and load
/// at each row of `loads` and the readiness settled, as a chunk's end
/// settles it, every 37 rows. Returns what each row served and the solves
/// run and passed.
fn run(c: &Case, data: &[Row], skip: bool, loads: &[usize]) -> (Vec<Served>, u64, u64) {
    let _mode = SkipMode::set(skip);
    let (solved0, unread0) = (SOLVED.with(|s| s.get()), UNREAD.with(|s| s.get()));
    let mut m = build(c);
    let floor = m.unread_floor();
    let k = c.cfg.k_total() as f64;
    let read_floor = c
        .targets
        .iter()
        .copied()
        .fold(c.cfg.min_weight.min(k), f64::min);
    let floors = [floor, m.own_floor(), c.cfg.min_weight, read_floor, 0.0];
    let mut out = Vec::with_capacity(data.len());
    let mut step = Step::default();
    for (i, r) in data.iter().enumerate() {
        if loads.contains(&i) {
            m = EwRidge::restore(&m.state()).unwrap();
        }
        if c.blend_every.is_some_and(|b| i % b == b - 1) {
            m.blend_toward_long_run();
        }
        let floor_preds: Vec<Vec<u64>> = floors
            .iter()
            // At a floor of 0 every row reads the fit, so only above `F`.
            .filter(|&&f| f >= floor)
            .map(|&f| bits(&m.predict_with_floor(&r.x, &r.y, r.d, f).pred))
            .collect();
        let mut infl = Vec::new();
        m.error_inflation_into(&mut infl);
        let mut row_infl = Vec::new();
        m.row_error_inflation_into(&r.x, r.d, &mut row_infl);
        let ill = m.ill_conditioned();
        m.step_into(&r.x, &r.y, r.d, r.w, &mut step);
        let mut tn = Vec::new();
        m.target_n_eff_into(&mut tn);
        let at_floor = m.n_eff() >= floor;
        out.push(Served {
            floors: floor_preds,
            infl: bits(&infl),
            row_infl: bits(&row_infl),
            ill,
            pred: bits(&step.pred),
            n_eff: step.n_eff.to_bits(),
            sig2: bits(&m.sigma2()),
            target_n_eff: bits(&tn),
            coef: at_floor.then(|| {
                m.coefficients()
                    .map(|fit| fit.iter().map(|b| bits(b)).collect())
            }),
            grams: (i % 25 == 0).then(|| {
                let (parts, _) = m.gram_parts();
                parts
                    .iter()
                    .map(|p| gram_bits(p, c.cfg.k_total()))
                    .collect()
            }),
        });
        if i % 37 == 36 {
            m.settle_readiness();
        }
    }
    (
        out,
        SOLVED.with(|s| s.get()) - solved0,
        UNREAD.with(|s| s.get()) - unread0,
    )
}

/// The reference mode does not outlive a test that panics in it (task 313):
/// the guard restores the model's way as the panic unwinds.
#[test]
fn the_skip_mode_is_restored_on_a_panic() {
    let caught = std::panic::catch_unwind(|| {
        let _mode = SkipMode::set(false);
        assert!(!SKIP_UNREAD.with(std::cell::Cell::get));
        panic!("a failing test in the reference mode");
    });
    assert!(caught.is_err());
    assert!(SKIP_UNREAD.with(std::cell::Cell::get));
}

/// Every served number to the bit against the model that runs every due
/// solve, across the cases and three seeds, with loads mid-stream -- one
/// in the middle of a run of passed solves -- and the solves passed plus
/// those run equal to the reference's.
#[test]
fn a_passed_solve_moves_no_served_bit() {
    let mut passed_in_all = 0;
    for c in cases() {
        let (mut skipped, mut total) = (0, 0);
        for seed in [311u64, 7, 2_026] {
            let data = rows(700, c.cfg.n_features, c.cfg.n_targets, seed);
            let (want, solved_ref, unread_ref) = run(&c, &data, false, &[]);
            assert_eq!(unread_ref, 0, "{}: the reference passes nothing", c.name);
            let (got, solved, unread) = run(&c, &data, true, &[]);
            let (loaded, solved_l, unread_l) = run(&c, &data, true, &[3, 77, 230, 401, 650]);
            for (i, ((w, g), l)) in want.iter().zip(&got).zip(&loaded).enumerate() {
                assert_eq!(w, g, "{} seed {seed} row {i}", c.name);
                assert_eq!(w, l, "{} seed {seed} row {i}, loaded", c.name);
            }
            assert_eq!(solved + unread, solved_ref, "{}: the schedule", c.name);
            assert_eq!((solved_l, unread_l), (solved, unread), "{}: loaded", c.name);
            skipped += unread;
            total += solved_ref;
        }
        eprintln!("{}: {skipped} of {total} due solves passed", c.name);
        assert_eq!(
            skipped > 0,
            c.skips,
            "{}: {skipped} of {total} passed",
            c.name
        );
        passed_in_all += skipped;
    }
    assert!(passed_in_all > 100, "{passed_in_all}");
}

/// The argument's floor, at the edge: on a stream whose weight climbs in
/// steps of every size through `F (1 − s)`, each due solve is passed
/// exactly where the weight after its row is below `F (1 − s)(1 − 2⁻¹⁶)`,
/// and run everywhere else; and no row after a passed solve, up to and
/// including the next due one, reads the weight `F`.
#[test]
fn a_due_solve_is_passed_exactly_below_the_floors_margin() {
    let mut cfg = base(4, 1);
    cfg.decay = Decay::Halflife(4_000.0);
    let mut m = EwRidge::new(cfg).unwrap();
    m.set_min_weight_output_only(true);
    let floor = m.unread_floor();
    assert_eq!(floor, 5.0);
    let edge = floor * (1.0 - crate::DEFAULT_SOLVE_SHARE) * (1.0 - 1.0 / 65_536.0);
    let mut s = 9u64;
    let (mut passed, mut run_below, mut since_pass) = (0, 0, false);
    let mut step = Step::default();
    for i in 0..8_000 {
        let x: Vec<f64> = (0..4).map(|_| lcg(&mut s)).collect();
        let w = [1e-3, 0.02, 0.003, 0.01, 1e-6][i % 5] * (1.0 + 0.5 * lcg(&mut s));
        let before = m.n_eff();
        if since_pass {
            assert!(before < floor, "row {i} reads {before} after a pass");
        }
        let (u0, s0) = (UNREAD.with(|c| c.get()), SOLVED.with(|c| c.get()));
        m.step_into(&x, &[Some(x[0])], f64::from(u8::from(i > 0)), w, &mut step);
        let after = m.n_eff();
        if UNREAD.with(|c| c.get()) > u0 {
            assert!(after < edge, "row {i}: passed at {after}");
            passed += 1;
            since_pass = true;
        }
        if SOLVED.with(|c| c.get()) > s0 {
            assert!(after >= edge, "row {i}: ran at {after}");
            run_below += usize::from(after < floor);
            since_pass = false;
        }
        if after > 2.0 * floor {
            break;
        }
    }
    eprintln!("{passed} passed, {run_below} run in [edge, F)");
    // Evidence: the stream passed solves and ran some in `[edge, F)`.
    assert!(
        passed > 20 && run_below > 0,
        "{passed} passed, {run_below} run below F"
    );
}

/// `ill_conditioned` names no fit below `min_weight`, where the model
/// predicts from none: the last solve's condition is the floor's business
/// there, and the last solve run may be older than the schedule's latest.
#[test]
fn no_fit_is_named_ill_below_min_weight() {
    let mut cfg = base(3, 1);
    cfg.ridge = vec![0.0];
    cfg.standardize = false;
    cfg.decay = Decay::Halflife(f64::INFINITY);
    cfg.solve_share = None;
    cfg.solve_every = 0.0;
    cfg.max_rows_between_solves = 1;
    cfg.min_weight = 30.0;
    let mut m = EwRidge::new(cfg).unwrap();
    let mut s = 278u64;
    let mut named = Vec::new();
    for _ in 0..60 {
        let (a, b, c) = (lcg(&mut s), lcg(&mut s), lcg(&mut s));
        // `x₁ = x₀ + 1e-11 b`, past `1/u` in condition.
        let x = [2.0 + a, 2.0 + a + 1e-11 * b, -1.0 + c];
        m.step(&x, &[Some(x[0] - x[2])], 1.0, 1.0);
        named.push((m.n_eff(), m.ill_conditioned()));
    }
    let slot_ill = |m: &EwRidge| (0..m.ready.system_of.len()).any(|s| m.slot_ill(s));
    assert!(slot_ill(&m), "the fixture: the last solve is withheld");
    for (w, ill) in &named {
        if *w < 30.0 {
            assert!(!ill, "named ill at {w}, below min_weight");
        }
    }
    assert!(named.iter().any(|(w, ill)| *w >= 30.0 && *ill), "{named:?}");
}

/// The module docs' item 6, a read of a passed fit that can move a served
/// number: a target whose weight has decayed to exactly 0 keeps the fit a
/// solve gave it before. Here the second target's only row before the end
/// is the first, below the floor of 1.5, whose solve passes; a thousand
/// rows of weight `1e-3`, a half-life apart, keep the stream under the
/// floor while the target's weight underflows; rows of weight 1 at no
/// clock step bring the stream past the floor, and the first solve run
/// meets the target at weight 0; then the target returns on a row too
/// light to bring a solve due. Running every due solve, the target kept the
/// prior fit its weight-0 solve gave it (a ridge over no rows: the zero
/// slope and the target's stale mean), and the row after its return
/// predicted from it; passing the first, it has no fit until the next
/// solve, on that row.
#[test]
fn a_target_decayed_to_nothing_keeps_the_fit_of_the_last_solve_run() {
    let mut cfg = base(1, 2);
    cfg.decay = Decay::Halflife(1.0);
    cfg.ridge = vec![0.1];
    cfg.standardize = false;
    cfg.min_weight = 1.5;
    let mut s = 3u64;
    // A row's feature, targets, weight and clock step.
    type Row1 = ([f64; 1], [Option<f64>; 2], f64, f64);
    let data: Vec<Row1> = (0..1_120)
        .map(|i| {
            let x = [lcg(&mut s)];
            let y1 = (i == 0 || i >= 1_111).then_some(1.0 - x[0]);
            let (w, d) = match i {
                1..=1_100 => (1e-3, 1.0),
                1_111 => (1e-3, 0.0),
                _ => (1.0, 0.0),
            };
            (x, [Some(2.0 * x[0]), y1], w, d)
        })
        .collect();
    let served = |skip: bool| {
        let _mode = SkipMode::set(skip);
        let mut m = EwRidge::new(cfg.clone()).unwrap();
        m.set_min_weight_output_only(true);
        let mut out = Vec::new();
        for (i, (x, y, w, d)) in data.iter().enumerate() {
            out.push(m.step(x, y, *d, *w).pred[1]);
            if i == 1_105 {
                assert_eq!(
                    m.acc.wj[1], 0.0,
                    "the fixture: the target's weight underflowed"
                );
            }
        }
        out
    };
    let (all, passed) = (served(false), served(true));
    let moved: Vec<usize> = (0..data.len())
        .filter(|&i| all[i].to_bits() != passed[i].to_bits())
        .collect();
    assert_eq!(
        moved,
        vec![1_112],
        "{:?} against {:?}",
        &all[1_108..],
        &passed[1_108..]
    );
    assert!(all[1_112].is_finite() && passed[1_112].is_nan());
}
