//! Window expressions as model targets (docs/PLAN.md task 104): a target
//! that is a formula of the row's future is resolved by the bank's own window
//! core, and the row is learned from once its window has closed and its
//! embargo has passed. Held against the column form -- the same formula
//! written by `with_windows(..., like=spec)` and fed back as a plain column
//! target under the same embargo -- bit for bit on every prediction, chunked
//! and whole; and the rules around it: what `fit_predict` refuses, what
//! `fit` takes, a state saved mid-window, and groups on clocks of their own.

use online_polars::{Bank, Like, Spec, WindowsConfig, WindowsRun};
use polars::prelude::*;

const TREE: &str = r#"["-", ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}], ["col", "mid"]]"#;

/// Two groups interleaved on one clock that steps by a random fraction, a
/// feature null now and then, and one gap past the cap half way through.
fn stream(n: usize, seed: u64) -> DataFrame {
    let mut s = seed;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut t = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut mid = Vec::with_capacity(n);
    let mut clock = 0.0;
    let mut level = 100.0;
    for i in 0..n {
        clock += if i == n / 2 {
            200.0
        } else {
            0.25 + lcg() * 2.0
        };
        t.push(clock);
        g.push(if lcg() < 0.5 { "a" } else { "b" });
        x.push(if i % 13 == 4 { None } else { Some(lcg() - 0.5) });
        level += lcg() - 0.5;
        mid.push(level);
    }
    df!("t" => t, "g" => g, "x" => x, "mid" => mid).unwrap()
}

fn spec(targets: &str, embargo: Option<f64>, extra: &str) -> Spec {
    let embargo = embargo.map_or(String::new(), |e| format!(r#""embargo": {e},"#));
    serde_json::from_str(&format!(
        r#"{{
            "name": "m",
            "model": {{"type": "ewridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": {targets},
            "features": ["x"],
            "clock": "t", "gap_cap": 100.0, "half_life": 50.0, "group": "g",
            "min_weight": 3.0, "emit_clocks": true,
            {embargo}
            {extra}
            "fit_intercept": true
        }}"#
    ))
    .unwrap()
}

fn native(embargo: Option<f64>) -> Spec {
    spec(
        &format!(r#"[{{"name": "fwd", "formula": {TREE}}}]"#),
        embargo,
        "",
    )
}

fn plain(embargo: Option<f64>) -> Spec {
    spec(r#"["fwd"]"#, embargo, "")
}

/// The column form: the same formula over the frame, `like=` the spec, so
/// a row the spec would skip gets a null target.
fn with_column(df: &DataFrame, like: &Spec) -> DataFrame {
    let config: WindowsConfig = serde_json::from_str(&format!(
        r#"{{"formulas": [{{"name": "fwd", "tree": {TREE}}}], "clock": "t", "gap_cap": 100.0,
             "group": "g"}}"#
    ))
    .unwrap();
    let config = WindowsConfig {
        like: Some(Like {
            spec: like.name.clone(),
            accept: like.features.clone(),
        }),
        ..config
    };
    let mut run = WindowsRun::new(config, df.schema()).unwrap();
    let mut out = run.feed(df, None).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    out
}

/// `df` through a fresh bank of `spec`, in `chunks` chunks, as the spec's
/// struct fields.
fn run(spec: &Spec, df: &DataFrame, chunks: usize) -> DataFrame {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    feed(&mut bank, df, chunks)
}

fn feed(bank: &mut Bank, df: &DataFrame, chunks: usize) -> DataFrame {
    feed_with(bank, df, chunks, false)
}

/// [`feed`] with `learn_only` said per call, as `ModelBank.fit` says it.
fn feed_with(bank: &mut Bank, df: &DataFrame, chunks: usize, learn_only: bool) -> DataFrame {
    let n = df.height();
    let step = n.div_ceil(chunks);
    let mut acc: Option<DataFrame> = None;
    let mut i = 0;
    while i < n {
        let len = step.min(n - i);
        let cols = bank
            .fit_predict_from_with(&df.slice(i as i64, len), 0, learn_only)
            .unwrap();
        let st = cols[0].struct_().unwrap().clone().unnest();
        match &mut acc {
            None => acc = Some(st),
            Some(a) => {
                a.vstack_mut(&st).unwrap();
            }
        }
        i += len;
    }
    acc.unwrap()
}

fn column(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
    df.column(name).unwrap().f64().unwrap().iter().collect()
}

/// The claim: a formula target learned natively is the column form fed back
/// as a plain target under the same embargo, prediction for prediction, on
/// every row -- the rows a gap cut and the rows the spec skips included --
/// whatever the chunking.
#[test]
fn a_formula_target_is_the_column_form_fed_back_under_the_embargo() {
    let df = stream(600, 7);
    let embargo = Some(12.5);
    let reference = run(&plain(embargo), &with_column(&df, &native(embargo)), 1);
    let want = column(&reference, "pred_fwd");
    assert!(
        want.iter().filter(|p| p.is_some()).count() > 300,
        "the reference predicts: {} rows",
        want.iter().filter(|p| p.is_some()).count()
    );
    for chunks in [1, 7, 600] {
        let got = run(&native(embargo), &df, chunks);
        assert_eq!(column(&got, "pred_fwd"), want, "{chunks} chunks");
        assert_eq!(
            column(&got, "learned_clock"),
            column(&reference, "learned_clock"),
            "{chunks} chunks: the same row learned at the same time"
        );
        // The target is not known at the row it is scored on.
        assert!(column(&got, "resid_fwd").iter().all(Option::is_none));
    }
}

/// `fit_predict` refuses an embargo below the longest forward window, none
/// included, before any row is fed; `fit` -- the call told it keeps the
/// state alone -- takes both, and predicts new rows as the embargoed fit
/// does, since each row is learned once its window has closed either way.
#[test]
fn fit_predict_refuses_a_short_embargo_and_fit_takes_it() {
    let df = stream(300, 3);
    for short in [None, Some(5.0)] {
        let mut bank = Bank::new(vec![native(short)]).unwrap();
        let err = bank.fit_predict(&df).unwrap_err().to_string();
        assert!(
            err.contains("fit_predict needs an embargo of at least 10"),
            "{err}"
        );
        assert!(err.contains("takes any embargo"), "{err}");
        assert_eq!(bank.rows_fed(), 0);
    }
    let later = df.slice(250, 50);
    let mut covered = Bank::new(vec![native(Some(10.0))]).unwrap();
    feed(&mut covered, &df.slice(0, 250), 3);
    let want = covered.predict(&later).unwrap();
    for short in [None, Some(5.0)] {
        let mut bank = Bank::new(vec![native(short)]).unwrap();
        feed_with(&mut bank, &df.slice(0, 250), 3, true);
        let got = bank.predict(&later).unwrap();
        let (g, w) = (
            got[0].struct_().unwrap().clone().unnest(),
            want[0].struct_().unwrap().clone().unnest(),
        );
        assert_eq!(column(&g, "pred_fwd"), column(&w, "pred_fwd"), "{short:?}");
        let err = bank.fit_predict(&later).unwrap_err().to_string();
        assert!(err.contains("fit_predict needs an embargo"), "{err}");
    }
}

/// `RunConfig::validate`, which the command line's `--dry-run` runs,
/// refuses a short embargo where the run would: in a run that writes
/// predictions. A run that keeps none (`--no-output`) takes it, and so does
/// a scoring run, which learns nothing. Before task 154 the dry run passed
/// the config and the run refused it at its first chunk.
#[test]
fn validate_refuses_a_short_embargo_where_the_run_would() {
    let cfg = |output: &str, embargo: Option<f64>| online_polars::RunConfig {
        input: "in.parquet".into(),
        output: output.into(),
        input_format: None,
        output_format: None,
        chunk_size: 64,
        load_state: None,
        skip_learned: false,
        save_state: Some("bank.state".into()),
        keep_columns: vec![],
        predict: false,
        closed_groups: None,
        specs: vec![native(embargo)],
    };
    for short in [None, Some(5.0)] {
        let err = cfg("out.parquet", short).validate().unwrap_err();
        assert!(
            err.contains("fit_predict needs an embargo of at least 10"),
            "{short:?}: {err}"
        );
        cfg("", short)
            .validate()
            .expect("a run that keeps no prediction takes any embargo");
        let mut scoring = cfg("out.parquet", short);
        scoring.predict = true;
        scoring.save_state = None;
        scoring.load_state = Some("bank.state".into());
        scoring.validate().expect("a scoring run learns nothing");
    }
    cfg("out.parquet", Some(10.0))
        .validate()
        .expect("an embargo of the window covers it");
}

/// A state saved while windows are open resumes as if nothing had been
/// saved: the core's held rows go with the bank.
#[test]
fn a_state_saved_mid_window_resumes_as_one_run() {
    let df = stream(400, 11);
    let whole = run(&native(Some(12.0)), &df, 1);
    let mut bank = Bank::new(vec![native(Some(12.0))]).unwrap();
    let first = feed(&mut bank, &df.slice(0, 203), 2);
    let bytes = bank.save_bytes().unwrap();
    let mut resumed = Bank::load_bytes(&bytes, Some(&[native(Some(12.0))])).unwrap();
    let mut out = first;
    out.vstack_mut(&feed(&mut resumed, &df.slice(203, 197), 3))
        .unwrap();
    assert_eq!(column(&out, "pred_fwd"), column(&whole, "pred_fwd"));
    assert_eq!(
        column(&out, "learned_clock"),
        column(&whole, "learned_clock")
    );
    // And the saved bytes are what the resumed bank writes back.
    let again = Bank::load_bytes(&bytes, None)
        .unwrap()
        .save_bytes()
        .unwrap();
    assert_eq!(again, bytes);
}

/// Each group has its own window core, as it has its own stream: groups
/// interleaved on clocks of their own run, and each group's predictions are
/// the predictions of that group's rows run alone.
#[test]
fn groups_are_independent_on_their_own_clocks() {
    let n = 400;
    let mut df = stream(n, 21);
    // Give each group its own clock: group b's rows restart at 0.5 and step
    // by their own gaps, so the stream steps back on most rows.
    let g: Vec<String> = df
        .column("g")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|v| v.unwrap().to_string())
        .collect();
    let t: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    let mut own = vec![0.0; n];
    let (mut ta, mut tb) = (0.0, 0.5);
    for i in 0..n {
        let step = if i == 0 { t[0] } else { t[i] - t[i - 1] };
        if g[i] == "a" {
            ta += step;
            own[i] = ta;
        } else {
            tb += step;
            own[i] = tb;
        }
    }
    df.with_column(Column::new("t".into(), own)).unwrap();
    let got = run(&native(Some(12.5)), &df, 5);
    assert_eq!(got.height(), n);
    for key in ["a", "b"] {
        let mask: BooleanChunked = g.iter().map(|v| Some(v == key)).collect();
        let alone = run(&native(Some(12.5)), &df.filter(&mask).unwrap(), 3);
        let picked = got.filter(&mask).unwrap();
        assert_eq!(
            column(&picked, "pred_fwd"),
            column(&alone, "pred_fwd"),
            "group {key}"
        );
        assert!(
            column(&alone, "pred_fwd")
                .iter()
                .filter(|p| p.is_some())
                .count()
                > 100
        );
    }
}

/// What the spec refuses by name: a formula with no operator looking ahead,
/// a group close, and a model that learns from no regression target.
#[test]
fn what_a_formula_target_may_not_be() {
    let back = r#"[{"name": "f", "formula": ["ewm_mean", ["col", "mid"], {"half_life": 5.0}]}]"#;
    let err = serde_json::from_str::<Spec>(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": {back}, "features": ["x"]}}"#
    ))
    .unwrap_err()
    .to_string();
    assert!(err.contains("looking ahead"), "{err}");
    let mut s = spec(
        &format!(r#"[{{"name": "fwd", "formula": {TREE}}}]"#),
        None,
        r#""group_close": "monotone","#,
    );
    let err = s.check().unwrap_err();
    assert!(
        err.contains("group_close does not work with a formula target"),
        "{err}"
    );
    let mut s: Spec = serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ew_class", "classes": ["u", "d"],
             "precision_prior": 1.0}},
             "targets": [{{"name": "fwd", "formula": {TREE}}}], "features": ["x"]}}"#
    ))
    .unwrap();
    let err = s.check().unwrap_err();
    assert!(
        err.contains("a formula target") && err.contains("classifies"),
        "{err}"
    );
}

/// One spec's struct column of a bank's output, unnested.
fn fields(cols: &[Column], spec: usize) -> DataFrame {
    cols[spec].struct_().unwrap().clone().unnest()
}

/// A Boolean column that is a group key and is read by a formula target
/// (review round 4, PA1). The chunk holds it twice under one name -- the
/// formula's boolean and the key's text -- and the key lookup fell back to
/// the first column that was not a number or a clock, the boolean, so
/// `fit_predict` and `predict` panicked at an `unreachable!`. Both roles in
/// one spec, and in two specs in either order: the bank keys the groups as
/// it keys the same values given as text, and predicts the same.
#[test]
fn a_boolean_group_column_a_formula_target_also_reads() {
    let base = stream(240, 13);
    let b: Vec<bool> = base
        .column("g")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|g| g == Some("a"))
        .collect();
    let text: Vec<&str> = b
        .iter()
        .map(|&v| if v { "true" } else { "false" })
        .collect();
    let mut df = base.clone();
    df.with_column(Column::new("b".into(), b)).unwrap();
    df.with_column(Column::new("bs".into(), text)).unwrap();
    let tree = r#"["-", ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}],
                       ["cast", ["col", "b"], "Float64"]]"#;
    let formula = |name: &str, group: Option<&str>| -> Spec {
        let group = group.map_or(String::new(), |g| format!(r#""group": "{g}","#));
        serde_json::from_str(&format!(
            r#"{{"name": "{name}", "model": {{"type": "ewridge", "ridge": 1e-6}},
                 "targets": [{{"name": "fwd", "formula": {tree}}}], "features": ["x"],
                 "clock": "t", "gap_cap": 100.0, "half_life": 50.0, "embargo": 12.5,
                 {group} "min_weight": 2.0}}"#
        ))
        .unwrap()
    };
    let grouped = |group: &str| -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "k", "model": {{"type": "ewridge", "ridge": 1e-6}},
                 "targets": ["mid"], "features": ["x"], "group": "{group}",
                 "half_life": 50.0, "min_weight": 2.0}}"#
        ))
        .unwrap()
    };
    let pred = |s: &Spec| {
        if s.name == "f" {
            "pred_fwd"
        } else {
            "pred_mid"
        }
    };
    let cases = [
        (
            "one spec",
            vec![formula("f", Some("b"))],
            vec![formula("f", Some("bs"))],
        ),
        (
            "the formula spec first",
            vec![formula("f", None), grouped("b")],
            vec![formula("f", None), grouped("bs")],
        ),
        (
            "the grouped spec first",
            vec![grouped("b"), formula("f", None)],
            vec![grouped("bs"), formula("f", None)],
        ),
    ];
    for (label, specs, reference) in cases {
        // `predict` on a fresh bank reads the chunk as `fit_predict` does.
        let scored = Bank::new(specs.clone()).unwrap().predict(&df).unwrap();
        let expected = Bank::new(reference.clone()).unwrap().predict(&df).unwrap();
        for (si, s) in specs.iter().enumerate() {
            let p = pred(s);
            assert_eq!(
                column(&fields(&scored, si), p),
                column(&fields(&expected, si), p),
                "{label}: a fresh bank's predict, {}",
                s.name
            );
        }
        let mut bank = Bank::new(specs.clone()).unwrap();
        let got = bank.fit_predict(&df.slice(0, 200)).unwrap();
        let mut keyed = Bank::new(reference).unwrap();
        let want = keyed.fit_predict(&df.slice(0, 200)).unwrap();
        let after = bank.predict(&df.slice(200, 40)).unwrap();
        let before = keyed.predict(&df.slice(200, 40)).unwrap();
        for (si, s) in specs.iter().enumerate() {
            let p = pred(s);
            let fit = column(&fields(&got, si), p);
            assert_eq!(fit, column(&fields(&want, si), p), "{label}: {}", s.name);
            assert!(
                fit.iter().flatten().count() > 50,
                "{label}: {} predicts",
                s.name
            );
            assert_eq!(
                column(&fields(&after, si), p),
                column(&fields(&before, si), p),
                "{label}: predict after the fit, {}",
                s.name
            );
        }
        assert_eq!(bank.groups(), keyed.groups(), "{label}: the keys");
    }
}

/// `describe()` reads a formula target as the models were handed it (review
/// round 4, PA9): a row is fed before its window has closed, so its target
/// is counted a null then, and the value it is released with joins the
/// target's statistics when it is released -- where `rows_learned` counts
/// it. Every value went uncounted: the target's row read `count 0` beside a
/// `rows_learned` of hundreds. The oracle is the column form, the same
/// formula over the frame: each group's statistics are those of its values,
/// computed here from scratch. A last row per group far past the rest
/// releases every row before it; its own window never closes, so it stays a
/// null on both sides.
#[test]
fn describe_counts_a_formula_target_as_the_models_were_handed_it() {
    let n = 300;
    let mut df = stream(n, 5);
    let last = df.column("t").unwrap().f64().unwrap().get(n - 1).unwrap();
    let tail = df!(
        "t" => [last + 1000.0, last + 1001.0],
        "g" => ["a", "b"],
        "x" => [Some(0.25), Some(-0.25)],
        "mid" => [100.0, 100.0],
    )
    .unwrap();
    df.vstack_mut(&tail).unwrap();
    let spec = native(Some(12.5));
    let reference = with_column(&df, &spec);
    let groups_of = |frame: &DataFrame| -> Vec<String> {
        frame
            .column("g")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .map(|v| v.unwrap().to_string())
            .collect()
    };
    let (fed_g, ref_g, ref_fwd) = (
        groups_of(&df),
        groups_of(&reference),
        column(&reference, "fwd"),
    );
    for chunks in [1, 7, 302] {
        let mut bank = Bank::new(vec![spec.clone()]).unwrap();
        feed(&mut bank, &df, chunks);
        let described = bank.describe(0, None).unwrap();
        let summary = bank.summary(0, None).unwrap();
        for (gi, g) in ["a", "b"].into_iter().enumerate() {
            let what = format!("{chunks} chunks, group {g}");
            let values: Vec<f64> = ref_g
                .iter()
                .zip(&ref_fwd)
                .filter_map(|(rg, v)| (rg == g).then_some(*v).flatten())
                .filter(|v| v.is_finite() && v.abs() <= 1e100)
                .collect();
            let fed = fed_g.iter().filter(|v| *v == g).count() as u64;
            assert!(values.len() > 100, "{what}: {} values", values.len());
            let k = values.len() as f64;
            let mean = values.iter().sum::<f64>() / k;
            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (k - 1.0)).sqrt();
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mask = described.column("group").unwrap().str().unwrap().equal(g)
                & described
                    .column("role")
                    .unwrap()
                    .str()
                    .unwrap()
                    .equal("target");
            let target = described.filter(&mask).unwrap();
            assert_eq!(target.height(), 1, "{what}");
            let u = |c: &str| target.column(c).unwrap().u64().unwrap().get(0);
            let f = |c: &str| target.column(c).unwrap().f64().unwrap().get(0);
            assert_eq!(u("count"), Some(values.len() as u64), "{what}");
            assert_eq!(u("null_count"), Some(fed - values.len() as u64), "{what}");
            let close = |a: Option<f64>, b: f64, rel: f64| {
                a.is_some_and(|a| (a - b).abs() <= rel * b.abs().max(1.0))
            };
            assert!(
                close(f("mean"), mean, 1e-12),
                "{what}: {:?} vs {mean}",
                f("mean")
            );
            assert!(
                close(f("std"), std, 1e-9),
                "{what}: {:?} vs {std}",
                f("std")
            );
            assert_eq!(f("min"), Some(min), "{what}");
            assert_eq!(f("max"), Some(max), "{what}");
            // The two frames agree: every value counted is a row learned.
            let learned = summary
                .column("rows_learned")
                .unwrap()
                .u64()
                .unwrap()
                .get(gi);
            assert_eq!(learned, Some(values.len() as u64), "{what}: rows_learned");
        }
    }
}
