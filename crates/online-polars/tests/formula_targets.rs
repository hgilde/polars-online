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
            "model": {{"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
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
        assert_eq!(bank.rows_seen(), 0);
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
        chunk_rows: 64,
        load_state: None,
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
        r#"{{"name": "m", "model": {{"type": "ew_ridge"}}, "targets": {back}, "features": ["x"]}}"#
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
