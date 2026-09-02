//! Bank-level integration tests: chunk invariance, save/load mid-stream,
//! per-group independence (docs/PLAN.md §9). The oracle tests live in pytest.

use online_polars::{Bank, Spec};
use polars::prelude::*;

fn spec_json(name: &str, group: bool) -> Spec {
    let g = if group { r#""group": "g","# } else { "" };
    serde_json::from_str(&format!(
        r#"{{
            "name": "{name}",
            "model": {{"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "halflife": 60.0,
            "max_dclock": 30.0,
            "weight": "w",
            {g}
            "min_periods": 5.0
        }}"#
    ))
    .unwrap()
}

/// Deterministic stream over 2 groups with nulls sprinkled in.
fn make_df(n: usize) -> DataFrame {
    let mut s = 1234u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut group = Vec::new();
    let mut t = Vec::new();
    let mut x0 = Vec::new();
    let mut x1 = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let g = i % 2;
        group.push(format!("g{g}"));
        clocks[g] += 1.0 + lcg().abs() * 5.0;
        t.push(clocks[g]);
        let a = lcg();
        let b = lcg();
        x0.push(if i % 17 == 5 { None } else { Some(a) });
        x1.push(Some(b));
        y.push(if i % 23 == 7 {
            None
        } else {
            Some(if g == 0 { 2.0 * a - b } else { -a } + 0.01 * lcg())
        });
        w.push(0.5 + lcg().abs());
    }
    df!(
        "g" => group,
        "t" => t,
        "x0" => x0,
        "x1" => x1,
        "y" => y,
        "w" => w,
    )
    .unwrap()
}

fn run_chunked(df: &DataFrame, n_chunks: usize) -> DataFrame {
    let mut bank = Bank::new(vec![spec_json("m", true)]).unwrap();
    let n = df.height();
    let mut outs: Vec<DataFrame> = Vec::new();
    let step = n.div_ceil(n_chunks);
    let mut i = 0;
    while i < n {
        let len = step.min(n - i);
        let chunk = df.slice(i as i64, len);
        let cols = bank.fit_predict(&chunk).unwrap();
        let h = chunk.height();
        outs.push(DataFrame::new(h, cols).unwrap());
        i += len;
    }
    let mut it = outs.into_iter();
    let mut acc = it.next().unwrap();
    for d in it {
        acc.vstack_mut(&d).unwrap();
    }
    acc
}

/// Everything except `coef`, which is emitted on the last row of every chunk
/// by design (docs/PLAN.md §3) and therefore legitimately depends on chunking.
fn drop_coef(df: &DataFrame) -> DataFrame {
    let keep: Vec<String> = df
        .get_column_names()
        .iter()
        .filter(|c| !c.starts_with("coef"))
        .map(|c| c.to_string())
        .collect();
    df.select(keep).unwrap()
}

#[test]
fn chunk_invariance() {
    let df = make_df(400);
    let one = run_chunked(&df, 1);
    let seven = run_chunked(&df, 7);
    let thousand = run_chunked(&df, 400);
    // Struct columns: compare via unnest for readable failures.
    for (a, b) in [(&one, &seven), (&one, &thousand)] {
        let ua = drop_coef(&a.clone().unnest(["m"], None).unwrap());
        let ub = drop_coef(&b.clone().unnest(["m"], None).unwrap());
        assert!(ua.equals_missing(&ub), "chunked runs differ");
    }
}

#[test]
fn save_load_mid_stream_is_identical() {
    let df = make_df(300);
    let first = df.slice(0, 150);
    let second = df.slice(150, 150);

    let mut b1 = Bank::new(vec![spec_json("m", true)]).unwrap();
    b1.fit_predict(&first).unwrap();
    let bytes = b1.save_bytes().unwrap();

    let mut b2 = Bank::load_bytes(&bytes, Some(b1.specs())).unwrap();
    let out1 = b1.fit_predict(&second).unwrap();
    let out2 = b2.fit_predict(&second).unwrap();
    let d1 = DataFrame::new(second.height(), out1)
        .unwrap()
        .unnest(["m"], None)
        .unwrap();
    let d2 = DataFrame::new(second.height(), out2)
        .unwrap()
        .unnest(["m"], None)
        .unwrap();
    assert!(d1.equals_missing(&d2));
}

#[test]
fn load_rejects_mismatched_specs() {
    let mut b1 = Bank::new(vec![spec_json("m", true)]).unwrap();
    b1.fit_predict(&make_df(50)).unwrap();
    let bytes = b1.save_bytes().unwrap();
    let other = vec![spec_json("different", true)];
    assert!(Bank::load_bytes(&bytes, Some(&other)).is_err());
    // and without expectations it loads fine
    assert!(Bank::load_bytes(&bytes, None).is_ok());
}

#[test]
fn groups_are_independent() {
    // Feeding only g0's rows must give the same outputs for g0 as feeding both.
    let df = make_df(400);
    let both = run_chunked(&df, 3);
    let mask = df.column("g").unwrap().str().unwrap();
    let is_g0: BooleanChunked = mask.iter().map(|v| Some(v == Some("g0"))).collect();
    let only = df.filter(&is_g0).unwrap();

    let mut bank = Bank::new(vec![spec_json("m", true)]).unwrap();
    let solo = DataFrame::new(only.height(), bank.fit_predict(&only).unwrap()).unwrap();

    let both_g0 = DataFrame::new(both.height(), vec![both.column("m").unwrap().clone()])
        .unwrap()
        .filter(&is_g0)
        .unwrap();
    let a = drop_coef(&both_g0.unnest(["m"], None).unwrap());
    let b = drop_coef(&solo.unnest(["m"], None).unwrap());
    assert!(a.equals_missing(&b));
}

#[test]
fn null_clock_errors_loudly() {
    let df = df!(
        "t" => [Some(1.0), None, Some(3.0)],
        "x0" => [1.0, 2.0, 3.0],
        "x1" => [1.0, 2.0, 3.0],
        "y" => [1.0, 2.0, 3.0],
        "w" => [1.0, 1.0, 1.0],
    )
    .unwrap();
    let mut bank = Bank::new(vec![spec_json("m", false)]).unwrap();
    let err = bank.fit_predict(&df).unwrap_err();
    assert!(err.to_string().contains("clock"), "{err}");
}

/// A value beyond `online_core::INPUT_BOUND` is treated exactly like a null in
/// the same position (docs/IMPROVEMENTS.md C2): a feature or weight skips the
/// row, a target makes it predict-only. At the bound itself the value is used.
#[test]
fn values_beyond_the_bound_are_missing() {
    let df = make_df(200);
    let run = |df: &DataFrame| {
        let mut bank = Bank::new(vec![spec_json("m", true)]).unwrap();
        let cols = bank.fit_predict(df).unwrap();
        let out = DataFrame::new(df.height(), cols).unwrap();
        drop_coef(&out.unnest(["m"], None).unwrap())
    };
    let with = |col: &str, v: Option<f64>| {
        let mut vals: Vec<Option<f64>> = df.column(col).unwrap().f64().unwrap().iter().collect();
        vals[100] = v;
        let mut d = df.clone();
        d.with_column(Column::new(col.into(), vals)).unwrap();
        d
    };
    let bound = online_core::INPUT_BOUND;
    for col in ["x0", "y", "w"] {
        let null = run(&with(col, None));
        for beyond in [bound * 10.0, -bound * 10.0, f64::INFINITY] {
            if col == "w" && beyond < 0.0 {
                continue; // a negative weight is an error, not a missing value
            }
            assert!(
                run(&with(col, Some(beyond))).equals_missing(&null),
                "{col} = {beyond} must act as a null"
            );
        }
        assert!(
            !run(&with(col, Some(bound))).equals_missing(&null),
            "{col} = {bound} is within the bound and must be used"
        );
    }
}

/// Under `on_clock_reset = "error"` a refused chunk leaves the whole bank as it
/// was (docs/IMPROVEMENTS.md C3): not just the group whose clock went
/// backwards, but every other group and spec that shared the chunk. The
/// corrected chunk then feeds normally and gives the same output as a bank
/// that never saw the bad one.
#[test]
fn a_refused_chunk_updates_nothing() {
    let strict = |name: &str| {
        let mut s = spec_json(name, true);
        s.on_clock_reset = online_core::OnClockReset::Error;
        s
    };
    let specs = || vec![strict("m"), strict("m2")];
    let df = make_df(200);
    let first = df.slice(0, 100);
    let good = df.slice(100, 100);
    // Send group g1's clock backwards halfway through the second chunk.
    let mut t = good.column("t").unwrap().f64().unwrap().to_vec();
    t[81] = Some(t[79].unwrap() - 1.0);
    let bad = good
        .clone()
        .with_column(Column::new("t".into(), t))
        .unwrap()
        .clone();

    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&first).unwrap();
    let before = bank.save_bytes().unwrap();
    let err = bank.fit_predict(&bad).unwrap_err().to_string();
    assert!(
        err.contains("goes backwards") && err.contains("row 81"),
        "{err}"
    );
    assert!(
        bank.save_bytes().unwrap() == before,
        "a refused chunk changed the bank"
    );

    let out = bank.fit_predict(&good).unwrap();
    let mut clean = Bank::new(specs()).unwrap();
    clean.fit_predict(&first).unwrap();
    let want = clean.fit_predict(&good).unwrap();
    assert_eq!(out, want);
}
