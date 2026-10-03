//! Bank-level integration tests: chunk invariance, save/load mid-stream,
//! per-group independence (docs/PLAN.md §9). The oracle tests live in pytest.

use online_polars::{Bank, ChunkOut, GroupKey, ModelKind, Spec, Stream};
use polars::prelude::*;

/// The fixture's clock is one clock over every row, so no view of it steps
/// back: ungrouped, or re-keyed onto another column, a spec here reads a
/// stream in order. The tests are about groups, keys and chunks, not the
/// clock; the one that is about the clock,
/// `an_ungrouped_view_of_interleaved_groups_is_refused_by_default`, builds a
/// frame whose groups keep clocks of their own.
fn spec_json(name: &str, group: bool) -> Spec {
    let g = if group { r#""group": "g","# } else { "" };
    serde_json::from_str(&format!(
        r#"{{
            "name": "{name}",
            "model": {{"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "half_life": 60.0,
            "gap_cap": 30.0,
            "weight": "w",
            {g}
            "min_weight": 5.0
        }}"#
    ))
    .unwrap()
}

/// An ungrouped spec over two interleaved groups with clocks of their own
/// reads them as one stream that steps back on every other row, and the
/// default policy refuses the chunk by name, leaving the bank untouched --
/// where the removed `max` policy absorbed every step into plausible, wrong
/// output.
#[test]
fn an_ungrouped_view_of_interleaved_groups_is_refused_by_default() {
    let spec: Spec = serde_json::from_str(
        r#"{
            "name": "u",
            "model": {"type": "ew_ridge", "ridge": 1e-6},
            "targets": ["y"], "features": ["x0", "x1"], "clock": "t",
            "half_life": 60.0, "gap_cap": 30.0, "weight": "w", "min_weight": 5.0
        }"#,
    )
    .unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    let err = bank
        .fit_predict(&make_df_clocked(200, true))
        .unwrap_err()
        .to_string();
    assert!(err.contains("goes backwards by"), "{err}");
    assert!(err.contains("restart_after_step_back is unset"), "{err}");
    assert_eq!(bank.rows_seen(), 0, "the refused chunk taught nothing");
}

/// Deterministic stream over 2 groups with nulls sprinkled in, on one clock.
fn make_df(n: usize) -> DataFrame {
    make_df_clocked(n, false)
}

/// [`make_df`], with a clock per group when `per_group` -- the same draws, so
/// each group's steps are the same and only the interleaving differs.
fn make_df_clocked(n: usize, per_group: bool) -> DataFrame {
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
        let c = if per_group { g } else { 0 };
        clocks[c] += 1.0 + lcg().abs() * 5.0;
        t.push(clocks[c]);
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
    run_spec_chunked(&spec_json("m", true), df, n_chunks)
}

/// `df` through a fresh bank of `spec` in `n_chunks` near-equal chunks,
/// stacked back into one frame.
fn run_spec_chunked(spec: &Spec, df: &DataFrame, n_chunks: usize) -> DataFrame {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
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
        .filter(|c| !(c.starts_with("coef") || c.starts_with("support_coef")))
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
fn groups_can_be_listed_and_dropped() {
    // docs/IMPROVEMENTS.md U3: a bank says which groups it holds, and can
    // forget the stale ones without touching the rest.
    let df = make_df(400);
    let mut bank = Bank::new(vec![spec_json("m", true), spec_json("u", false)]).unwrap();
    assert_eq!(bank.groups(), vec![vec![], vec![]]);
    assert_eq!(bank.rows_seen(), 0);
    bank.fit_predict(&df.slice(0, 200)).unwrap();

    let key = |k: &str| GroupKey(Some(k.to_string()));
    let t = df.column("t").unwrap().f64().unwrap();
    let gs = df.column("g").unwrap().str().unwrap();
    let x0 = df.column("x0").unwrap().f64().unwrap();
    let last_clock = |g: Option<&str>| {
        (0..200)
            .rev()
            .find(|&i| g.is_none_or(|g| gs.get(i) == Some(g)))
            .map(|i| t.get(i).unwrap())
    };
    // A group's count is of the rows the null policy let through.
    let processed = |g: Option<&str>| {
        (0..200)
            .filter(|&i| g.is_none_or(|g| gs.get(i) == Some(g)) && x0.get(i).is_some())
            .count() as u64
    };
    assert!(
        processed(None) < 200,
        "the fixture should have null features"
    );
    // Sorted by key; an ungrouped spec has the one key "".
    assert_eq!(
        bank.groups(),
        vec![
            vec![
                (key("g0"), processed(Some("g0")), last_clock(Some("g0"))),
                (key("g1"), processed(Some("g1")), last_clock(Some("g1"))),
            ],
            vec![(GroupKey::ungrouped(), processed(None), last_clock(None))],
        ]
    );
    // The bank counts every row it was fed, skipped or not.
    assert_eq!(bank.rows_seen(), 200);

    // Scoped to one spec, counting only what was actually there.
    assert_eq!(bank.drop_groups(&[key("g1"), key("zz")], Some(0)), Ok(1));
    assert_eq!(bank.groups()[0].len(), 1);
    assert_eq!(bank.groups()[1].len(), 1);
    assert!(bank.drop_groups(&[key("g0")], Some(2)).is_err());

    // Untouched groups continue exactly as before; the dropped one restarts.
    let second = df.slice(200, 200);
    let mut control = Bank::new(vec![spec_json("m", true), spec_json("u", false)]).unwrap();
    control.fit_predict(&df.slice(0, 200)).unwrap();
    let expected = DataFrame::new(200, control.fit_predict(&second).unwrap()).unwrap();
    let got = DataFrame::new(200, bank.fit_predict(&second).unwrap()).unwrap();
    let is_g0: BooleanChunked = second
        .column("g")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|v| Some(v == Some("g0")))
        .collect();
    assert!(
        got.filter(&is_g0)
            .unwrap()
            .equals_missing(&expected.filter(&is_g0).unwrap())
    );
    assert!(
        got.column("u")
            .unwrap()
            .as_materialized_series()
            .equals_missing(expected.column("u").unwrap().as_materialized_series())
    );
    assert!(
        !got.column("m")
            .unwrap()
            .as_materialized_series()
            .equals_missing(expected.column("m").unwrap().as_materialized_series())
    );
    // The dropped group's rows still count as fed, and the count survives a save.
    assert_eq!(bank.rows_seen(), 400);
    // ... but the restarted group only processed the second chunk's rows.
    assert_eq!(
        bank.groups()[0][1].1 + processed(Some("g1")),
        control.groups()[0][1].1
    );
    let reloaded = Bank::load_bytes(&bank.save_bytes().unwrap(), None).unwrap();
    assert_eq!(reloaded.rows_seen(), 400);
    assert_eq!(reloaded.groups(), bank.groups());
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

/// With `restart_after_step_back` unset a refused chunk leaves the whole bank as it
/// was (docs/IMPROVEMENTS.md C3): not just the group whose clock went
/// backwards, but every other group and spec that shared the chunk. The
/// corrected chunk then feeds normally and gives the same output as a bank
/// that never saw the bad one.
#[test]
fn a_refused_chunk_updates_nothing() {
    // A step back is refused by default (`restart_after_step_back` unset).
    let specs = || vec![spec_json("m", true), spec_json("m2", true)];
    let df = make_df(200);
    let first = df.slice(0, 100);
    let good = df.slice(100, 100);
    // Send group g1's clock backwards halfway through the second chunk, and
    // have the chunk bring a group the bank has never seen.
    let mut t = good.column("t").unwrap().f64().unwrap().to_vec();
    t[81] = Some(t[79].unwrap() - 1.0);
    let mut g: Vec<Option<String>> = good
        .column("g")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|v| v.map(str::to_string))
        .collect();
    g[0] = Some("brand-new".to_string());
    let bad = good
        .clone()
        .with_column(Column::new("t".into(), t))
        .unwrap()
        .with_column(Column::new("g".into(), g))
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
    assert!(
        !bank.groups()[0]
            .iter()
            .any(|(k, ..)| k.as_str() == Some("brand-new")),
        "a refused chunk left its new group behind"
    );

    let out = bank.fit_predict(&good).unwrap();
    let mut clean = Bank::new(specs()).unwrap();
    clean.fit_predict(&first).unwrap();
    let want = clean.fit_predict(&good).unwrap();
    assert_eq!(out, want);
}

#[test]
fn coef_is_the_output_s_last_coef_per_group() {
    // `Bank::coef` reads the same coefficients the `coef` field reports: for
    // every group, the list on the group's last row of the chunk, and the
    // same again from a bank restored from the file.
    let df = make_df(400);
    let mut bank = Bank::new(vec![spec_json("m", true)]).unwrap();
    let out = bank.fit_predict(&df).unwrap().remove(0);
    let coef = out.struct_().unwrap().field_by_name("coef").unwrap();
    let coef = coef.list().unwrap();
    let gs = df.column("g").unwrap().str().unwrap();

    let rows = bank.coef(0, None).unwrap();
    assert_eq!(rows.len(), 2, "one row per group, single instance");
    for row in &rows {
        assert_eq!(row.instance, "");
        let g = row.group.as_str().unwrap();
        let last = (0..400).rev().find(|&i| gs.get(i) == Some(g)).unwrap();
        let reported: Vec<f64> = coef
            .get_as_series(last)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(row.coef.as_deref(), Some(reported.as_slice()), "group {g}");
        assert_eq!(reported.len(), 3, "intercept + 2 features");
    }
    assert_eq!(bank.coef(0, Some("g0")).unwrap().len(), 1);
    assert!(bank.coef(0, Some("zzz")).unwrap().is_empty());
    assert!(
        bank.coef(1, None)
            .unwrap_err()
            .contains("spec index 1 out of range")
    );

    let restored = Bank::load_bytes(&bank.save_bytes().unwrap(), None).unwrap();
    assert_eq!(restored.coef(0, None).unwrap(), rows);

    // `coef` does not wait for `min_weight` (5 here): the spec solves every
    // row, so after one row per group there is a fit, and `n_eff` is what
    // says how little is behind it. Under a clock schedule the first row of
    // a stream has not solved, and the row is `None`, as `coef` is; the
    // default schedule solves it, all of its weight being new (docs/PLAN.md
    // task 115 (b)).
    let mut fresh = Bank::new(vec![spec_json("m", true)]).unwrap();
    fresh.fit_predict(&df.slice(0, 2)).unwrap();
    assert!(
        fresh
            .coef(0, None)
            .unwrap()
            .iter()
            .all(|r| r.coef.is_some() && r.n_eff < 5.0)
    );
    let lazy: Spec = serde_json::from_str(
        &serde_json::to_string(&spec_json("m", true))
            .unwrap()
            .replace(r#","max_rows_between_solves":1"#, "")
            .replace(r#""solve_every":null"#, r#""solve_every":10.0"#),
    )
    .unwrap();
    let written = serde_json::to_string(&lazy).unwrap();
    assert!(!written.contains("max_rows_between_solves\":1"));
    assert!(written.contains("\"solve_every\":10.0"), "{written}");
    let mut lazy = Bank::new(vec![lazy]).unwrap();
    lazy.fit_predict(&df.slice(0, 2)).unwrap();
    assert!(
        lazy.coef(0, None)
            .unwrap()
            .iter()
            .all(|r| r.coef.is_none() && r.n_eff > 0.0)
    );
}

#[test]
fn coef_fields_name_every_slot_of_every_list() {
    // `coef_fields` is the layout the models write (`Bank::coef` and the
    // `coef` field), one row per coefficient, named on the field grammar:
    // the `coef_{target}_{term}` prefix takes the combo and instance suffix
    // of the `pred` field it belongs to.
    let spec: Spec = serde_json::from_str(
        r#"{
            "name": "m",
            "model": {"type": "ew_ridge", "ridge": [0.0, 0.5],
                      "feature_sets": [["a", ["x0"]], ["b", ["x0", "x1"]]]},
            "targets": ["y", "z"],
            "features": ["x0", "x1"],
            "half_life": [50.0, 200.0],
            "min_weight": 5.0
        }"#,
    )
    .unwrap();
    let fields = online_polars::coef_fields(&spec);
    let per_instance = 2 * 4 * 3; // targets x (sets x ridges) x (intercept + 2)
    assert_eq!(fields.len(), 2 * per_instance);
    let preds: Vec<_> = online_polars::output_index(&spec)
        .into_iter()
        .filter(|m| m.kind == "pred")
        .collect();
    for (i, f) in fields.iter().enumerate() {
        assert_eq!(f.position, i % per_instance);
        assert_eq!(f.field, format!("coef@h{}", f.half_life.unwrap() as i64));
        let pred = preds
            .iter()
            .find(|m| {
                m.target.as_deref() == Some(f.target.as_str())
                    && m.ridge == f.ridge
                    && m.feature_set == f.feature_set
                    && m.half_life == f.half_life
            })
            .unwrap_or_else(|| panic!("no pred field for {f:?}"));
        let suffix = pred
            .field
            .strip_prefix(&format!("pred_{}", f.target))
            .unwrap();
        assert_eq!(f.name, format!("coef_{}_{}{suffix}", f.target, f.term));
    }
    let terms: Vec<_> = fields[..3].iter().map(|f| f.term.as_str()).collect();
    assert_eq!(terms, ["intercept", "x0", "x1"]);
    assert_eq!(fields[0].name, "coef_y_intercept__a_r0@h50");
    assert_eq!(fields[per_instance - 1].name, "coef_z_x1__b_r0.5@h50");
    assert_eq!(fields[per_instance].name, "coef_y_intercept__a_r0@h200");

    // The list a bank reports is exactly this long, so every name has a value.
    let mut df = make_df(200);
    let z = df.column("y").unwrap().clone().with_name("z".into());
    df.with_column(z).unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    bank.fit_predict(&df).unwrap();
    for row in bank.coef(0, None).unwrap() {
        if let Some(coef) = row.coef {
            assert_eq!(coef.len(), per_instance, "instance {}", row.instance);
        }
    }

    // `holt` names its two terms; `ew_cov` has none.
    let holt: Spec = serde_json::from_str(
        r#"{"name": "h", "model": {"type": "holt"}, "targets": ["y"], "features": [],
            "half_life": 50.0, "min_weight": 2.0}"#,
    )
    .unwrap();
    let names: Vec<_> = online_polars::coef_fields(&holt)
        .into_iter()
        .map(|f| f.name)
        .collect();
    assert_eq!(names, ["coef_y_level", "coef_y_trend"]);
    let cov: Spec = serde_json::from_str(
        r#"{"name": "c", "model": {"type": "ew_cov", "stats": ["mean"]}, "targets": [],
            "features": ["x0"], "half_life": 50.0, "min_weight": 2.0}"#,
    )
    .unwrap();
    assert!(online_polars::coef_fields(&cov).is_empty());
}

#[test]
fn integer_group_keys_match_the_string_cast() {
    // `group_indices` buckets an integer key on its value rather than on the
    // text polars' String cast would give it (docs/PERFORMANCE.md P11). The
    // two must be the same partition with the same key text, whatever the
    // width and sign, at the extremes, and with nulls -- and the output must
    // be bit-identical to feeding the cast column.
    let n = 400;
    let base = make_df(n);
    // Nine distinct values, cycling, with a null every 13th row; the
    // extremes of every width so that the text of each is exercised.
    let cycle = |i: usize, lo: i128, hi: i128| -> Option<i128> {
        if i % 13 == 3 {
            None
        } else {
            Some(match i % 9 {
                0 => lo,
                1 => hi,
                2 => 0,
                3 => 1,
                4 => (-1i128).max(lo),
                5 => hi - 1,
                6 => lo + 1,
                7 => 7,
                _ => 42.min(hi),
            })
        }
    };
    let dtypes = [
        (DataType::Int8, i8::MIN as i128, i8::MAX as i128),
        (DataType::Int16, i16::MIN as i128, i16::MAX as i128),
        (DataType::Int32, i32::MIN as i128, i32::MAX as i128),
        (DataType::Int64, i64::MIN as i128, i64::MAX as i128),
        (DataType::UInt8, 0, u8::MAX as i128),
        (DataType::UInt16, 0, u16::MAX as i128),
        (DataType::UInt32, 0, u32::MAX as i128),
        (DataType::UInt64, 0, u64::MAX as i128),
    ];
    for (dtype, lo, hi) in dtypes {
        let vals: Vec<Option<i128>> = (0..n).map(|i| cycle(i, lo, hi)).collect();
        // Build the typed column from its text, so the expected key text is
        // the one and only source of truth.
        let text: Vec<Option<String>> = vals.iter().map(|v| v.map(|v| v.to_string())).collect();
        let as_text = Series::new("g".into(), text.clone());
        let typed = as_text.cast(&dtype).unwrap();
        assert_eq!(
            typed.null_count(),
            as_text.null_count(),
            "{dtype}: lossless cast"
        );
        let mut df = base.clone();
        df.with_column(typed.clone().into_column()).unwrap();
        let mut df_text = base.clone();
        df_text.with_column(as_text.into_column()).unwrap();

        let mut bank = Bank::new(vec![spec_json("m", true)]).unwrap();
        let out = DataFrame::new(n, bank.fit_predict(&df).unwrap()).unwrap();
        let mut bank_text = Bank::new(vec![spec_json("m", true)]).unwrap();
        let out_text = DataFrame::new(n, bank_text.fit_predict(&df_text).unwrap()).unwrap();
        assert!(
            out.equals_missing(&out_text),
            "{dtype}: same output as the String path"
        );

        let mut want: Vec<GroupKey> = text
            .iter()
            .map(|v| GroupKey(v.clone()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        want.sort();
        let have: Vec<GroupKey> = bank.groups()[0].iter().map(|(k, _, _)| k.clone()).collect();
        assert_eq!(have, want, "{dtype}: key text");
        assert_eq!(
            bank.groups(),
            bank_text.groups(),
            "{dtype}: same groups, counts and clocks"
        );
    }
}

#[test]
fn only_models_that_predict_a_target_have_residual_fields() {
    // `ChunkOut::new` allocates no `resid` buffer for a model whose
    // `predicts_no_target()` is true (docs/PERFORMANCE.md §13), so the
    // schema must never ask for a `resid` field of such a model, and a
    // model that predicts a target must always have one. One spec per
    // `ModelKind`; a new kind that lands on the wrong side of this line
    // shows up here rather than as an index out of bounds in `assemble`.
    let kinds = [
        r#"{"type": "ew_ridge", "ridge": 1e-6}"#,
        r#"{"type": "lasso", "lasso_path": [0.1]}"#,
        r#"{"type": "kalman", "coef_half_life": 100.0}"#,
        r#"{"type": "huber"}"#,
        r#"{"type": "quantile", "quantile": 0.5}"#,
        r#"{"type": "ftrl"}"#,
        r#"{"type": "sgd", "learning_rate": 0.01}"#,
        r#"{"type": "pa"}"#,
        r#"{"type": "holt"}"#,
        r#"{"type": "rls"}"#,
        r#"{"type": "ew_cov", "stats": ["mean"]}"#,
        r#"{"type": "kmeans", "k": 2}"#,
        r#"{"type": "micro", "eps": 1.0}"#,
        r#"{"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0}"#,
        r#"{"type": "seqtest"}"#,
        r#"{"type": "marginal"}"#,
        // The regime and covariance kinds that predict no target, so this
        // test covers every ModelKind rather than a subset (review
        // 2026-09-18, T3). Only parsed here -- their run-time needs (rcov's
        // `group_close`, a hazard column) are not required to read the output
        // schema.
        r#"{"type": "deco"}"#,
        r#"{"type": "rcov"}"#,
        r#"{"type": "hmm", "k": 2, "precision_prior": 0.1}"#,
        r#"{"type": "corrchange"}"#,
        r#"{"type": "bocpd"}"#,
    ];
    let mut seen = Vec::new();
    for model in kinds {
        let features = if model.contains("holt") {
            "[]"
        } else {
            r#"["x0", "x1"]"#
        };
        let spec: Spec = serde_json::from_str(&format!(
            r#"{{"name": "s", "model": {model}, "targets": ["y"], "features": {features},
                "half_life": 50.0, "min_weight": 2.0}}"#
        ))
        .unwrap_or_else(|e| panic!("{model}: {e}"));
        let has_resid = online_polars::output_index(&spec)
            .iter()
            .any(|f| f.kind == "resid");
        assert_eq!(
            has_resid,
            !spec.model.predicts_no_target(),
            "{model}: resid field present = {has_resid}"
        );
        seen.push(spec.model.kind_name().to_string());
    }
    seen.sort();
    seen.dedup();
    // Against `ModelKind::KINDS`, not `kinds.len()`: a new kind added to the
    // enum and forgotten here now fails this test rather than passing a
    // subset (review 2026-09-18, T3).
    assert_eq!(
        seen.len(),
        ModelKind::KINDS.len(),
        "every ModelKind needs a spec here: have {seen:?}, KINDS is {:?}",
        ModelKind::KINDS
    );
}

/// `n` rows of `k` finite features `x0..`, a target `y` on the first two and
/// a weight `w`, no clock and no nulls: every row is accepted, so the
/// coefficient schedule is exactly the one `coef_every` and the chunk ends
/// describe.
fn make_wide_df(n: usize, k: usize) -> DataFrame {
    let mut s = 99u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut xs: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k];
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for _ in 0..n {
        for x in xs.iter_mut() {
            x.push(lcg());
        }
        y.push(2.0 * xs[0].last().unwrap() - xs[1].last().unwrap() + 0.01 * lcg());
        w.push(0.5 + lcg().abs());
    }
    let mut cols: Vec<Column> = xs
        .into_iter()
        .enumerate()
        .map(|(j, x)| Column::new(format!("x{j}").into(), x))
        .collect();
    cols.push(Column::new("y".into(), y));
    cols.push(Column::new("w".into(), w));
    DataFrame::new(n, cols).unwrap()
}

fn features_json(k: usize) -> String {
    let names: Vec<String> = (0..k).map(|j| format!("\"x{j}\"")).collect();
    format!("[{}]", names.join(", "))
}

/// The default `ew_cov` (mean, std, corr) over `k` features: `k + k +
/// k(k-1)/2` statistics a row, 230 for `k = 20`.
fn ew_cov_spec(k: usize) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "c", "model": {{"type": "ew_cov"}}, "targets": ["x0"],
            "features": {}, "half_life": 200.0, "min_weight": 2.0}}"#,
        features_json(k)
    ))
    .unwrap()
}

/// An ungrouped, unclocked `ew_ridge` on `x0`, `x1` that solves every row and
/// reports its coefficients every `coef_every` learned rows (0: chunk ends only).
fn ridge_spec(coef_every: u32) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": ["y"], "features": ["x0", "x1"], "weight": "w", "half_life": 500.0,
            "min_weight": 5.0, "coef_every": {coef_every}}}"#
    ))
    .unwrap()
}

/// Rows on which `coef` is reported: the non-null positions of the field.
fn coef_rows(out: &DataFrame, name: &str) -> Vec<usize> {
    let coef = out
        .column(name)
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("coef")
        .unwrap();
    (0..out.height())
        .filter(|&i| coef.get(i).unwrap() != AnyValue::Null)
        .collect()
}

#[test]
fn runs_through_a_wide_model_are_invisible() {
    // The bank feeds a task's rows through its stream in runs of
    // `ChunkOut::run_rows` rows, one set of buffers each (docs/PERFORMANCE.md
    // §13). A 230-statistic `ew_cov` gets runs of 1 104 rows, so 5 000 rows
    // in one chunk are five runs, the last of them partial; the same rows in
    // five chunks are one run each, and one row at a time is 5 000 runs of
    // one. Chunk invariance says all three are the same computation, and
    // `equals_missing` on every field says the runs kept it that way.
    let k = 20;
    let spec = ew_cov_spec(k);
    let stream = Stream::new(&spec).unwrap();
    assert_eq!(stream.n_slots(), k + k + k * (k - 1) / 2);
    assert_eq!(
        ChunkOut::run_rows(&spec, stream.n_models(), stream.n_slots()),
        1104
    );
    let df = make_wide_df(5000, k);
    let one = run_spec_chunked(&spec, &df, 1);
    let five = run_spec_chunked(&spec, &df, 5);
    let rows = run_spec_chunked(&spec, &df, 5000);
    assert_eq!(one.height(), 5000);
    let unnest = |d: &DataFrame| drop_coef(&d.clone().unnest(["c"], None).unwrap());
    assert!(unnest(&one).equals_missing(&unnest(&five)), "1 vs 5 chunks");
    assert!(
        unnest(&one).equals_missing(&unnest(&rows)),
        "1 vs 5000 chunks"
    );
    // The run split has to be exercised for the test to mean anything: one
    // chunk spans several runs, and the field values are actually there.
    let fields = unnest(&one);
    let corr = fields.column("corr_x0_x1").unwrap();
    assert!(corr.null_count() < 10, "statistics were reported");
}

#[test]
fn coef_is_reported_at_the_chunk_s_end_across_runs() {
    // The last row of a *chunk* reports the coefficients, not the last row
    // of every run inside it: `process_chunk`'s `last` flag is what tells the
    // two apart. A narrow `ew_ridge` gets runs of 65 520 rows (four values
    // a row -- pred, resid, n_eff, settled_frac -- against a 2 MiB budget),
    // so a chunk of twice that plus a hundred is three runs with two
    // boundaries inside it, and exactly one `coef`.
    let spec = ridge_spec(0);
    let stream = Stream::new(&spec).unwrap();
    let run_rows = ChunkOut::run_rows(&spec, stream.n_models(), stream.n_slots());
    assert_eq!(run_rows, 65_520);
    let n = 2 * run_rows + 100;
    let df = make_wide_df(n, 2);

    let one = run_spec_chunked(&spec, &df, 1);
    assert_eq!(coef_rows(&one, "m"), vec![n - 1], "one chunk, one coef");

    // Two chunks: each of the two is two runs (the budget plus fifty rows),
    // and each reports once, on its own last row.
    let two = run_spec_chunked(&spec, &df, 2);
    let half = n.div_ceil(2);
    assert_eq!(coef_rows(&two, "m"), vec![half - 1, n - 1], "two chunks");

    // Everything else is the same computation either way.
    let strip = |d: &DataFrame| drop_coef(&d.clone().unnest(["m"], None).unwrap());
    assert!(strip(&one).equals_missing(&strip(&two)));

    // `coef_every` counts learned rows across runs, since it is stream
    // state: every thousandth row, and the chunk's last, whatever run they
    // fall in. Every row here is accepted, so learned rows are rows.
    let every = run_spec_chunked(&ridge_spec(1000), &df, 1);
    let mut want: Vec<usize> = (0..n).filter(|i| (i + 1) % 1000 == 0).collect();
    want.push(n - 1);
    assert_eq!(coef_rows(&every, "m"), want, "coef_every across runs");
}

#[test]
fn run_rows_is_an_odd_number_of_lines_within_the_budget() {
    // Every run length is a whole number of 128-byte lines (16 f64) and an
    // odd one, so that a slot's stride never shares a power of two with the
    // cache's set count; at least five lines; and, above that floor, the
    // run's buffers fit the 2 MiB budget with less than two lines to spare.
    let budget = 2usize << 20;
    // Every row carries `settled_frac` beside `n_eff` now
    // (docs/WARMUP-AND-CONVERGENCE.md §3); `withheld_reason` is a byte and
    // does not count against the f64 budget.
    let cases: Vec<(Spec, usize)> = vec![
        (ew_cov_spec(20), 232),     // 230 statistics + n_eff + settled_frac
        (ew_cov_spec(2), 7),        // 5 statistics + n_eff + settled_frac
        (ridge_spec(0), 4),         // pred + resid + n_eff + settled_frac
        (ew_cov_spec(200), 20_302), // wider than the budget's floor allows
    ];
    for (spec, width) in cases {
        let stream = Stream::new(&spec).unwrap();
        let (nm, ns) = (stream.n_models(), stream.n_slots());
        let rows = ChunkOut::run_rows(&spec, nm, ns);
        assert_eq!(rows % 16, 0, "{}: whole lines", spec.name);
        assert_eq!((rows / 16) % 2, 1, "{}: an odd number of them", spec.name);
        assert!(rows >= 80, "{}: at least five lines", spec.name);
        let bytes = rows * width * 8;
        if rows > 80 {
            assert!(bytes <= budget, "{}: {bytes} bytes over budget", spec.name);
            assert!(
                (rows + 32) * width * 8 > budget,
                "{}: {rows} rows leaves more than two lines unused",
                spec.name
            );
        } else {
            assert!(bytes > budget, "{}: at the floor for a reason", spec.name);
        }
    }
    assert_eq!(ChunkOut::run_rows(&ew_cov_spec(20), 1, 230), 1104);
    assert_eq!(ChunkOut::run_rows(&ridge_spec(0), 1, 1), 65_520);
    // A no-target model with one statistic and one instance writes three
    // values a row -- the statistic, `n_eff`, `settled_frac` -- and gets
    // the widest run there is.
    assert_eq!(ChunkOut::run_rows(&ew_cov_spec(2), 1, 1), 87_376);
}

/// A blocked `ewridge` spec (docs/PLAN.md task 71): a block of 16 under a
/// solve every 30 rows, so that rows are held at most points of the
/// stream, closing each group when a larger key arrives.
fn blocked_ridge_spec(block: usize) -> Spec {
    serde_json::from_str(&format!(
        r#"{{
            "name": "m",
            "model": {{"type": "ew_ridge", "ridge": 1e-6, "solve_every": 1e9,
                       "max_rows_between_solves": 30, "gram_block_rows": {block}}},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "half_life": 60.0,
            "gap_cap": 30.0,
            "weight": "w",
            "group": "g",
            "group_close": "monotone",
            "min_weight": 5.0,
            "coef_every": 1
        }}"#
    ))
    .unwrap()
}

/// Rows held by each open group of spec 0, read from the JSON export
/// (the one place a test outside the crate can see the block).
fn held_rows(bank: &Bank) -> Vec<(String, usize)> {
    let doc: serde_json::Value =
        serde_json::from_str(&bank.save_json_string(false).unwrap()).unwrap();
    doc["states"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let key = pair[0].to_string();
            let held = pair[1]["models"][0]["model"]["EwRidge"]["acc"]["grams"]["grams"][0]
                ["pending"]["lam"]
                .as_array()
                .map_or(0, Vec::len);
            (key, held)
        })
        .collect()
}

/// Every matrix read in `EwCov` asserts, in a debug build, that no rows are
/// held; the wiring's one possible mistake -- reading the Gram past a block
/// it has not merged -- is otherwise silent, a stale number rather than a
/// crash. The Python suite runs the release build, where the assertion is
/// compiled out, so this is the debug-build pass over every reader the bank
/// has with rows in flight at each call: `gram`, the closed row (the same
/// builder, reached from inside `fit_predict`), `coef`, `last_row`,
/// `summary`, `describe`, `predict`, the two exports and a load. The
/// readers must also leave the block where it was: the stream after them is
/// the stream without them, to the bit.
#[test]
fn every_bank_reader_survives_a_held_block() {
    // g0's rows first, then g1's: the monotone close finishes g0 when g1's
    // first row arrives.
    let df = make_df(400)
        .sort(
            ["g"],
            SortMultipleOptions::default().with_maintain_order(true),
        )
        .unwrap();
    let cut = 200 + 21;
    let (head, rest) = (df.slice(0, cut), df.slice(cut as i64, df.height() - cut));

    // Both groups are mid-block at the points that matter: g0 at its close
    // (checked on a bank that never closes it) and g1 at the cut.
    let mut open = Bank::new(vec![{
        let mut s = blocked_ridge_spec(16);
        s.group_close = None;
        s
    }])
    .unwrap();
    open.fit_predict(&df.slice(0, 200)).unwrap();
    assert!(
        held_rows(&open).iter().all(|(_, held)| *held > 0),
        "g0 holds nothing at its close: {:?}",
        held_rows(&open)
    );

    let mut bank = Bank::new(vec![blocked_ridge_spec(16)]).unwrap();
    bank.fit_predict(&head).unwrap();
    let held = held_rows(&bank);
    assert!(
        held.iter().any(|(k, h)| k.contains("g1") && *h > 0),
        "g1 holds nothing at the cut: {held:?}"
    );
    let mut untouched = Bank::new(vec![blocked_ridge_spec(16)]).unwrap();
    untouched.fit_predict(&head).unwrap();

    let closed = bank.closed_groups(None, false).unwrap();
    assert_eq!(closed.height(), 1, "g0 closed while holding rows");
    assert_eq!(
        bank.gram(0, None).unwrap().len(),
        1,
        "g1 is the one open group"
    );
    bank.gram(0, Some("g1")).unwrap();
    bank.coef(0, None).unwrap();
    bank.last_row(0, None).unwrap();
    bank.summary(0, None).unwrap();
    bank.describe(0, None).unwrap();
    bank.predict(&rest).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let mut again = Bank::load_bytes(&bytes, Some(bank.specs())).unwrap();
    assert_eq!(held_rows(&again), held, "the held rows travel in the state");

    // None of that moved the block: the read bank, the loaded bank and the
    // bank nobody read all continue identically.
    let unnest = |cols: Vec<Column>| {
        DataFrame::new(rest.height(), cols)
            .unwrap()
            .unnest(["m"], None)
            .unwrap()
    };
    let after_reads = unnest(bank.fit_predict(&rest).unwrap());
    let after_load = unnest(again.fit_predict(&rest).unwrap());
    let after_none = unnest(untouched.fit_predict(&rest).unwrap());
    assert!(
        after_reads.equals_missing(&after_none),
        "a read moved the block"
    );
    assert!(
        after_load.equals_missing(&after_none),
        "the load moved the block"
    );
}

/// A wide `marginal` with its pairs split across the pool (docs/PLAN.md task
/// 126) reads every pair the unsplit one reads, to the bit, fed whole or in
/// chunks, for a count and for `"auto"` -- and the counts are the ones the
/// stream runs with, so the comparison is of a split model.
#[test]
fn a_sharded_marginal_reads_the_unsplit_pairs() {
    let p = 1_500;
    let n = 400;
    let mut s = 77u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut cols: Vec<Column> = Vec::new();
    let mut level = vec![0.0; p];
    let mut xs: Vec<Vec<f64>> = (0..p).map(|_| Vec::with_capacity(n)).collect();
    let (mut y0, mut y1, mut t, mut w) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        for (j, v) in level.iter_mut().enumerate() {
            *v = 0.8 * *v + lcg();
            xs[j].push(*v);
        }
        y0.push(level[0] - level[1] + 0.2 * lcg());
        y1.push((i % 4 != 1).then(|| level[2].abs() + lcg()));
        t.push(i as f64 + if i >= 250 { 1e4 } else { 0.0 });
        w.push(if i % 9 == 4 { 0.0 } else { 1.0 + 0.5 * lcg() });
    }
    for (j, x) in xs.into_iter().enumerate() {
        cols.push(Column::new(format!("x{j}").into(), x));
    }
    cols.push(Column::new("y0".into(), y0));
    cols.push(Column::new("y1".into(), y1));
    cols.push(Column::new("t".into(), t));
    cols.push(Column::new("w".into(), w));
    let df = DataFrame::new(n, cols).unwrap();
    let features: Vec<String> = (0..p).map(|j| format!("\"x{j}\"")).collect();
    let spec = |shards: &str| -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {{"type": "marginal", "lags": [1, 4], "cross_lags": [1],
                "bins": 5, "bin_warm_rows": 60 {shards}}},
                "targets": ["y0", "y1"], "features": [{}], "clock": "t",
                "half_life": 80.0, "gap_cap": 20.0, "weight": "w"}}"#,
            features.join(", ")
        ))
        .unwrap()
    };
    let read = |spec: Spec, chunks: usize| -> DataFrame {
        let mut bank = Bank::new(vec![spec]).unwrap();
        let step = n.div_ceil(chunks);
        for start in (0..n).step_by(step) {
            bank.fit_predict(&df.slice(start as i64, step.min(n - start)))
                .unwrap();
        }
        bank.marginal(0, None).unwrap()
    };
    // The count each spec's stream steps with.
    let count = |spec: &Spec| -> usize {
        let models = online_polars::build_models(spec).unwrap();
        let (_, model) = &models[0];
        online_polars::pool()
            .unwrap()
            .install(|| online_polars::marginal_shards(spec, model))
    };
    let unsplit = spec("");
    assert_eq!(count(&unsplit), 1);
    let want = read(unsplit, 1);
    assert_eq!(want.height(), 2 * p);
    for (shards, expect) in [
        (r#", "shards": 6"#, Some(6)),
        (r#", "shards": "auto""#, None),
    ] {
        let split = spec(shards);
        let resolved = count(&split);
        // A pool of one thread never splits (review 2026-09-26, D9/E4: the
        // assertion failed on a one-core runner with nothing wrong).
        let threads = online_polars::thread_pool_size().unwrap();
        match expect {
            Some(c) => assert_eq!(resolved, c),
            None if threads > 1 => assert!(
                resolved > 1,
                "auto splits lags and bins at 1,500 features on {threads} threads: {resolved}"
            ),
            None => assert_eq!(resolved, 1, "one thread: no split"),
        }
        for chunks in [1, 3] {
            let got = read(split.clone(), chunks);
            assert!(got.equals_missing(&want), "{shards}, {chunks} chunks");
        }
    }
}

/// `shards` is a setting, not state: a bank saved under one count loads
/// under the count of the specs given to `load`, which is the count it then
/// runs with, and keeps the saved one when given none (review 2026-09-26,
/// F3: the load compared the counts and refused the specs).
#[test]
fn a_bank_saved_under_one_shard_count_loads_under_another() {
    let spec = |shards: &str| -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {{"type": "marginal", "lags": [1, 2]{shards}}},
                "targets": ["y"], "features": ["x0", "x1", "x2"], "half_life": 20.0}}"#
        ))
        .unwrap()
    };
    let n = 40;
    let x0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos()).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let y: Vec<f64> = (0..n).map(|i| x0[i] - x2[i] * 0.5).collect();
    let df = DataFrame::new(
        n,
        vec![
            Column::new("x0".into(), x0),
            Column::new("x1".into(), x1),
            Column::new("x2".into(), x2),
            Column::new("y".into(), y),
        ],
    )
    .unwrap();
    let mut bank = Bank::new(vec![spec(r#", "shards": 3"#)]).unwrap();
    bank.fit_predict(&df.slice(0, 25)).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let mut want = Bank::new(vec![spec("")]).unwrap();
    want.fit_predict(&df).unwrap();
    for given in [r#", "shards": "auto""#, "", r#", "shards": 2"#] {
        let expected = vec![spec(given)];
        let mut loaded = Bank::load_bytes(&bytes, Some(&expected)).unwrap();
        // A bank fills its specs' defaults; the count is the one given.
        let mut filled = expected[0].clone();
        filled.fill_defaults();
        assert_eq!(
            loaded.specs()[0],
            filled,
            "the count given is the count run"
        );
        loaded.fit_predict(&df.slice(25, 15)).unwrap();
        assert!(
            loaded
                .marginal(0, None)
                .unwrap()
                .equals_missing(&want.marginal(0, None).unwrap()),
            "{given}"
        );
    }
    let kept = Bank::load_bytes(&bytes, None).unwrap();
    let mut saved = spec(r#", "shards": 3"#);
    saved.fill_defaults();
    assert_eq!(kept.specs()[0], saved, "no specs given: the saved count");
    let other = vec![spec(r#", "cross_lags": [1]"#)];
    assert!(
        Bank::load_bytes(&bytes, Some(&other)).is_err(),
        "a real difference still refuses"
    );
}

/// Docs/PLAN.md task 115 (b): the weight-share cadence is the spec's, set
/// again on restore, as the window's budget is, so a state saved before the
/// rule existed -- its models' config without it, as an explicit
/// `solve_every` also leaves it -- takes it when it loads under a spec that
/// leaves `solve_every` out.
#[test]
fn the_solve_cadence_is_the_specs_on_restore() {
    let spec = |model: &str| -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x0"],
                "half_life": 1e6}}"#
        ))
        .unwrap()
    };
    let explicit = spec(r#"{"type": "ew_ridge", "solve_every": 1e9}"#);
    let default = spec(r#"{"type": "ew_ridge"}"#);
    let saved = Stream::new(&explicit).unwrap().save();
    assert_eq!(
        Stream::restore(&explicit, &saved).unwrap().models[0]
            .1
            .solve_share(),
        None
    );
    let resumed = Stream::restore(&default, &saved).unwrap();
    assert_eq!(
        resumed.models[0].1.solve_share(),
        Some(online_polars::online_core::DEFAULT_SOLVE_SHARE)
    );
}
