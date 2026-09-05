//! `Bank::last_row` (docs/PLAN.md task 34): the output struct on the last
//! row each stream learned from, kept with the state. The contract is that
//! it *is* the output frame's row -- every field, the bit included -- and
//! that it survives the file, ignores `predict`, and steps back over the
//! skipped rows a chunk may end in.

use online_polars::{Bank, Spec, Stream};
use polars::prelude::*;

/// Every diagnostic there is, on a two-halflife grid over two groups, so
/// every buffer of the row is exercised: `sigma`, `resid_z`, the conformal
/// interval, the metrics, a residual quantile, autocorrelation, drift,
/// `n_eff`, `coef`, and the selected and averaged predictions `assemble`
/// derives from the row itself.
fn rich_spec() -> Spec {
    serde_json::from_str(
        r#"{
            "name": "m",
            "model": {"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "halflife": [10.0, 20.0],
            "max_dclock": 30.0,
            "weight": "w",
            "group": "g",
            "min_periods": 5.0,
            "emit_sigma": true,
            "emit_resid_z": true,
            "emit_selected": true,
            "emit_averaged": true,
            "emit_metrics": true,
            "conformal": 0.9,
            "resid_quantiles": [0.5],
            "emit_autocorr": true,
            "emit_drift": true
        }"#,
    )
    .unwrap()
}

/// A `lasso`, for `lam_selected`, and an `ew_cov`, for a model whose row is
/// statistics with no residual at all.
fn other_specs() -> Vec<Spec> {
    let lasso = r#"{
        "name": "l",
        "model": {"type": "lasso", "lasso_path": [0.1, 0.0]},
        "targets": ["y"], "features": ["x0", "x1"], "clock": "t",
        "halflife": 20.0, "max_dclock": 30.0, "group": "g"
    }"#;
    let cov = r#"{
        "name": "c",
        "model": {"type": "ew_cov"},
        "targets": ["x0"], "features": ["x0", "x1", "y"], "clock": "t",
        "halflife": 20.0, "max_dclock": 30.0
    }"#;
    vec![
        serde_json::from_str(lasso).unwrap(),
        serde_json::from_str(cov).unwrap(),
    ]
}

/// Two interleaved groups with a null feature every 17th row (skipped) and a
/// null target every 23rd (predict-only), as `tests/bank.rs` makes them.
fn make_df(n: usize) -> DataFrame {
    let mut s = 4321u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut group, mut t, mut x0, mut x1, mut y, mut w) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let g = i % 2;
        group.push(format!("g{g}"));
        clocks[g] += 1.0 + lcg().abs() * 5.0;
        t.push(clocks[g]);
        let (a, b) = (lcg(), lcg());
        x0.push((i % 17 != 5).then_some(a));
        x1.push(Some(b));
        y.push((i % 23 != 7).then(|| if g == 0 { 2.0 * a - b } else { -a } + 0.01 * lcg()));
        w.push(0.5 + lcg().abs());
    }
    df!("g" => group, "t" => t, "x0" => x0, "x1" => x1, "y" => y, "w" => w).unwrap()
}

/// `df` through `bank` in `n_chunks` near-equal chunks, stacked.
fn feed(bank: &mut Bank, df: &DataFrame, n_chunks: usize) -> DataFrame {
    let n = df.height();
    let step = n.div_ceil(n_chunks);
    let mut acc: Option<DataFrame> = None;
    let mut i = 0;
    while i < n {
        let chunk = df.slice(i as i64, step.min(n - i));
        let out = DataFrame::new(chunk.height(), bank.fit_predict(&chunk).unwrap()).unwrap();
        match &mut acc {
            Some(a) => {
                a.vstack_mut(&out).unwrap();
            }
            None => acc = Some(out),
        }
        i += chunk.height();
    }
    acc.unwrap()
}

/// `Bank::last_row` of one spec as a frame: `group` then the struct's
/// fields, one row per group.
fn last_rows(bank: &Bank, spec: usize, name: &str) -> DataFrame {
    let (keys, col) = bank.last_row(spec, None).unwrap();
    let groups: Vec<Option<&str>> = keys.iter().map(|k| k.as_str()).collect();
    DataFrame::new(keys.len(), vec![Column::new("group".into(), groups), col])
        .unwrap()
        .unnest([name], None)
        .unwrap()
}

/// Row `i` of `out`'s struct `name`, unnested, with `group` in front.
fn output_row(out: &DataFrame, name: &str, group: Option<&str>, i: usize) -> DataFrame {
    let mut row = out
        .select([name])
        .unwrap()
        .slice(i as i64, 1)
        .unnest([name], None)
        .unwrap();
    row.insert_column(0, Column::new("group".into(), [group]))
        .unwrap();
    row
}

/// The last row of `out` that the stream of `group` learned from: the last
/// row of that group whose `n_eff` field is not null (a skipped row has
/// every field null).
fn last_learned(df: &DataFrame, out: &DataFrame, name: &str, n_eff: &str, group: &str) -> usize {
    let gs = df.column("g").unwrap().str().unwrap();
    let st = out.column(name).unwrap().struct_().unwrap();
    let n_eff = st.field_by_name(n_eff).unwrap();
    (0..df.height())
        .rev()
        .find(|&i| gs.get(i) == Some(group) && n_eff.get(i).unwrap() != AnyValue::Null)
        .unwrap()
}

fn assert_same(want: &DataFrame, got: &DataFrame, what: &str) {
    assert!(want.equals_missing(got), "{what}:\nwant {want}\ngot  {got}");
}

#[test]
fn last_row_is_the_output_s_last_learned_row_per_group() {
    let df = make_df(400);
    let mut specs = vec![rich_spec()];
    specs.extend(other_specs());
    let mut bank = Bank::new(specs.clone()).unwrap();
    let out = feed(&mut bank, &df, 7);

    // The grouped specs: the frame's last learned row of each group, whole.
    for (si, name, n_eff) in [(0, "m", "n_eff@h10"), (1, "l", "n_eff")] {
        let rows = last_rows(&bank, si, name);
        assert_eq!(rows.height(), 2, "{name}: one row per group");
        for g in ["g0", "g1"] {
            let i = last_learned(&df, &out, name, n_eff, g);
            let want = output_row(&out, name, Some(g), i);
            let got = rows
                .filter(&rows.column("group").unwrap().str().unwrap().equal(g))
                .unwrap();
            assert_same(&want, &got, &format!("{name} / {g} (row {i})"));
        }
    }
    // The ungrouped `ew_cov`: the frame's last row, statistics and `n_eff`.
    let i = (0..400)
        .rev()
        .find(|&i| {
            out.column("c")
                .unwrap()
                .struct_()
                .unwrap()
                .field_by_name("n_eff")
                .unwrap()
                .get(i)
                .unwrap()
                != AnyValue::Null
        })
        .unwrap();
    assert_same(
        &output_row(&out, "c", Some(""), i),
        &last_rows(&bank, 2, "c"),
        "c",
    );

    // Chunk invariance: one chunk or seven, the same row, `coef` aside --
    // the frame reports `coef` on a chunk's last row, so whether the group's
    // last learned row carries it is the chunking's to decide, as it is in
    // the frame (docs/PLAN.md §3).
    let mut one = Bank::new(specs.clone()).unwrap();
    feed(&mut one, &df, 1);
    let no_coef = |d: &DataFrame| {
        let keep: Vec<String> = d
            .get_column_names()
            .iter()
            .filter(|c| !c.starts_with("coef"))
            .map(|c| c.to_string())
            .collect();
        d.select(keep).unwrap()
    };
    for (si, name) in [(0, "m"), (1, "l"), (2, "c")] {
        assert_same(
            &no_coef(&last_rows(&one, si, name)),
            &no_coef(&last_rows(&bank, si, name)),
            &format!("{name}: 1 chunk vs 7"),
        );
    }

    // The file carries it, to the bit.
    let restored = Bank::load_bytes(&bank.save_bytes().unwrap(), Some(&specs)).unwrap();
    for (si, name) in [(0, "m"), (1, "l"), (2, "c")] {
        assert_same(
            &last_rows(&bank, si, name),
            &last_rows(&restored, si, name),
            &format!("{name}: after save/load"),
        );
    }

    // Narrowed to a group, a group never seen, a spec out of range.
    let (keys, col) = bank.last_row(0, Some("g1")).unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(col.len(), 1);
    let (keys, col) = bank.last_row(0, Some("zzz")).unwrap();
    assert!(keys.is_empty());
    assert_eq!(col.len(), 0);
    assert_eq!(col.name(), "m");
    assert!(
        bank.last_row(3, None)
            .unwrap_err()
            .contains("spec index 3 out of range")
    );
}

#[test]
fn predict_does_not_move_it_and_a_fresh_group_has_none() {
    let df = make_df(200);
    let mut bank = Bank::new(vec![rich_spec()]).unwrap();
    feed(&mut bank, &df.slice(0, 120), 3);
    let before = last_rows(&bank, 0, "m");
    bank.predict(&df.slice(120, 80)).unwrap();
    assert_same(&before, &last_rows(&bank, 0, "m"), "after predict");

    // A group whose every row so far was skipped has a stream (it is listed
    // by `groups`) and no learned row: a row of nulls, `group` aside.
    let skipped = df
        .slice(0, 6)
        .lazy()
        .with_columns([
            lit("g9").alias("g"),
            lit(NULL).cast(DataType::Float64).alias("x0"),
        ])
        .collect()
        .unwrap();
    bank.fit_predict(&skipped).unwrap();
    let rows = last_rows(&bank, 0, "m");
    assert_eq!(rows.height(), 3);
    let g9 = rows
        .filter(&rows.column("group").unwrap().str().unwrap().equal("g9"))
        .unwrap();
    assert_eq!(g9.height(), 1);
    for c in g9.columns().iter().skip(1) {
        assert_eq!(c.null_count(), 1, "{} should be null", c.name());
    }
    // And the other two are as they were.
    let others = rows
        .filter(&rows.column("group").unwrap().str().unwrap().not_equal("g9"))
        .unwrap();
    assert_same(
        &before,
        &others,
        "the groups the skipped chunk did not touch",
    );

    // A bank that has seen nothing has no rows to report and no error.
    let empty = Bank::new(vec![rich_spec()]).unwrap();
    let (keys, col) = empty.last_row(0, None).unwrap();
    assert!(keys.is_empty());
    assert_eq!(col.len(), 0);
}

#[test]
fn a_chunk_that_ends_in_skipped_rows_keeps_the_row_before_them() {
    let df = make_df(200);
    let mut bank = Bank::new(vec![rich_spec()]).unwrap();
    let out = feed(&mut bank, &df, 1);
    let i = last_learned(&df, &out, "m", "n_eff@h10", "g0");
    // Skip g0's next four rows: null `x0` skips a row; the clock still moves.
    let tail = df
        .slice(200 - 40, 40)
        .lazy()
        .with_columns([when(col("g").eq(lit("g0")))
            .then(lit(NULL).cast(DataType::Float64))
            .otherwise(col("x0"))
            .alias("x0")])
        .with_columns([(col("t") + lit(1000.0)).alias("t")])
        .collect()
        .unwrap();
    let out2 = DataFrame::new(tail.height(), bank.fit_predict(&tail).unwrap()).unwrap();
    let rows = last_rows(&bank, 0, "m");
    // g0 learned nothing from the chunk: its row is still row `i` of the
    // first frame.
    let g0 = rows
        .filter(&rows.column("group").unwrap().str().unwrap().equal("g0"))
        .unwrap();
    assert_same(
        &output_row(&out, "m", Some("g0"), i),
        &g0,
        "g0 kept its row",
    );
    // g1 did: its row is the second frame's last learned one.
    let j = last_learned(&tail, &out2, "m", "n_eff@h10", "g1");
    let g1 = rows
        .filter(&rows.column("group").unwrap().str().unwrap().equal("g1"))
        .unwrap();
    assert_same(&output_row(&out2, "m", Some("g1"), j), &g1, "g1 moved on");
}

#[test]
fn a_saved_row_of_the_wrong_shape_is_refused() {
    let spec = rich_spec();
    let mut stream = Stream::new(&spec).unwrap();
    assert!(stream.last_row().is_none());
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    bank.fit_predict(&make_df(60)).unwrap();
    // A stream's saved state, through the bank's file.
    let restored = Bank::load_bytes(&bank.save_bytes().unwrap(), None).unwrap();
    let (_, col) = restored.last_row(0, None).unwrap();
    assert_eq!(col.len(), 2);

    // The same state with a row that is not this spec's shape.
    let saved = {
        let mut s = stream.save();
        s.last_row = None;
        s
    };
    stream = Stream::restore(&spec, &saved).unwrap();
    assert!(stream.last_row().is_none());
    let mut bad = saved.clone();
    bad.last_row = Some(online_polars::LastRow {
        pred: vec![0.0; 3],
        ..Default::default()
    });
    let err = Stream::restore(&spec, &bad).err().expect("refused");
    assert!(err.contains("wrong shape"), "{err}");
}
