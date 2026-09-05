//! `seqtest` in the bank (docs/ENHANCEMENTS.md E42): the column mode's
//! plumbing, and the comparison mode's second phase -- a spec whose targets
//! are residual fields two other specs of the same bank report for the
//! chunk. The arithmetic is pinned in `online-core`; the pytest suite holds
//! the bank to a pure-polars twin. This file pins what the bank adds: the
//! phases, the layouts, the rollback and the refusals.

use online_core::{OnlineModel, SeqTest, SeqTestCfg};
use online_polars::{Bank, Spec, output_fields, output_index};
use polars::prelude::*;

fn ridge(name: &str, halflife: f64, group: bool) -> Spec {
    let g = if group { r#""group": "g","# } else { "" };
    serde_json::from_str(&format!(
        r#"{{
            "name": "{name}",
            "model": {{"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "halflife": {halflife},
            "max_dclock": 30.0,
            {g}
            "min_periods": 5.0
        }}"#
    ))
    .unwrap()
}

fn compare(name: &str, a: &str, b: &str, group: bool) -> Spec {
    let g = if group { r#""group": "g","# } else { "" };
    serde_json::from_str(&format!(
        r#"{{
            "name": "{name}",
            "model": {{"type": "seqtest", "a": "{a}", "b": "{b}"}},
            "targets": ["y"],
            "features": [],
            "clock": "t",
            "max_dclock": 30.0,
            {g}
            "min_periods": 0
        }}"#
    ))
    .unwrap()
}

fn column_mode(name: &str, target: &str, group: bool) -> Spec {
    let g = if group { r#""group": "g","# } else { "" };
    serde_json::from_str(&format!(
        r#"{{
            "name": "{name}",
            "model": {{"type": "seqtest"}},
            "targets": ["{target}"],
            "features": [],
            {g}
            "min_periods": 0
        }}"#
    ))
    .unwrap()
}

/// Two interleaved groups (so the bank lays the chunk out group by group),
/// a null feature and a null target now and then, and a target one ridge's
/// halflife suits better than the other's: `y` follows a slope that drifts.
fn make_df(n: usize) -> DataFrame {
    let mut s = 20260905u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut group, mut t, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![], vec![]);
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let g = i % 2;
        group.push(format!("g{g}"));
        clocks[g] += 1.0 + lcg().abs() * 3.0;
        t.push(clocks[g]);
        let (a, b) = (lcg(), lcg());
        x0.push(if i % 37 == 5 { None } else { Some(a) });
        x1.push(Some(b));
        let slope = 1.0 + (i as f64 / 90.0).sin();
        y.push(if i % 41 == 7 {
            None
        } else {
            Some(slope * a - 0.5 * b + 0.05 * lcg())
        });
    }
    df!("g" => group, "t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

fn run(bank: &mut Bank, df: &DataFrame, n_chunks: usize) -> DataFrame {
    let n = df.height();
    let step = n.div_ceil(n_chunks);
    let mut acc: Option<DataFrame> = None;
    let mut i = 0;
    while i < n {
        let len = step.min(n - i);
        let chunk = df.slice(i as i64, len);
        let out = DataFrame::new(len, bank.fit_predict(&chunk).unwrap()).unwrap();
        match &mut acc {
            None => acc = Some(out),
            Some(a) => {
                a.vstack_mut(&out).unwrap();
            }
        }
        i += len;
    }
    acc.unwrap()
}

fn field(df: &DataFrame, spec: &str, name: &str) -> Series {
    df.column(spec)
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
}

#[test]
fn the_fields_and_their_dtypes() {
    let spec = column_mode("s", "y", false);
    assert_eq!(
        output_fields(&spec),
        ["log_e_pos_y", "log_e_neg_y", "n_pos_y", "n_neg_y", "n_eff"]
    );
    let idx = output_index(&spec);
    let dtypes: Vec<&str> = idx.iter().map(|f| f.dtype.as_str()).collect();
    assert_eq!(dtypes, ["f64", "f64", "i64", "i64", "f64"]);
    assert!(idx.iter().all(|f| f.halflife.is_none() && f.lam.is_none()));
    assert_eq!(idx[0].target.as_deref(), Some("y"));
    assert_eq!(idx[0].kind, "log_e_pos");

    let cmp = compare("c", "a", "b", false);
    assert_eq!(
        output_fields(&cmp),
        ["log_e_a_y", "log_e_b_y", "wins_a_y", "wins_b_y", "n_eff"]
    );
}

#[test]
fn column_mode_is_the_core_model_on_the_signs() {
    // The bank feeds the model the column; nulls bet nothing, every row
    // counts, and the four fields are the four slots read before the row.
    let df = make_df(300);
    let mut bank = Bank::new(vec![column_mode("s", "y", false)]).unwrap();
    let out = run(&mut bank, &df, 4);
    let mut twin = SeqTest::new(SeqTestCfg {
        n_targets: 1,
        min_periods: 0.0,
    })
    .unwrap();
    let ys = df.column("y").unwrap().f64().unwrap();
    let (pos, neg) = (
        field(&out, "s", "log_e_pos_y"),
        field(&out, "s", "log_e_neg_y"),
    );
    let (n_pos, n_neg) = (field(&out, "s", "n_pos_y"), field(&out, "s", "n_neg_y"));
    let n_eff = field(&out, "s", "n_eff");
    assert_eq!(n_pos.dtype(), &DataType::Int64);
    for i in 0..df.height() {
        let step = twin.step(&[], &[ys.get(i)], 1.0, 1.0);
        assert_eq!(pos.f64().unwrap().get(i), Some(step.pred[0]), "row {i}");
        assert_eq!(neg.f64().unwrap().get(i), Some(step.pred[1]), "row {i}");
        assert_eq!(n_pos.i64().unwrap().get(i), Some(step.pred[2] as i64));
        assert_eq!(n_neg.i64().unwrap().get(i), Some(step.pred[3] as i64));
        assert_eq!(n_eff.f64().unwrap().get(i), Some(step.n_eff));
    }
    assert_eq!(n_eff.f64().unwrap().get(299), Some(299.0));
}

/// The comparison, done by hand: the two sides' residuals from a bank
/// without the comparison, `|resid_b| - |resid_a|` as a column, and a
/// column-mode seqtest over it.
fn by_hand(df: &DataFrame, group: bool, n_chunks: usize) -> DataFrame {
    let mut sides = Bank::new(vec![ridge("a", 20.0, group), ridge("b", 200.0, group)]).unwrap();
    let out = run(&mut sides, df, n_chunks);
    let ra = field(&out, "a", "resid_y");
    let rb = field(&out, "b", "resid_y");
    let diff: Vec<Option<f64>> = ra
        .f64()
        .unwrap()
        .iter()
        .zip(rb.f64().unwrap().iter())
        .map(|(x, y)| Some(y?.abs() - x?.abs()))
        .collect();
    let with = df
        .clone()
        .with_column(Column::new("d".into(), diff))
        .unwrap()
        .clone();
    let mut test = Bank::new(vec![column_mode("c", "d", group)]).unwrap();
    run(&mut test, &with, n_chunks)
}

#[test]
fn a_comparison_is_the_column_mode_on_the_residual_difference() {
    let df = make_df(600);
    for group in [false, true] {
        let mut bank = Bank::new(vec![
            compare("c", "a", "b", group),
            ridge("a", 20.0, group),
            ridge("b", 200.0, group),
        ])
        .unwrap();
        let out = run(&mut bank, &df, 5);
        let want = by_hand(&df, group, 5);
        for (got, exp) in [
            ("log_e_a_y", "log_e_pos_d"),
            ("log_e_b_y", "log_e_neg_d"),
            ("wins_a_y", "n_pos_d"),
            ("wins_b_y", "n_neg_d"),
            ("n_eff", "n_eff"),
        ] {
            let g = field(&out, "c", got);
            let w = field(&want, "c", exp);
            assert!(g.equals_missing(&w), "group={group} {got}: {g:?} vs {w:?}");
        }
        // The comparison comes first in the bank and still reads the two
        // sides' output for the same chunk: the columns come back in spec
        // order, the comparison's built last.
        assert_eq!(out.get_column_names(), ["c", "a", "b"]);
        // The shorter halflife tracks the drifting slope: `a` wins more rows
        // and the evidence for it is positive, for the pooled stream and
        // for each group.
        let at = |name: &str, i: usize| field(&out, "c", name).get(i).unwrap().extract::<f64>();
        for i in [598, 599] {
            let (wa, wb) = (at("wins_a_y", i).unwrap(), at("wins_b_y", i).unwrap());
            let (ea, eb) = (at("log_e_a_y", i).unwrap(), at("log_e_b_y", i).unwrap());
            assert!(
                wa > wb && ea > 0.0 && eb <= 0.0,
                "row {i}: {wa} {wb} {ea} {eb}"
            );
        }
    }
}

#[test]
fn a_suffix_picks_the_grid_instance() {
    // `a` is a two-halflife grid; the comparison with `a_suffix = "@h20"`
    // is the comparison of a single-instance spec at halflife 20.
    let df = make_df(400);
    let mut grid = ridge("a", 20.0, true);
    grid.halflife = Some(online_polars::FloatOrList::List(vec![
        online_polars::Num(20.0),
        online_polars::Num(200.0),
    ]));
    let picked: Spec = serde_json::from_str(
        r#"{"name": "c", "model": {"type": "seqtest", "a": "a", "b": "b",
            "a_suffix": "@h20"}, "targets": ["y"], "features": [], "group": "g",
            "clock": "t", "max_dclock": 30.0}"#,
    )
    .unwrap();
    let mut with_grid = Bank::new(vec![grid, ridge("b", 200.0, true), picked]).unwrap();
    let mut plain = Bank::new(vec![
        ridge("a", 20.0, true),
        ridge("b", 200.0, true),
        compare("c", "a", "b", true),
    ])
    .unwrap();
    let (got, want) = (run(&mut with_grid, &df, 3), run(&mut plain, &df, 3));
    assert!(
        got.column("c")
            .unwrap()
            .equals_missing(want.column("c").unwrap())
    );
}

#[test]
fn a_comparison_is_chunk_invariant_over_interleaved_groups() {
    let df = make_df(500);
    let specs = || {
        vec![
            ridge("a", 20.0, true),
            ridge("b", 200.0, true),
            compare("c", "a", "b", true),
        ]
    };
    let one = run(&mut Bank::new(specs()).unwrap(), &df, 1);
    let many = run(&mut Bank::new(specs()).unwrap(), &df, 61);
    let each = run(&mut Bank::new(specs()).unwrap(), &df, 500);
    for other in [&many, &each] {
        for f in ["log_e_a_y", "log_e_b_y", "wins_a_y", "wins_b_y", "n_eff"] {
            assert!(
                field(&one, "c", f).equals_missing(&field(other, "c", f)),
                "{f} differs between chunkings"
            );
        }
    }
    // A comparison over the pooled stream while its sides are per group:
    // the same residuals, one e-process.
    let mut pooled = Bank::new(vec![
        ridge("a", 20.0, true),
        ridge("b", 200.0, true),
        compare("c", "a", "b", false),
    ])
    .unwrap();
    let out = run(&mut pooled, &df, 7);
    let n_eff = field(&out, "c", "n_eff");
    assert_eq!(n_eff.f64().unwrap().get(499), Some(499.0));
}

#[test]
fn scoring_reads_the_state_before_the_chunk() {
    let df = make_df(400);
    let specs = || {
        vec![
            ridge("a", 20.0, true),
            ridge("b", 200.0, true),
            compare("c", "a", "b", true),
        ]
    };
    let mut bank = Bank::new(specs()).unwrap();
    run(&mut bank, &df.slice(0, 300), 3);
    let rest = df.slice(300, 100);
    let scored = DataFrame::new(100, bank.predict(&rest).unwrap()).unwrap();
    let learned = DataFrame::new(100, bank.fit_predict(&rest).unwrap()).unwrap();
    // Every scored row of a group carries the e-values as they stood, which
    // is what `fit_predict` reports on the group's first row.
    for g in ["g0", "g1"] {
        let gs = rest.column("g").unwrap().str().unwrap();
        let first = (0..100).find(|&i| gs.get(i) == Some(g)).unwrap();
        for f in ["log_e_a_y", "log_e_b_y", "wins_a_y", "wins_b_y", "n_eff"] {
            let s = field(&scored, "c", f);
            let l = field(&learned, "c", f);
            for i in (0..100).filter(|&i| gs.get(i) == Some(g)) {
                assert!(
                    s.get(i).unwrap() == l.get(first).unwrap(),
                    "{g} {f} row {i}: scored {:?} vs learned first {:?}",
                    s.get(i),
                    l.get(first)
                );
            }
        }
    }
    // Scoring learned nothing, on either side or in the comparison.
    let again = DataFrame::new(100, bank.predict(&rest).unwrap()).unwrap();
    assert!(!field(&again, "c", "n_eff").equals_missing(&field(&scored, "c", "n_eff")));
    let mut twin = Bank::new(specs()).unwrap();
    run(&mut twin, &df.slice(0, 300), 3);
    let fresh = DataFrame::new(100, twin.predict(&rest).unwrap()).unwrap();
    assert!(fresh.equals_missing(&scored));
}

#[test]
fn a_refused_chunk_updates_neither_phase() {
    let strict = |mut s: Spec| {
        s.on_clock_reset = online_core::OnClockReset::Error;
        s
    };
    let specs = || {
        vec![
            strict(ridge("a", 20.0, true)),
            strict(ridge("b", 200.0, true)),
            strict(compare("c", "a", "b", true)),
        ]
    };
    let df = make_df(200);
    let (first, good) = (df.slice(0, 100), df.slice(100, 100));
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
    assert!(err.contains("goes backwards"), "{err}");
    assert_eq!(
        bank.save_bytes().unwrap(),
        before,
        "a refused chunk changed the bank"
    );
    let out = bank.fit_predict(&good).unwrap();
    let mut clean = Bank::new(specs()).unwrap();
    clean.fit_predict(&first).unwrap();
    assert_eq!(out, clean.fit_predict(&good).unwrap());
}

#[test]
fn save_and_load_mid_stream_continue_identically() {
    let df = make_df(300);
    let specs = || {
        vec![
            ridge("a", 20.0, true),
            ridge("b", 200.0, true),
            compare("c", "a", "b", true),
        ]
    };
    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&df.slice(0, 150)).unwrap();
    let bytes = bank.save_bytes().unwrap();
    let mut restored = Bank::load_bytes(&bytes, Some(&specs())).unwrap();
    let rest = df.slice(150, 150);
    assert_eq!(
        bank.fit_predict(&rest).unwrap(),
        restored.fit_predict(&rest).unwrap()
    );
}

#[test]
fn the_refusals_name_the_problem() {
    let err = |specs: Vec<Spec>| match Bank::new(specs) {
        Ok(_) => panic!("the bank accepted the specs"),
        Err(e) => e,
    };
    let e = err(vec![ridge("a", 20.0, false), compare("c", "a", "b", false)]);
    assert!(e.contains("b = \"b\" is not a spec of this bank"), "{e}");
    assert!(e.contains("[\"a\", \"c\"]"), "{e}");

    let e = err(vec![
        ridge("a", 20.0, false),
        column_mode("s", "y", false),
        compare("c", "a", "s", false),
    ]);
    assert!(e.contains("b = \"s\" is itself a seqtest"), "{e}");

    let mut wrong = compare("c", "a", "b", false);
    wrong.targets = vec!["z".into()];
    let e = err(vec![
        ridge("a", 20.0, false),
        ridge("b", 200.0, false),
        wrong,
    ]);
    assert!(
        e.contains("target \"z\" names no residual of a = \"a\""),
        "{e}"
    );
    assert!(e.contains("no field \"resid_z\""), "{e}");
    assert!(e.contains("[\"resid_y\"]"), "{e}");

    // A grid's residual fields carry the suffix, and so must the target.
    let mut grid = ridge("a", 20.0, false);
    grid.halflife = Some(online_polars::FloatOrList::List(vec![
        online_polars::Num(20.0),
        online_polars::Num(200.0),
    ]));
    let e = err(vec![
        grid.clone(),
        ridge("b", 200.0, false),
        compare("c", "a", "b", false),
    ]);
    assert!(e.contains("[\"resid_y@h20\", \"resid_y@h200\"]"), "{e}");
    let mut suffixed = compare("c", "a", "b", false);
    suffixed.targets = vec!["y@h20".into()];
    let e = err(vec![grid.clone(), ridge("b", 200.0, false), suffixed]);
    assert!(e.contains("names no residual of b = \"b\""), "{e}");
    // `a_suffix` picks the grid instance, and the target stays `y`.
    let picked = |suffix: &str| -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "c", "model": {{"type": "seqtest", "a": "a", "b": "b",
                "a_suffix": "{suffix}"}}, "targets": ["y"], "features": []}}"#
        ))
        .unwrap()
    };
    assert!(Bank::new(vec![grid.clone(), ridge("b", 200.0, false), picked("@h20")]).is_ok());
    let e = err(vec![grid.clone(), ridge("b", 200.0, false), picked("@h50")]);
    assert!(e.contains("no field \"resid_y@h50\""), "{e}");
    assert!(e.contains("a_suffix the grid part"), "{e}");
    let e = picked("@h20")
        .clone()
        .validate()
        .map(|_| String::new())
        .unwrap_or_else(|e| e);
    assert!(e.is_empty(), "{e}");
    let orphan: Spec = serde_json::from_str(
        r#"{"name": "c", "model": {"type": "seqtest", "a_suffix": "@h20"}, "targets": ["y"],
            "features": []}"#,
    )
    .unwrap();
    let e = orphan.validate().unwrap_err();
    assert!(
        e.contains("a_suffix/b_suffix pick a side's grid instance"),
        "{e}"
    );

    let one_sided: Result<Spec, _> = serde_json::from_str(
        r#"{"name": "c", "model": {"type": "seqtest", "a": "a"}, "targets": ["y"],
            "features": []}"#,
    );
    let e = one_sided.unwrap().validate().unwrap_err();
    assert!(e.contains("a and b go together (got a only)"), "{e}");
    let e = compare("c", "a", "a", false).validate().unwrap_err();
    assert!(e.contains("both \"a\" with the same suffix"), "{e}");
    // Two instances of one grid are a comparison like any other.
    let two_instances: Spec = serde_json::from_str(
        r#"{"name": "c", "model": {"type": "seqtest", "a": "a", "b": "a",
            "a_suffix": "@h20", "b_suffix": "@h400"}, "targets": ["y"], "features": []}"#,
    )
    .unwrap();
    two_instances.validate().unwrap();
    let e = compare("c", "c", "b", false).validate().unwrap_err();
    assert!(e.contains("name the spec itself"), "{e}");
    let mut weighted = column_mode("s", "y", false);
    weighted.weight = Some("w".into());
    let e = weighted.validate().unwrap_err();
    assert!(e.contains("weight does not apply to seqtest"), "{e}");
    let mut decayed = column_mode("s", "y", false);
    decayed.halflife = Some(online_polars::FloatOrList::Float(online_polars::Num(50.0)));
    let e = decayed.validate().unwrap_err();
    assert!(e.contains("halflife/lam do not apply to seqtest"), "{e}");
    let mut featured = column_mode("s", "y", false);
    featured.features = vec!["x0".into()];
    let e = featured.validate().unwrap_err();
    assert!(e.contains("seqtest takes no features (got 1)"), "{e}");
    let mut sigma = column_mode("s", "y", false);
    sigma.emit_sigma = true;
    let e = sigma.validate().unwrap_err();
    assert!(e.contains("emit_sigma does not apply to seqtest"), "{e}");
}
