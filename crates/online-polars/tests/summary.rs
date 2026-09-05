//! `Bank::summary` and `Bank::describe` (docs/PLAN.md task 35): what each
//! stream has been fed, kept with the state. The contract is that the
//! numbers are the frame's -- counted and averaged over the rows routed to
//! the group, undecayed, with the models' own notion of a usable value --
//! that chunking cannot move a bit of them, that they survive the file and
//! ignore `predict`, and that a file whose summary is not its spec's is
//! refused rather than reported.

use online_polars::{Bank, Spec, Stream};
use polars::prelude::*;

/// The input bound the models apply (`online_core::INPUT_BOUND`): a
/// magnitude beyond it is "missing", as a NaN or an infinity is.
fn usable(v: f64) -> bool {
    v.is_finite() && v.abs() <= online_polars::online_core::INPUT_BOUND
}

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap()
}

/// The specs every test runs: a grouped, weighted, sessioned ridge whose
/// session changes reset it (so `resets == session_changes`); a ridge that
/// resets on a backwards clock (so `resets == clock_backwards`); an `ew_cov`
/// (unsupervised: no target rows in `describe`); an `ew_class` on a string
/// label (counts only for the label); a ridge on the row-count clock (no
/// clock range); and a comparison, whose one "target" is the comparison's
/// own difference of residuals.
fn specs() -> Vec<Spec> {
    vec![
        spec(
            r#"{"name": "m", "model": {"type": "ew_ridge", "ridge": 1e-6},
                "targets": ["y"], "features": ["x0", "x1"], "clock": "t",
                "session": "sess", "session_gap": "reset", "weight": "w",
                "group": "g", "halflife": 10.0, "max_dclock": 30.0}"#,
        ),
        spec(
            r#"{"name": "r", "model": {"type": "ew_ridge", "ridge": 1e-6},
                "targets": ["y"], "features": ["x0", "x1"], "clock": "t",
                "on_clock_reset": "reset_state", "group": "g",
                "halflife": 10.0, "max_dclock": 30.0}"#,
        ),
        spec(
            r#"{"name": "c", "model": {"type": "ew_cov"},
                "targets": ["x0"], "features": ["x0", "x1", "y"], "clock": "t",
                "group": "g", "halflife": 10.0, "max_dclock": 30.0}"#,
        ),
        spec(
            r#"{"name": "k", "model": {"type": "ew_class", "classes": ["up", "down"],
                "precision_prior": 1.0},
                "targets": ["lbl"], "features": ["x0", "x1"], "clock": "t",
                "group": "g", "halflife": 10.0, "max_dclock": 30.0}"#,
        ),
        spec(
            r#"{"name": "n", "model": {"type": "ew_ridge", "ridge": 1e-6},
                "targets": ["y"], "features": ["x1"], "group": "g",
                "halflife": 10.0}"#,
        ),
        spec(
            r#"{"name": "s", "model": {"type": "seqtest", "a": "m", "b": "r"},
                "targets": ["y"], "features": [], "group": "g", "clock": "t",
                "max_dclock": 30.0}"#,
        ),
    ]
}

/// Two interleaved groups. `x0` has every kind of missing value -- null,
/// NaN, infinity, a magnitude beyond the bound -- `y` and `lbl` have nulls,
/// `w` has zeros (accepted, weight nothing) and nulls (skipped), the clock
/// steps back now and then, and the session changes twice per group with
/// the clock jumping forward as it does.
fn make_df(n: usize) -> DataFrame {
    let mut s = 97u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut group: Vec<String> = Vec::new();
    let mut t: Vec<f64> = Vec::new();
    let mut sess: Vec<String> = Vec::new();
    let mut x0: Vec<Option<f64>> = Vec::new();
    let mut x1: Vec<f64> = Vec::new();
    let mut y: Vec<Option<f64>> = Vec::new();
    let mut w: Vec<Option<f64>> = Vec::new();
    let mut lbl: Vec<Option<String>> = Vec::new();
    let mut clocks = [0.0f64, 0.0];
    let mut counts = [0usize, 0];
    for i in 0..n {
        let g = i % 2;
        let k = counts[g];
        counts[g] += 1;
        group.push(format!("g{g}"));
        let session = k * 3 / (n / 2).max(1);
        if k > 0 && k * 3 % (n / 2).max(1) < 3 && session > 0 {
            clocks[g] += 500.0; // a new session starts well after the last
        } else if k % 41 == 40 {
            clocks[g] -= 3.0; // a step back within the session
        } else {
            clocks[g] += 1.0 + lcg().abs() * 5.0;
        }
        t.push(clocks[g]);
        sess.push(format!("s{session}"));
        let (a, b) = (lcg(), lcg());
        x0.push(match k % 100 {
            17 => None,
            29 => Some(f64::NAN),
            31 => Some(f64::INFINITY),
            37 => Some(1e300),
            _ => Some(a),
        });
        x1.push(b);
        y.push((k % 23 != 7).then(|| if g == 0 { 2.0 * a - b } else { -a } + 0.01 * lcg()));
        w.push(match k % 100 {
            13 | 63 => Some(0.0),
            19 => None,
            _ => Some(0.5 + lcg().abs()),
        });
        lbl.push((k % 11 != 3).then(|| if a + b > 0.0 { "up" } else { "down" }.to_string()));
    }
    df!("g" => group, "t" => t, "sess" => sess, "x0" => x0, "x1" => x1,
        "y" => y, "w" => w, "lbl" => lbl)
    .unwrap()
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

fn f64s(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
    df.column(name).unwrap().f64().unwrap().iter().collect()
}

fn u64s(df: &DataFrame, name: &str) -> Vec<Option<u64>> {
    df.column(name).unwrap().u64().unwrap().iter().collect()
}

fn strs(df: &DataFrame, name: &str) -> Vec<Option<String>> {
    df.column(name)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|s| s.map(str::to_string))
        .collect()
}

/// A frame's single row for `group`.
fn row_of(df: &DataFrame, group: &str) -> DataFrame {
    let got = df
        .filter(&df.column("group").unwrap().str().unwrap().equal(group))
        .unwrap();
    assert_eq!(got.height(), 1, "one row for {group}:\n{df}");
    got
}

fn assert_same(want: &DataFrame, got: &DataFrame, what: &str) {
    assert!(want.equals_missing(got), "{what}:\nwant {want}\ngot  {got}");
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1e-300)
}

/// What one group of one spec should report, worked out from the frame
/// the plain way: a pass over the group's rows in frame order.
struct Oracle {
    fed: u64,
    processed: u64,
    learned: u64,
    zero_weight: u64,
    weight_sum: f64,
    clock: Option<(f64, f64)>,
    session_changes: u64,
    backwards: u64,
    resets: u64,
    /// Per input column, features then targets then weight.
    columns: Vec<ColumnOracle>,
}

/// Count, nulls, mean, std, min, max of one column's usable values.
type ColumnOracle = (u64, u64, Option<f64>, Option<f64>, Option<f64>, Option<f64>);

fn oracle(df: &DataFrame, s: &Spec, group: &str, reset_on: &str) -> Oracle {
    let gs = strs(df, "g");
    let rows: Vec<usize> = (0..df.height())
        .filter(|&i| gs[i].as_deref() == Some(group))
        .collect();
    let col = |name: &str| -> Vec<f64> {
        if name == "lbl" {
            // The class index, as the bank extracts a label.
            strs(df, name)
                .iter()
                .map(|v| match v.as_deref() {
                    Some("up") => 0.0,
                    Some("down") => 1.0,
                    _ => f64::NAN,
                })
                .collect()
        } else {
            f64s(df, name)
                .iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect()
        }
    };
    let features: Vec<Vec<f64>> = s.features.iter().map(|c| col(c)).collect();
    let targets: Vec<Vec<f64>> = s.targets.iter().map(|c| col(c)).collect();
    let weight: Option<Vec<f64>> = s.weight.as_deref().map(col);
    let clock: Option<Vec<f64>> = s.clock.as_deref().map(col);
    let sess = strs(df, "sess");
    let mut o = Oracle {
        fed: 0,
        processed: 0,
        learned: 0,
        zero_weight: 0,
        weight_sum: 0.0,
        clock: None,
        session_changes: 0,
        backwards: 0,
        resets: 0,
        columns: Vec::new(),
    };
    let (mut prev_clock, mut prev_sess): (Option<f64>, Option<&str>) = (None, None);
    for &i in &rows {
        o.fed += 1;
        let accept = features.iter().all(|f| usable(f[i]))
            && weight.as_ref().map(|w| usable(w[i])).unwrap_or(true);
        let session_changed = match (s.session.as_deref(), prev_sess) {
            (Some(_), Some(p)) => sess[i].as_deref() != Some(p),
            _ => false,
        };
        let below = match (clock.as_ref().map(|c| c[i]), prev_clock) {
            (Some(c), Some(p)) => c < p,
            _ => false,
        };
        let backwards = below && !session_changed;
        o.session_changes += u64::from(session_changed);
        o.backwards += u64::from(backwards);
        o.resets += u64::from(match reset_on {
            "session" => session_changed,
            "backwards" => backwards,
            _ => false,
        });
        if let Some(c) = clock.as_ref().map(|c| c[i]) {
            o.clock = Some(o.clock.map_or((c, c), |(lo, hi)| (lo.min(c), hi.max(c))));
            prev_clock = Some(c);
        }
        if s.session.is_some() {
            prev_sess = sess[i].as_deref();
        }
        if accept {
            o.processed += 1;
            let w = weight.as_ref().map(|w| w[i]).unwrap_or(1.0);
            o.weight_sum += w;
            if w == 0.0 {
                o.zero_weight += 1;
            } else if targets.is_empty() || targets.iter().any(|t| usable(t[i])) {
                o.learned += 1;
            }
        }
    }
    for c in features.iter().chain(&targets).chain(&weight) {
        let ok: Vec<f64> = rows.iter().map(|&i| c[i]).filter(|v| usable(*v)).collect();
        let n = ok.len() as f64;
        let stats = if ok.is_empty() {
            (None, None, None, None)
        } else {
            let mean = ok.iter().sum::<f64>() / n;
            let m2 = ok.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
            let std = (ok.len() >= 2).then(|| (m2 / (n - 1.0)).sqrt());
            let min = ok.iter().copied().fold(f64::INFINITY, f64::min);
            let max = ok.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (Some(mean), std, Some(min), Some(max))
        };
        o.columns.push((
            ok.len() as u64,
            rows.len() as u64 - ok.len() as u64,
            stats.0,
            stats.1,
            stats.2,
            stats.3,
        ));
    }
    o
}

#[test]
fn summary_and_describe_are_the_frame_s_numbers() {
    let df = make_df(700);
    let specs = specs();
    let mut bank = Bank::new(specs.clone()).unwrap();
    let out = feed(&mut bank, &df, 5);

    // Every group actually met the events the fixture promises.
    let m_all = bank.summary(0, None).unwrap();
    assert!(
        u64s(&m_all, "session_changes")
            .iter()
            .all(|v| v == &Some(2))
    );
    assert!(
        u64s(&m_all, "clock_backwards")
            .iter()
            .all(|v| v.unwrap() >= 5)
    );
    assert!(
        u64s(&m_all, "rows_zero_weight")
            .iter()
            .all(|v| v.unwrap() >= 5)
    );
    assert!(
        u64s(&m_all, "rows_skipped")
            .iter()
            .all(|v| v.unwrap() >= 10)
    );

    for (si, s) in specs.iter().enumerate() {
        let reset_on = match s.name.as_str() {
            "m" => "session",
            "r" => "backwards",
            _ => "never",
        };
        let summary = bank.summary(si, None).unwrap();
        assert_eq!(
            summary.get_column_names(),
            [
                "group",
                "rows_fed",
                "rows_processed",
                "rows_skipped",
                "rows_learned",
                "rows_zero_weight",
                "weight_sum",
                "clock_min",
                "clock_max",
                "last_clock",
                "session_changes",
                "clock_backwards",
                "resets"
            ]
        );
        assert_eq!(summary.height(), 2, "{}: one row per group", s.name);
        let describe = bank.describe(si, None).unwrap();
        assert_eq!(
            describe.get_column_names(),
            [
                "group",
                "column",
                "role",
                "count",
                "null_count",
                "mean",
                "std",
                "min",
                "max"
            ]
        );
        for g in ["g0", "g1"] {
            let want = if s.name == "s" {
                // The comparison's target is not a column of the frame: it
                // is `|resid_b| - |resid_a|`, worked out from the two
                // sides' output, null where either side did not predict.
                let side = |name: &str| -> Vec<f64> {
                    out.column(name)
                        .unwrap()
                        .struct_()
                        .unwrap()
                        .field_by_name("resid_y")
                        .unwrap()
                        .f64()
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap_or(f64::NAN))
                        .collect()
                };
                let (a, b) = (side("m"), side("r"));
                let diff: Vec<f64> = a.iter().zip(&b).map(|(x, y)| y.abs() - x.abs()).collect();
                let df2 = df
                    .clone()
                    .with_column(Column::new("y".into(), diff))
                    .unwrap()
                    .clone();
                oracle(&df2, s, g, reset_on)
            } else {
                oracle(&df, s, g, reset_on)
            };
            let row = row_of(&summary, g);
            let what = format!("{} / {g}", s.name);
            let u = |c: &str| u64s(&row, c)[0];
            let f = |c: &str| f64s(&row, c)[0];
            assert_eq!(u("rows_fed"), Some(want.fed), "{what}: rows_fed");
            assert_eq!(
                u("rows_processed"),
                Some(want.processed),
                "{what}: processed"
            );
            assert_eq!(
                u("rows_skipped"),
                Some(want.fed - want.processed),
                "{what}: skipped"
            );
            assert_eq!(u("rows_learned"), Some(want.learned), "{what}: learned");
            assert_eq!(
                u("rows_zero_weight"),
                Some(want.zero_weight),
                "{what}: zero weight"
            );
            assert!(
                close(f("weight_sum").unwrap(), want.weight_sum),
                "{what}: weight_sum {:?} vs {}",
                f("weight_sum"),
                want.weight_sum
            );
            assert_eq!(f("clock_min"), want.clock.map(|c| c.0), "{what}: clock_min");
            assert_eq!(f("clock_max"), want.clock.map(|c| c.1), "{what}: clock_max");
            assert_eq!(
                u("session_changes"),
                Some(want.session_changes),
                "{what}: session_changes"
            );
            assert_eq!(
                u("clock_backwards"),
                Some(want.backwards),
                "{what}: backwards"
            );
            assert_eq!(u("resets"), Some(want.resets), "{what}: resets");
            // `last_clock` is what `groups` reports, and null on a row count.
            let groups = &bank.groups()[si];
            let (_, rows_seen, last_clock) = groups
                .iter()
                .find(|(k, _, _)| k.as_str() == Some(g))
                .unwrap();
            assert_eq!(f("last_clock"), *last_clock, "{what}: last_clock");
            assert_eq!(u("rows_processed"), Some(*rows_seen), "{what}: rows_seen");

            // The columns, in spec order: features, targets, weight -- with
            // an unsupervised model's mirrored target left out and a label
            // column's numbers withheld.
            let d = describe
                .filter(&describe.column("group").unwrap().str().unwrap().equal(g))
                .unwrap();
            let mut expect: Vec<(&str, &str, usize)> = Vec::new(); // (column, role, oracle index)
            let (nf, nt) = (s.features.len(), s.targets.len());
            for (j, c) in s.features.iter().enumerate() {
                expect.push((c, "feature", j));
            }
            if !s.model.is_unsupervised() {
                for (j, c) in s.targets.iter().enumerate() {
                    expect.push((c, "target", nf + j));
                }
            }
            if let Some(w) = &s.weight {
                expect.push((w, "weight", nf + nt));
            }
            assert_eq!(d.height(), expect.len(), "{what}: describe rows\n{d}");
            let (cols, roles) = (strs(&d, "column"), strs(&d, "role"));
            let (count, nulls) = (u64s(&d, "count"), u64s(&d, "null_count"));
            let (mean, std, min, max) = (
                f64s(&d, "mean"),
                f64s(&d, "std"),
                f64s(&d, "min"),
                f64s(&d, "max"),
            );
            for (r, (name, role, oi)) in expect.iter().enumerate() {
                let w = &want.columns[*oi];
                let what = format!("{what} / {name}");
                assert_eq!(cols[r].as_deref(), Some(*name), "{what}: name");
                assert_eq!(roles[r].as_deref(), Some(*role), "{what}: role");
                assert_eq!(count[r], Some(w.0), "{what}: count");
                assert_eq!(nulls[r], Some(w.1), "{what}: nulls");
                if s.name == "k" && *role == "target" {
                    assert!(w.0 > 0);
                    assert_eq!((mean[r], std[r], min[r], max[r]), (None, None, None, None));
                    continue;
                }
                let same = |got: Option<f64>, want: Option<f64>| match (got, want) {
                    (Some(a), Some(b)) => close(a, b),
                    (a, b) => a == b,
                };
                assert!(
                    same(mean[r], w.2),
                    "{what}: mean {:?} vs {:?}",
                    mean[r],
                    w.2
                );
                assert!(same(std[r], w.3), "{what}: std {:?} vs {:?}", std[r], w.3);
                assert_eq!(min[r], w.4, "{what}: min");
                assert_eq!(max[r], w.5, "{what}: max");
            }
        }
    }
    // Every kind of missing value in `x0` was seen as a null.
    let d = bank.describe(0, Some("g0")).unwrap();
    let x0 = d
        .filter(&d.column("column").unwrap().str().unwrap().equal("x0"))
        .unwrap();
    assert!(u64s(&x0, "null_count")[0].unwrap() >= 4 * 3);
    assert!(
        f64s(&x0, "max")[0].unwrap() <= 1.0,
        "1e300 and inf were not counted"
    );
}

#[test]
fn chunking_cannot_move_a_bit() {
    let df = make_df(700);
    let specs = specs();
    let mut banks: Vec<Bank> = [1usize, 7, 350]
        .iter()
        .map(|&n| {
            let mut b = Bank::new(specs.clone()).unwrap();
            feed(&mut b, &df, n);
            b
        })
        .collect();
    // ... nor feeding the same rows through `fit_predict` one at a time.
    let mut single = Bank::new(specs.clone()).unwrap();
    for i in 0..df.height() {
        single.fit_predict(&df.slice(i as i64, 1)).unwrap();
    }
    banks.push(single);
    let bits = |d: &DataFrame| -> Vec<Vec<Option<u64>>> {
        d.columns()
            .iter()
            .filter(|c| c.dtype() == &DataType::Float64)
            .map(|c| {
                c.f64()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(f64::to_bits))
                    .collect()
            })
            .collect()
    };
    for (si, spec) in specs.iter().enumerate() {
        let (s0, d0) = (
            banks[0].summary(si, None).unwrap(),
            banks[0].describe(si, None).unwrap(),
        );
        for b in &banks[1..] {
            let (s, d) = (b.summary(si, None).unwrap(), b.describe(si, None).unwrap());
            assert_same(&s0, &s, &format!("{}: summary", spec.name));
            assert_same(&d0, &d, &format!("{}: describe", spec.name));
            assert_eq!(bits(&s0), bits(&s), "{}: summary bits", spec.name);
            assert_eq!(bits(&d0), bits(&d), "{}: describe bits", spec.name);
        }
    }
}

#[test]
fn the_file_carries_it_and_a_re_save_is_the_same_bytes() {
    let df = make_df(700);
    let specs = specs();
    let mut bank = Bank::new(specs.clone()).unwrap();
    feed(&mut bank, &df.slice(0, 500), 4);
    let bytes = bank.save_bytes().unwrap();
    let mut restored = Bank::load_bytes(&bytes, Some(&specs)).unwrap();
    for (si, spec) in specs.iter().enumerate() {
        assert_same(
            &bank.summary(si, None).unwrap(),
            &restored.summary(si, None).unwrap(),
            &format!("{}: summary after load", spec.name),
        );
        assert_same(
            &bank.describe(si, None).unwrap(),
            &restored.describe(si, None).unwrap(),
            &format!("{}: describe after load", spec.name),
        );
    }
    // A loaded bank saves to the same bytes: nothing is lost or reordered
    // on the way through the file.
    assert_eq!(restored.save_bytes().unwrap(), bytes, "re-save");
    // And both continue as one stream would have.
    let tail = df.slice(500, 200);
    let out_a = bank.fit_predict(&tail).unwrap();
    let out_b = restored.fit_predict(&tail).unwrap();
    for (a, b) in out_a.iter().zip(&out_b) {
        assert!(
            a.as_materialized_series()
                .equals_missing(b.as_materialized_series()),
            "{}: output after load",
            a.name()
        );
    }
    for (si, spec) in specs.iter().enumerate() {
        assert_same(
            &bank.summary(si, None).unwrap(),
            &restored.summary(si, None).unwrap(),
            &format!("{}: summary continued", spec.name),
        );
        assert_same(
            &bank.describe(si, None).unwrap(),
            &restored.describe(si, None).unwrap(),
            &format!("{}: describe continued", spec.name),
        );
    }
    assert_eq!(
        bank.save_bytes().unwrap(),
        restored.save_bytes().unwrap(),
        "the two banks are the same file"
    );
    // Whole-stream counts agree with the fed frame.
    let s = bank.summary(0, None).unwrap();
    assert_eq!(
        u64s(&s, "rows_fed").iter().map(|v| v.unwrap()).sum::<u64>(),
        700
    );
}

#[test]
fn predict_moves_nothing_and_an_unseen_group_or_empty_bank_is_empty() {
    let df = make_df(300);
    let specs = specs();
    let mut bank = Bank::new(specs.clone()).unwrap();
    feed(&mut bank, &df.slice(0, 200), 2);
    let before: Vec<(DataFrame, DataFrame)> = (0..specs.len())
        .map(|si| {
            (
                bank.summary(si, None).unwrap(),
                bank.describe(si, None).unwrap(),
            )
        })
        .collect();
    bank.predict(&df.slice(200, 100)).unwrap();
    for (si, (s, d)) in before.iter().enumerate() {
        assert_same(s, &bank.summary(si, None).unwrap(), "summary after predict");
        assert_same(
            d,
            &bank.describe(si, None).unwrap(),
            "describe after predict",
        );
    }
    // Narrowed to a group; a group never seen; a spec out of range.
    assert_eq!(bank.summary(0, Some("g1")).unwrap().height(), 1);
    assert_eq!(bank.describe(0, Some("g1")).unwrap().height(), 4);
    assert_eq!(bank.summary(0, Some("zzz")).unwrap().height(), 0);
    assert_eq!(bank.describe(0, Some("zzz")).unwrap().height(), 0);
    assert!(
        bank.summary(6, None)
            .unwrap_err()
            .contains("spec index 6 out of range")
    );
    assert!(
        bank.describe(6, None)
            .unwrap_err()
            .contains("spec index 6 out of range")
    );
    let empty = Bank::new(specs).unwrap();
    let s = empty.summary(0, None).unwrap();
    assert_eq!(s.height(), 0);
    assert_eq!(s.width(), 13);
    let d = empty.describe(0, None).unwrap();
    assert_eq!(d.height(), 0);
    assert_eq!(d.width(), 9);
}

#[test]
fn a_group_of_skipped_rows_is_fed_and_not_processed() {
    let df = make_df(100);
    let mut bank = Bank::new(specs()).unwrap();
    let skipped = df
        .slice(0, 10)
        .lazy()
        .with_columns([lit(NULL).cast(DataType::Float64).alias("x0")])
        .collect()
        .unwrap();
    bank.fit_predict(&skipped).unwrap();
    for si in [0, 1, 2, 3] {
        let s = bank.summary(si, None).unwrap();
        for g in ["g0", "g1"] {
            let r = row_of(&s, g);
            assert_eq!(u64s(&r, "rows_fed")[0], Some(5));
            assert_eq!(u64s(&r, "rows_processed")[0], Some(0));
            assert_eq!(u64s(&r, "rows_skipped")[0], Some(5));
            assert_eq!(u64s(&r, "rows_learned")[0], Some(0));
            assert_eq!(f64s(&r, "weight_sum")[0], Some(0.0));
            assert!(f64s(&r, "clock_min")[0].is_some(), "the clock still moved");
        }
        let d = bank.describe(si, Some("g0")).unwrap();
        let x0 = d
            .filter(&d.column("column").unwrap().str().unwrap().equal("x0"))
            .unwrap();
        assert_eq!(u64s(&x0, "count")[0], Some(0));
        assert_eq!(u64s(&x0, "null_count")[0], Some(5));
        assert_eq!(f64s(&x0, "mean")[0], None);
        assert_eq!(f64s(&x0, "min")[0], None);
    }
}

#[test]
fn a_saved_summary_that_is_not_its_spec_s_is_refused_and_none_stays_none() {
    let df = make_df(120);
    let specs = specs();
    let mut bank = Bank::new(specs.clone()).unwrap();
    feed(&mut bank, &df, 3);
    let bytes = bank.save_bytes().unwrap();
    let refused = |mutate: &dyn Fn(&mut online_polars::DataSummary, &mut u64), what: &str| {
        let mut st = Stream::new(&specs[0]).unwrap().save();
        // Start from a real summary: a freshly built stream's would pass.
        let live = Bank::load_bytes(&bytes, Some(&specs)).unwrap();
        let live = live.summary(0, Some("g0")).unwrap();
        st.rows_seen = u64s(&live, "rows_processed")[0].unwrap();
        let mut summary = st.summary.clone().unwrap();
        summary.rows_fed = u64s(&live, "rows_fed")[0].unwrap();
        // Make the columns consistent with `rows_fed` first, and give it a
        // clock range to halve.
        for c in &mut summary.columns {
            c.nulls = summary.rows_fed;
        }
        summary.rows_learned = 0;
        (summary.clock_min, summary.clock_max) = (Some(0.0), Some(1.0));
        assert!(
            Stream::restore(&specs[0], &{
                let mut ok = st.clone();
                ok.summary = Some(summary.clone());
                ok
            })
            .is_ok(),
            "the starting point passes"
        );
        mutate(&mut summary, &mut st.rows_seen);
        st.summary = Some(summary);
        let err = Stream::restore(&specs[0], &st).err().unwrap_or_default();
        assert!(
            err.contains("saved data summary of spec \"m\" is not its spec's"),
            "{what}: {err:?}"
        );
    };
    refused(
        &|s, _| s.columns.pop().map(|_| ()).unwrap(),
        "a column short",
    );
    refused(&|s, _| s.columns.push(Default::default()), "a column over");
    refused(
        &|s, rows_seen| s.rows_fed = *rows_seen - 1,
        "fewer fed than processed",
    );
    refused(
        &|s, rows_seen| s.rows_learned = *rows_seen + 1,
        "more learned than processed",
    );
    refused(&|s, _| s.weight_sum = f64::NAN, "a NaN weight sum");
    refused(&|s, _| s.weight_sum = -1.0, "a negative weight sum");
    refused(&|s, _| s.clock_min = None, "half a clock range");
    refused(
        &|s, _| (s.clock_min, s.clock_max) = (Some(2.0), Some(1.0)),
        "a backwards clock range",
    );
    refused(&|s, _| s.resets = s.rows_fed + 1, "more resets than rows");
    refused(
        &|s, _| s.columns[0].nulls += 1,
        "a column with a row too many",
    );
    refused(
        &|s, _| s.columns[1].m2 = f64::INFINITY,
        "an infinite moment",
    );

    // A file from before the summary existed restores as `None` (the
    // frozen 0.1.x and schema-2 fixtures feed such a bank on and check that
    // `None` it stays: `tests/state_v1.rs`, `tests/state_schema2.rs`).
    let mut st = Stream::new(&specs[0]).unwrap().save();
    st.summary = None;
    let stream = Stream::restore(&specs[0], &st).unwrap();
    assert!(stream.summary().is_none());
    assert!(stream.save().summary.is_none());
}

/// Every prefix of a state file, and every one of a spread of bit flips,
/// is either loaded or refused with an error -- never a panic -- and what
/// loads can be read and saved again.
#[test]
fn truncated_and_bit_flipped_files_are_refused_or_loaded_never_panic() {
    let df = make_df(60);
    // One small spec so every prefix length can be tried.
    let specs = vec![spec(
        r#"{"name": "m", "model": {"type": "ew_ridge", "ridge": 1e-6},
            "targets": ["y"], "features": ["x0"], "clock": "t", "weight": "w",
            "group": "g", "halflife": 10.0, "max_dclock": 30.0}"#,
    )];
    let mut bank = Bank::new(specs.clone()).unwrap();
    feed(&mut bank, &df, 2);
    let bytes = bank.save_bytes().unwrap();
    let survives = |b: &[u8], what: &str| {
        if let Ok(loaded) = Bank::load_bytes(b, Some(&specs)) {
            // Loaded: usable and re-saveable, whatever the bytes were.
            let _ = loaded.summary(0, None).unwrap();
            let _ = loaded.describe(0, None).unwrap();
            let _ = loaded.last_row(0, None).unwrap();
            let _ = loaded.groups();
            loaded
                .save_bytes()
                .unwrap_or_else(|e| panic!("{what}: re-save: {e}"));
        }
    };
    let mut loaded_short = 0;
    for n in 0..bytes.len() {
        if Bank::load_bytes(&bytes[..n], Some(&specs)).is_ok() {
            loaded_short += 1;
        }
        survives(&bytes[..n], &format!("prefix {n}"));
    }
    assert_eq!(loaded_short, 0, "a truncated file must not load");
    let mut loaded_flipped = 0;
    for pos in (0..bytes.len()).step_by(3) {
        for bit in [0, 3, 7] {
            let mut b = bytes.clone();
            b[pos] ^= 1 << bit;
            if b != bytes && Bank::load_bytes(&b, Some(&specs)).is_ok() {
                loaded_flipped += 1;
            }
            survives(&b, &format!("byte {pos} bit {bit}"));
        }
    }
    // Most of the file is `f64` payload, where a flipped bit is a slightly
    // different but consistent state and loads; a flip in the framing --
    // the magic, a length, a type tag -- is refused. Both must happen for
    // the sweep to have tested anything.
    let tried = bytes.len().div_ceil(3) * 3;
    assert!(
        loaded_flipped > 0 && loaded_flipped < tried,
        "{loaded_flipped} of {tried} loaded"
    );
    // Still the same bank after all that.
    assert_eq!(bank.save_bytes().unwrap(), bytes);
}
