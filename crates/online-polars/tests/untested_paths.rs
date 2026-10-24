//! Paths of the bank, its streams and its spec that no test reached
//! (task 252: "every untested path gets a test"), each driven through the
//! public API and held to what it does -- the value, the refusal and its
//! words, or the state -- rather than merely run.

use online_polars::{
    Bank, Format, Spec, Stream, Threads, coef_fields, output_index, resolved_defaults,
};
use polars::prelude::*;

/// A column put in a frame, by its name: added, or in place of the one of
/// that name.
trait Put {
    fn put(&mut self, s: Series) -> PolarsResult<()>;
}

impl Put for DataFrame {
    fn put(&mut self, s: Series) -> PolarsResult<()> {
        self.with_column(s.into_column()).map(|_| ())
    }
}

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}: {json}"))
}

/// A seeded stream of `n` rows: `x0`, `x1` uniform in `[-1, 1)`, `y` a
/// noisy line in them, `t` a clock a unit apart.
fn rows(n: usize, seed: u64) -> DataFrame {
    let mut s = seed;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut x0, mut x1, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let (a, b) = (lcg(), lcg());
        x0.push(a);
        x1.push(b);
        y.push(1.0 + 2.0 * a - b + 0.1 * lcg());
    }
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    df!("t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

fn field(out: &[Column], spec: usize, name: &str) -> Series {
    out[spec]
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn floats(s: &Series) -> Vec<Option<f64>> {
    s.f64().unwrap().iter().collect()
}

/// A clock column of an integer or a temporal dtype is held in integers, and
/// a null in it is refused by row, as a float clock's null is: `Int64`,
/// `UInt64` and a `Datetime` alike.
#[test]
fn a_null_in_an_integer_or_temporal_clock_is_refused_by_row() {
    let base = rows(6, 1);
    let ints = Series::new(
        "t".into(),
        [Some(0i64), Some(1), Some(2), None, Some(4), Some(5)],
    );
    let cases: Vec<(&str, Series, &str)> = vec![
        ("Int64", ints.clone(), r#""gap_cap": 5.0"#),
        (
            "UInt64",
            ints.cast(&DataType::UInt64).unwrap(),
            r#""gap_cap": 5.0"#,
        ),
        (
            "Datetime",
            ints.cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap(),
            r#""gap_cap": "1s", "half_life": "10us""#,
        ),
    ];
    for (what, clock, policy) in cases {
        let mut df = base.clone();
        df.put(clock).unwrap();
        let half_life = if policy.contains("half_life") {
            ""
        } else {
            r#", "half_life": 10.0"#
        };
        let s = spec(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x0", "x1"], "clock": "t", {policy}{half_life}}}"#
        ));
        let mut bank = Bank::new(vec![s]).unwrap();
        let e = bank.fit_predict(&df).unwrap_err().to_string();
        assert!(
            e.contains("clock column \"t\" has a null/non-finite value at row 3"),
            "{what}: {e}"
        );
        assert_eq!(
            bank.rows_fed(),
            0,
            "{what}: the refused chunk taught nothing"
        );
    }
}

/// Every integer width and every temporal dtype a clock can have: the
/// emitted clocks come back in the column's own dtype, and two specs that
/// keep their clocks in two dtypes are refused one table, the refusal
/// naming both as Polars names them.
#[test]
fn every_clock_dtype_is_emitted_as_itself_and_named_as_polars_names_it() {
    let n = 8;
    let base = rows(n, 2);
    let ints = Series::new("u".into(), (0..n as i64).collect::<Vec<_>>());
    let ns = |unit: TimeUnit| DataType::Datetime(unit, None);
    let cases: Vec<(DataType, &str)> = vec![
        (DataType::Int8, r#""gap_cap": 5.0, "half_life": 10.0"#),
        (DataType::Int16, r#""gap_cap": 5.0, "half_life": 10.0"#),
        (DataType::UInt8, r#""gap_cap": 5.0, "half_life": 10.0"#),
        (DataType::UInt16, r#""gap_cap": 5.0, "half_life": 10.0"#),
        (DataType::UInt32, r#""gap_cap": 5.0, "half_life": 10.0"#),
        (DataType::UInt64, r#""gap_cap": 5.0, "half_life": 10.0"#),
        (DataType::Date, r#""gap_cap": "5d", "half_life": "10d""#),
        (
            ns(TimeUnit::Milliseconds),
            r#""gap_cap": "5s", "half_life": "10ms""#,
        ),
        (
            ns(TimeUnit::Nanoseconds),
            r#""gap_cap": "5s", "half_life": "10ns""#,
        ),
        (
            DataType::Duration(TimeUnit::Microseconds),
            r#""gap_cap": "5s", "half_life": "10us""#,
        ),
    ];
    for (dtype, policy) in cases {
        let mut df = base.clone();
        df.put(ints.cast(&dtype).unwrap()).unwrap();
        let a = spec(
            r#"{"name": "a", "model": {"type": "ewridge"}, "targets": ["y"],
                "features": ["x0", "x1"], "clock": "t", "gap_cap": 5.0, "half_life": 10.0}"#,
        );
        let b = spec(&format!(
            r#"{{"name": "b", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x0", "x1"], "clock": "u", {policy}, "emit_clocks": true}}"#
        ));
        let mut bank = Bank::new(vec![a, b]).unwrap();
        let out = bank.fit_predict(&df).unwrap();
        let scored = field(&out, 1, "scored_clock");
        assert_eq!(scored.dtype(), &dtype, "the emitted clock keeps its dtype");
        assert_eq!(
            scored
                .cast(&DataType::Int64)
                .unwrap()
                .i64()
                .unwrap()
                .get(n - 1),
            ints.get(n - 1).unwrap().try_extract::<i64>().ok(),
            "{dtype}: the last row's clock is its own value"
        );
        let e = bank.groups_table(&[0, 1]).unwrap_err();
        assert!(
            e.contains("spec \"a\" keeps its clock as f64")
                && e.contains(&format!("spec \"b\" as {dtype}")),
            "{dtype}: {e}"
        );
        // One spec at a time reads.
        let one = bank.groups_table(&[1]).unwrap();
        assert_eq!(one.column("clock_last").unwrap().dtype(), &dtype);
    }
}

/// The closed groups of a spec past the bank's are refused by index, and a
/// bank of two closing specs, one `ew_cov` with lags and one `ewridge`,
/// gives the `ewridge` rows a null `lags`.
#[test]
fn closed_groups_name_a_spec_out_of_range_and_null_another_kinds_lags() {
    let cov = spec(
        r#"{"name": "c", "model": {"type": "ew_cov", "lags": [1]}, "features": ["x0", "x1"],
            "group": "g", "half_life": 10.0, "group_close": "monotone"}"#,
    );
    let ridge = spec(
        r#"{"name": "r", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0", "x1"], "group": "g", "half_life": 10.0,
            "group_close": "monotone"}"#,
    );
    let mut df = rows(40, 3);
    df.put(Series::new(
        "g".into(),
        (0..40i64).map(|i| i / 20).collect::<Vec<_>>(),
    ))
    .unwrap();
    let mut bank = Bank::new(vec![cov, ridge]).unwrap();
    bank.fit_predict(&df).unwrap();
    assert_eq!(
        bank.closed_groups(Some(2), false).unwrap_err(),
        "spec index 2 out of range"
    );
    // The queue survives a save and a load, lags and all.
    let mut loaded = Bank::load_bytes(&bank.save_bytes().unwrap(), None).unwrap();
    let closed = bank.closed_groups(None, true).unwrap();
    assert!(closed.equals_missing(&loaded.closed_groups(None, true).unwrap()));
    assert_eq!(closed.height(), 2, "group 0 closed in each spec");
    let spec_col: Vec<Option<&str>> = closed
        .column("spec")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .collect();
    let lags = closed.column("lags").unwrap().list().unwrap().clone();
    for (i, s) in spec_col.iter().enumerate() {
        match *s {
            Some("c") => {
                let l = lags.get_as_series(i).expect("ew_cov keeps its lags");
                assert_eq!(l.i64().unwrap().get(0), Some(1));
            }
            Some("r") => assert!(lags.get_as_series(i).is_none(), "ewridge has no lags"),
            other => panic!("spec {other:?}"),
        }
    }
}

/// Two specs of one name are refused at the bank, naming it.
#[test]
fn a_bank_of_two_specs_of_one_name_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0}"#,
    );
    let e = Bank::new(vec![s.clone(), s]).err().expect("accepted");
    assert_eq!(e, "duplicate spec name \"m\"");
}

/// A stream state restored under another spec: a grid of another size is
/// refused; a diagnostic the spec has since gained or lost starts over,
/// leaving the state exactly a fresh stream's. (One kept under the same
/// switches but shaped for another lag or horizon is damage: see
/// `bank/damaged_file_tests.rs`.)
#[test]
fn a_stream_restored_under_another_spec_starts_its_diagnostics_over_or_is_refused() {
    let base = r#""name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
        "features": ["x0", "x1"]"#;
    let at = |extra: &str| spec(&format!(r#"{{{base}, "half_life": 10.0{extra}}}"#));
    let grid = spec(&format!(r#"{{{base}, "half_life": [10.0, 20.0]}}"#));
    let e = Stream::restore(&at(""), &Stream::new(&grid).unwrap().save())
        .err()
        .expect("a grid of two restored as one");
    assert_eq!(e, "saved state has a different number of model instances");

    let options = [
        r#", "emit_sigma": true"#,
        r#", "emit_autocorr": true, "resid_autocorr_lag": 2"#,
        r#", "emit_drift": true"#,
        r#", "conformal": 0.9"#,
        r#", "emit_calibration": true"#,
        r#", "emit_metrics": true"#,
    ];
    for opt in options {
        for (from, to) in [(at(""), at(opt)), (at(opt), at(""))] {
            let restored = Stream::restore(&to, &Stream::new(&from).unwrap().save())
                .unwrap_or_else(|e| panic!("{opt}: {e}"));
            // As Debug text: a fresh interval's quantile is NaN, which
            // `==` would hold unequal to itself.
            assert_eq!(
                format!("{:?}", restored.persisted),
                format!("{:?}", Stream::new(&to).unwrap().persisted),
                "{opt}: the gained or lost diagnostic starts over"
            );
        }
    }
}

/// `holt` under `lam` is `holt` under the half-life `lam` stands for,
/// `-ln 2 / ln lam`, to the rounding of that conversion.
#[test]
fn holt_under_lam_is_holt_under_the_half_life_it_stands_for() {
    let lam = 0.9f64;
    let h = -std::f64::consts::LN_2 / lam.ln();
    let df = rows(80, 4);
    let run = |model: String, decay: String| {
        let s = spec(&format!(
            r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": [], {decay}}}"#
        ));
        let mut bank = Bank::new(vec![s]).unwrap();
        floats(&field(&bank.fit_predict(&df).unwrap(), 0, "pred_y"))
    };
    // The trend's half-life defaults to four times the level's.
    let by_lam = run(r#"{"type": "holt"}"#.into(), format!(r#""lam": {lam}"#));
    let by_h = run(
        format!(r#"{{"type": "holt", "trend_half_life": {:?}}}"#, 4.0 * h),
        format!(r#""half_life": {h:?}"#),
    );
    let mut compared = 0;
    for (a, b) in by_lam.iter().zip(&by_h) {
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
            compared += 1;
        }
    }
    assert!(compared > 60, "only {compared} rows predicted");
}

/// A windowed spread (`window_size` with `emit_sigma`) restarts with the
/// model at a session change under `session_gap = "reset"`: the rows after
/// the change are a fresh bank's on those rows alone, to the bit.
#[test]
fn a_session_reset_restarts_a_windowed_spread_with_its_model() {
    let n = 120;
    let mut df = rows(n, 5);
    df.put(Series::new(
        "s".into(),
        (0..n)
            .map(|i| if i < 60 { "a" } else { "b" })
            .collect::<Vec<_>>(),
    ))
    .unwrap();
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "window_size": 20.0},
            "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 5.0,
            "half_life": 10.0, "session": "s", "session_gap": "reset", "emit_sigma": true}"#,
    );
    let whole = Bank::new(vec![s.clone()])
        .unwrap()
        .fit_predict(&df)
        .unwrap();
    let after = Bank::new(vec![s])
        .unwrap()
        .fit_predict(&df.slice(60, 60))
        .unwrap();
    for name in ["pred_y", "sigma_y", "weight_sum"] {
        let w = floats(&field(&whole, 0, name));
        let a = floats(&field(&after, 0, name));
        assert_eq!(&w[60..], &a[..], "{name} after the reset");
    }
    let sigma = floats(&field(&whole, 0, "sigma_y"));
    assert!(
        sigma[..60].iter().flatten().count() > 0,
        "the spread ran before the change"
    );
}

/// `max_error_inflation`'s notice under `lam` names the `lam` that opens
/// the gate, the half-life the same notice names under `half_life` read as
/// `lam = 2^(-1/h)`; and with `min_weight` given, the weight it needs before
/// it solves is that, not a row per coefficient.
#[test]
fn the_noise_gates_notice_under_lam_names_the_lam_that_opens_it() {
    let k = 10;
    let n = 1200;
    let mut s = 11u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut cols: Vec<Column> = Vec::new();
    let mut y = vec![1.0; n];
    for j in 0..k {
        let x: Vec<f64> = (0..n).map(|_| lcg()).collect();
        for (yi, xi) in y.iter_mut().zip(&x) {
            *yi += (j + 1) as f64 * xi;
        }
        cols.push(Column::new(format!("x{j}").into(), x));
    }
    for yi in &mut y {
        *yi += 0.5 * lcg();
    }
    cols.push(Column::new("y".into(), y));
    cols.push(Column::new(
        "t".into(),
        (0..n).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    let df = DataFrame::new(n, cols).unwrap();
    let features: Vec<String> = (0..k).map(|j| format!("\"x{j}\"")).collect();
    let notice = |decay: &str| {
        let s = spec(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge", "max_rows_between_solves": 1}},
                "targets": ["y"], "features": [{}], {decay}}}"#,
            features.join(", ")
        ));
        let mut bank = Bank::new(vec![s]).unwrap();
        bank.fit_predict(&df).unwrap();
        let notices = bank.take_notices();
        assert_eq!(notices.len(), 1, "{decay}: {notices:?}");
        notices[0].clone()
    };
    let h = 2.0f64;
    let by_h = notice(&format!(r#""half_life": {h:?}"#));
    let need_h: f64 = by_h
        .split("aise the half_life above ")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .unwrap_or_else(|| panic!("{by_h}"))
        .trim_end_matches(',')
        .parse()
        .unwrap();
    let by_lam = notice(&format!(r#""lam": {:?}"#, (-1.0 / h).exp2()));
    let need_lam: f64 = by_lam
        .split("aise lam above ")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .unwrap_or_else(|| panic!("{by_lam}"))
        .trim_end_matches(',')
        .parse()
        .unwrap();
    // Tolerance (task 252 (b)): both figures are read back from a notice's
    // text, printed rounded for a reader, and a half-life becomes a factor
    // through libm's `exp2`.
    assert!(
        (need_lam - (-1.0 / need_h).exp2()).abs() < 1e-5,
        "lam {need_lam} against half_life {need_h}: {by_lam}"
    );
    // With min_weight given, the floor is the spec's: a floor of 3 rows is
    // met where 11 coefficients are not, so the figure the notice names
    // moves down.
    let floored = notice(&format!(r#""half_life": {h:?}, "min_weight": 3.0"#));
    assert_ne!(floored, by_h, "the floor moved nothing");
}

/// A state saved to a path that names no file is refused before anything
/// is written, naming the path.
#[test]
fn a_state_saved_to_a_path_that_names_no_file_is_refused() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0}"#,
    );
    let bank = Bank::new(vec![s]).unwrap();
    let e = bank.save(std::path::Path::new("..")).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        e.to_string().contains(".. : not a file path")
            || e.to_string().contains("..: not a file path"),
        "{e}"
    );
}

/// Each format's name is the one its config key and its extension use.
#[test]
fn each_format_names_itself_as_its_config_key() {
    for (f, name, path) in [
        (Format::Parquet, "parquet", "a.pq"),
        (Format::Ipc, "ipc", "a.feather"),
        (Format::Csv, "csv", "a.CSV"),
        (Format::Ndjson, "ndjson", "a.jsonl"),
    ] {
        assert_eq!(f.name(), name);
        assert_eq!(Format::from_path(std::path::Path::new(path)).unwrap(), f);
    }
}

/// A pool is shown by what it is: the process pool, or one of a count.
#[test]
fn a_thread_pool_shows_what_it_is() {
    let process = Threads::new(None).unwrap();
    assert_eq!(format!("{process:?}"), "Threads(Process)");
    assert_eq!(process.asked(), None);
    // A count the process pool already has is that pool; another is a pool
    // of its own.
    let n = online_polars::pool().unwrap().current_num_threads();
    let same = Threads::new(Some(n)).unwrap();
    assert_eq!(format!("{same:?}"), format!("Threads(Default({n}))"));
    assert_eq!(same.asked(), Some(n));
    let other = Threads::new(Some(n + 1)).unwrap();
    assert_eq!(format!("{other:?}"), format!("Threads(Sized({}))", n + 1));
    assert_eq!(other.asked(), Some(n + 1));
}

/// `deco` with three blocks: each pair's field is over the two blocks'
/// columns, the second and third pairs too, and under `lam` its `coef`
/// fields carry the `lam` and no half-life.
#[test]
fn deco_pair_fields_name_their_blocks_columns_and_its_coef_carries_lam() {
    let s = spec(
        r#"{"name": "d", "model": {"type": "deco", "blocks": [["a", ["x0", "x1"]],
            ["b", ["x2", "x3"]], ["c", ["x4", "x5"]]]},
            "features": ["x0", "x1", "x2", "x3", "x4", "x5"], "lam": 0.99}"#,
    );
    s.clone().check().unwrap();
    let index = output_index(&s);
    let cols = |name: &str| {
        index
            .iter()
            .find(|f| f.field == name)
            .unwrap_or_else(|| {
                panic!(
                    "{name} not in {:?}",
                    index.iter().map(|f| &f.field).collect::<Vec<_>>()
                )
            })
            .columns
            .clone()
    };
    let strs = |v: &[&str]| Some(v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    assert_eq!(cols("rho_a_b"), strs(&["x0", "x1", "x2", "x3"]));
    assert_eq!(cols("rho_a_c"), strs(&["x0", "x1", "x4", "x5"]));
    assert_eq!(cols("rho_b_c"), strs(&["x2", "x3", "x4", "x5"]));
    let coef = coef_fields(&s);
    assert!(!coef.is_empty());
    for c in &coef {
        assert_eq!((c.half_life, c.lam), (None, Some(0.99)), "{}", c.name);
    }
}

/// Two outputs that render to one field name, the names being the columns'
/// own, are refused naming both, with the columns each is over; under
/// `lam` there is no half-life to name.
#[test]
fn two_outputs_of_one_name_are_refused_naming_their_columns() {
    let mut s = spec(
        r#"{"name": "c", "model": {"type": "ew_cov", "stats": ["corr"]},
            "features": ["a_b", "c", "a", "b_c"], "lam": 0.99}"#,
    );
    let e = s.check().unwrap_err();
    assert!(
        e.contains("two outputs render to the same field name \"corr_a_b_c\"")
            && e.contains("over columns [\"a_b\", \"c\"]")
            && e.contains("over columns [\"a\", \"b_c\"]")
            && !e.contains("half_life"),
        "{e}"
    );
}

/// `gram_threads` given as a count is shown as that count among the
/// resolved defaults, as `"auto"` is shown for none.
#[test]
fn a_gram_threads_count_is_shown_among_the_defaults() {
    let with = |extra: &str| {
        spec(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"{extra}}}, "targets": ["y"],
                "features": ["x0"], "half_life": 10.0}}"#
        ))
    };
    let shown = |s: &Spec| resolved_defaults(s).unwrap()["stream"]["gram_threads"].clone();
    assert_eq!(shown(&with(r#", "gram_threads": 3"#)), serde_json::json!(3));
    assert_eq!(shown(&with("")), serde_json::json!("auto"));
}

/// What a resumed bank has learned, read off a frame by each group's last
/// learned clock: a row at or before it is learned, after it or with no
/// clock it is not, for a float, an integer, an unsigned and a temporal
/// clock; a group the bank has not seen is not; a spec that reads no clock
/// takes no part; an integer clock is compared with a float one as
/// doubles; and a temporal clock against a numeric one is refused.
#[test]
fn what_a_bank_has_learned_is_read_off_each_clock_type() {
    let learned_on = |clock: Series, policy: &str| {
        let n = clock.len();
        let mut df = rows(n, 6);
        df.put(clock).unwrap();
        df.put(Series::new("g".into(), vec!["a"; n])).unwrap();
        let clocked = spec(&format!(
            r#"{{"name": "c", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x0"], "clock": "t", "group": "g", {policy}}}"#
        ));
        let plain = spec(
            r#"{"name": "p", "model": {"type": "ewridge"}, "targets": ["y"],
                "features": ["x0"], "half_life": 10.0}"#,
        );
        let mut bank = Bank::new(vec![clocked, plain]).unwrap();
        bank.fit_predict(&df).unwrap();
        bank.learned().unwrap()
    };
    let numbers = r#""gap_cap": 50.0, "half_life": 10.0"#;
    let probe = |clock: Series, groups: &[&str]| {
        let n = clock.len();
        let mut df = rows(n, 7);
        df.put(clock).unwrap();
        df.put(Series::new("g".into(), groups.to_vec())).unwrap();
        df
    };
    let kept = |l: &online_polars::Learned, df: &DataFrame| -> Vec<bool> {
        l.unlearned(df)
            .unwrap()
            .iter()
            .map(|b| b.unwrap())
            .collect()
    };
    let ints = |v: &[Option<i64>]| Series::new("t".into(), v.to_vec());
    // Learned through clock 4 in group "a".
    let learned_ints = ints(&[Some(0), Some(2), Some(4)]);
    let frame = |dtype: &DataType| {
        probe(
            ints(&[Some(3), Some(4), Some(5), None, Some(1)])
                .cast(dtype)
                .unwrap(),
            &["a", "a", "a", "a", "b"],
        )
    };
    for (dtype, policy) in [
        (DataType::Float64, numbers),
        (DataType::Int64, numbers),
        (DataType::UInt64, numbers),
        (
            DataType::Datetime(TimeUnit::Nanoseconds, None),
            r#""gap_cap": "1s", "half_life": "10ns""#,
        ),
    ] {
        let l = learned_on(learned_ints.cast(&dtype).unwrap(), policy);
        assert_eq!(
            kept(&l, &frame(&dtype)),
            [false, false, true, true, true],
            "{dtype}: at or before 4 learned; a null, and a new group, not"
        );
    }
    // Past what an Int64 holds: after every clock an integer bank holds.
    let l = learned_on(learned_ints.cast(&DataType::UInt64).unwrap(), numbers);
    let past = probe(Series::new("t".into(), [3u64, u64::MAX]), &["a", "a"]);
    assert_eq!(kept(&l, &past), [false, true]);
    // An integer clock against a float one, and the reverse, as doubles.
    let l = learned_on(learned_ints.cast(&DataType::Float64).unwrap(), numbers);
    assert_eq!(
        kept(&l, &frame(&DataType::Int64)),
        [false, false, true, true, true]
    );
    let l = learned_on(learned_ints.clone(), numbers);
    let halves = probe(
        Series::new("t".into(), [3.5f64, 4.0, 4.5]),
        &["a", "a", "a"],
    );
    assert_eq!(kept(&l, &halves), [false, false, true]);
    // A temporal clock against a numeric one is refused by the frame's
    // clock check before any comparison: the spec's clock parameters are
    // numbers, which a temporal column cannot read.
    let e = l
        .unlearned(&frame(&DataType::Datetime(TimeUnit::Nanoseconds, None)))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("a temporal clock, but half_life is a plain number"),
        "{e}"
    );
}

/// A temporal clock column is a clock and nothing else: another spec that
/// reads it as a weight, or as a column of a target's formula, is refused
/// naming the role; and `kalman`'s `q`, a number of the clock's own units,
/// is refused on it with the duration to give instead.
#[test]
fn a_temporal_clock_column_is_refused_where_it_cannot_be_read() {
    let n = 6;
    let mut df = rows(n, 8);
    let stamps = Series::new("t".into(), (0..n as i64).collect::<Vec<_>>())
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
        .unwrap();
    df.put(stamps).unwrap();
    let clocked = spec(
        r#"{"name": "a", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0"], "clock": "t", "gap_cap": "1s", "half_life": "10us"}"#,
    );
    let refusal = |other: Spec| {
        let mut bank = Bank::new(vec![clocked.clone(), other]).unwrap();
        bank.fit_predict(&df).unwrap_err().to_string()
    };
    let weighted = spec(
        r#"{"name": "b", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0"], "weight": "t", "half_life": 10.0}"#,
    );
    let e = refusal(weighted);
    assert!(
        e.contains("spec \"b\": column \"t\" has dtype")
            && e.contains("and b reads it as a weight, which must be numeric"),
        "{e}"
    );
    let formula = spec(
        r#"{"name": "c", "model": {"type": "ewridge"},
            "targets": [{"name": "fwd", "formula": ["rewm_mean", ["col", "t"], {"half_life": 5.0, "window_size": 10.0}]}],
            "features": ["x0"], "half_life": 10.0, "embargo": 10.0}"#,
    );
    let e = refusal(formula);
    assert!(
        e.contains("reads it as a column of a target's formula, which must be numeric"),
        "{e}"
    );
    let kalman = spec(
        r#"{"name": "k", "model": {"type": "kalman", "q": [0.1, 0.1]}, "targets": ["y"],
            "features": ["x0"], "clock": "t", "gap_cap": 5.0, "lam": 1.0}"#,
    );
    let e = Bank::new(vec![kalman])
        .unwrap()
        .fit_predict(&df)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("but q is a number in the clock's own units")
            && e.contains("give coef_half_life as a duration"),
        "{e}"
    );
}

/// A formula target over an unsigned column reads it as the numbers it
/// holds: the same predictions as over its `Float64` cast.
#[test]
fn a_formula_target_over_an_unsigned_column_reads_its_numbers() {
    let n = 60;
    let mut df = rows(n, 9);
    let mid: Vec<u64> = (0..n as u64).map(|i| (i * 7) % 13).collect();
    df.put(Series::new("mid".into(), mid)).unwrap();
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"},
            "targets": [{"name": "fwd", "formula": ["rewm_mean", ["col", "mid"], {"half_life": 2.0, "window_size": 3.0}]}],
            "features": ["x0", "x1"], "clock": "t", "gap_cap": 5.0, "half_life": 10.0,
            "embargo": 3.0}"#,
    );
    let run = |df: &DataFrame| {
        let mut bank = Bank::new(vec![s.clone()]).unwrap();
        floats(&field(&bank.fit_predict(df).unwrap(), 0, "pred_fwd"))
    };
    let unsigned = run(&df);
    let mut as_float = df.clone();
    as_float
        .put(
            df.column("mid")
                .unwrap()
                .as_materialized_series()
                .cast(&DataType::Float64)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(unsigned, run(&as_float));
    assert!(unsigned.iter().flatten().count() > 20, "it predicted");
    // An unsigned clock column, which the window core reads in its integer
    // key form: the same predictions as under its signed cast.
    let clocked = |dtype: &DataType| {
        let mut d = df.clone();
        d.put(
            Series::new("t".into(), (0..n as i64).collect::<Vec<_>>())
                .cast(dtype)
                .unwrap(),
        )
        .unwrap();
        d
    };
    let by_u64 = run(&clocked(&DataType::UInt64));
    assert_eq!(by_u64, run(&clocked(&DataType::Int64)));
    assert!(by_u64.iter().flatten().count() > 20, "it predicted");
}

/// `emit_specification` under `lam` is the specification test under the
/// half-life `lam` stands for: a memory that forgets, read as one.
#[test]
fn the_specification_test_under_lam_is_the_test_under_its_half_life() {
    let lam = 0.98f64;
    let h = -std::f64::consts::LN_2 / lam.ln();
    let df = rows(300, 10);
    let run = |decay: String| {
        let s = spec(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x0", "x1"], "emit_specification": true, {decay}}}"#
        ));
        let mut bank = Bank::new(vec![s]).unwrap();
        let out = bank.fit_predict(&df).unwrap();
        let st = out[0].struct_().unwrap();
        let name = st
            .fields_as_series()
            .iter()
            .map(|f| f.name().to_string())
            .find(|n| n.starts_with("ljung_box"))
            .expect("a Ljung-Box field");
        floats(&field(&out, 0, &name))
    };
    let by_lam = run(format!(r#""lam": {lam}"#));
    let by_h = run(format!(r#""half_life": {h:?}"#));
    let mut compared = 0;
    for (a, b) in by_lam.iter().zip(&by_h) {
        if let (Some(a), Some(b)) = (a, b) {
            assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} vs {b}");
            compared += 1;
        }
    }
    assert!(compared > 100, "only {compared} rows tested");
}

/// A diagnostic's default memory under a window of one row: the window
/// holds no more than one row's Kish size, so the half-life is the
/// continuous form's, `ln 2 / (2 / W)`.
#[test]
fn a_window_of_one_row_gives_its_diagnostics_the_continuous_memory() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "window_size": 1.0}, "targets": ["y"],
            "features": ["x0"], "half_life": "inf", "emit_breaks": true}"#,
    );
    let d = resolved_defaults(&s).unwrap();
    let shown = d["stream"]["breaks_half_life"]
        .as_f64()
        .unwrap_or_else(|| panic!("{d}"));
    assert_eq!(shown, std::f64::consts::LN_2 / 2.0);
}

/// A state loaded against specs that differ names where: past a key the
/// two spell alike (`"600s"` and `"10m"`), the first key whose value
/// differs, with the path into a list of lists.
#[test]
fn a_state_of_other_specs_is_refused_naming_the_first_key_that_differs() {
    let at = |half_life: &str, set: &str| {
        spec(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge", "feature_sets": [["a", ["{set}"]]]}},
                "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": "1h",
                "half_life": "{half_life}"}}"#
        ))
    };
    let saved = Bank::new(vec![at("600s", "x0")])
        .unwrap()
        .save_bytes()
        .unwrap();
    let e = Bank::load_bytes(&saved, Some(&[at("10m", "x1")]))
        .err()
        .expect("loaded");
    assert!(
        e.contains(
            "spec \"m\" (1 of 1) differs at model.feature_sets[0][1][0]: \"x0\" in the file, \
             \"x1\" in the bank"
        ),
        "{e}"
    );
    assert!(!e.contains("half_life"), "{e}");
    assert!(Bank::load_bytes(&saved, Some(&[at("10m", "x0")])).is_ok());
}

/// An audit pooled over no group is an empty table; and on a `Date` clock
/// its steps are `Duration`s in milliseconds, polars' unit for `Date -
/// Date`, a day apart here.
#[test]
fn an_audit_pools_nothing_into_nothing_and_steps_a_date_clock_in_milliseconds() {
    use online_polars::AuditTable;
    let s = spec(
        r#"{"name": "a", "model": {"type": "audit"}, "targets": ["x0"],
            "features": ["x0", "x1"], "clock": "d", "gap_cap": "30d"}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    let empty = bank.audit(0, None, AuditTable::Columns, true).unwrap();
    assert_eq!(empty.height(), 0, "nothing to pool");
    let n = 10;
    let mut df = rows(n, 12);
    df.put(
        Series::new("d".into(), (0..n as i32).collect::<Vec<_>>())
            .cast(&DataType::Date)
            .unwrap(),
    )
    .unwrap();
    bank.fit_predict(&df).unwrap();
    let clock = bank.audit(0, None, AuditTable::Clock, true).unwrap();
    let mean = clock.column("step_mean").unwrap();
    assert_eq!(mean.dtype(), &DataType::Duration(TimeUnit::Milliseconds));
    assert_eq!(
        mean.cast(&DataType::Int64).unwrap().i64().unwrap().get(0),
        Some(86_400_000)
    );
}

/// The averaged forecast where every slot's error is exactly 0 so far: the
/// ratio's limit is the argmin, every slot at 0 weighing 1, so it is their
/// common prediction, exactly.
#[test]
fn the_averaged_forecast_at_a_best_error_of_zero_is_the_exact_slots() {
    // A constant target: every slot predicts it exactly once it has a row.
    let n = 40;
    let mut df = rows(n, 13);
    df.put(Series::new("y".into(), vec![3.0; n])).unwrap();
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": [0.0, 1.0]},
            "targets": ["y"], "features": ["x0"], "half_life": 10.0,
            "emit_averaged": true, "emit_sigma": true}"#,
    );
    let out = Bank::new(vec![s]).unwrap().fit_predict(&df).unwrap();
    let avg = floats(&field(&out, 0, "pred_y__averaged"));
    let present: Vec<f64> = avg.iter().flatten().copied().collect();
    assert!(present.len() > 30, "{avg:?}");
    assert!(present.iter().all(|&p| p == 3.0), "{present:?}");
}

/// The averaged forecast where one slot fits exactly and the other does
/// not: the exact slot's error is 0, the limit of the ratio is the argmin,
/// and the forecast is the exact slot's prediction alone. Under a literal
/// `lam` at unit steps: that the exact slot's error comes out 0 at all is a
/// rounding, and under `half_life` it was one of Apple's `exp2` (a libm 1
/// ulp off left errors of 4e-16 on every row; docs/PLAN.md task 273).
#[test]
fn the_averaged_forecast_at_one_exact_slot_is_that_slots() {
    let n = 40;
    let mut df = rows(n, 16);
    let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
    df.put(Series::new("x0".into(), x)).unwrap();
    df.put(Series::new("y".into(), y)).unwrap();
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": [0.0, 1.0]},
            "targets": ["y"], "features": ["x0"], "lam": 0.933, "fit_intercept": false,
            "emit_averaged": true, "emit_sigma": true}"#,
    );
    let out = Bank::new(vec![s]).unwrap().fit_predict(&df).unwrap();
    let avg = floats(&field(&out, 0, "pred_y__averaged"));
    let exact = floats(&field(&out, 0, "pred_y__r0"));
    let sig0 = floats(&field(&out, 0, "sigma_y__r0"));
    let rows_at_zero: Vec<usize> = (0..n).filter(|&i| sig0[i] == Some(0.0)).collect();
    assert!(!rows_at_zero.is_empty(), "the exact slot's error: {sig0:?}");
    for i in rows_at_zero {
        assert_eq!(avg[i], exact[i], "row {i}");
    }
}

/// A formula target resolved to a value that is no number -- a division by
/// a zero -- stays a null in the data summary: every row fed is counted
/// once, usable or not, and the infinite ones are not usable.
#[test]
fn a_formula_target_resolved_to_no_number_stays_a_null_in_the_summary() {
    let n = 40;
    let mut df = rows(n, 14);
    let b: Vec<f64> = (0..n).map(|i| if i % 5 == 0 { 0.0 } else { 1.0 }).collect();
    df.put(Series::new("b".into(), b)).unwrap();
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"},
            "targets": [{"name": "fwd", "formula": ["/", ["rewm_mean", ["col", "x1"], {"half_life": 2.0, "window_size": 3.0}], ["col", "b"]]}],
            "features": ["x0"], "clock": "t", "gap_cap": 5.0, "half_life": 10.0,
            "embargo": 3.0}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&df).unwrap();
    let d = bank.describe(0, None).unwrap();
    let row = d
        .clone()
        .lazy()
        .filter(col("column").eq(lit("fwd")))
        .collect()
        .unwrap();
    let get = |name: &str| {
        row.column(name)
            .unwrap()
            .cast(&DataType::Int64)
            .unwrap()
            .i64()
            .unwrap()
            .get(0)
            .unwrap()
    };
    let (count, unusable) = (get("count"), get("unusable_count"));
    assert_eq!(count + unusable, n as i64, "{d}");
    // Eight rows divide by zero; the last rows' windows are still open.
    assert!(unusable >= 8, "{d}");
    assert!(count >= 25, "{d}");
}

/// A chunk's column is empty exactly when it has no rows.
#[test]
fn a_chunk_column_is_empty_when_it_has_no_rows() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0, "group": "g"}"#,
    );
    let mut df = rows(3, 17);
    df.put(Series::new("g".into(), ["a", "b", "a"])).unwrap();
    for (frame, empty) in [(df.clone(), false), (df.slice(0, 0), true)] {
        let chunk = online_polars::chunk_from_frame(&frame, std::slice::from_ref(&s)).unwrap();
        let col = chunk.key(&s, "group", "g").unwrap();
        assert_eq!(col.is_empty(), empty);
        assert_eq!(col.len(), frame.height());
    }
}

/// A spec's non-finite numbers are written as the words that read back.
#[test]
fn a_specs_non_finite_numbers_are_written_as_words() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": "nan"}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0}"#,
    );
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains(r#""ridge":"nan""#), "{text}");
    let back: Spec = serde_json::from_str(&text).unwrap();
    assert_eq!(serde_json::to_string(&back).unwrap(), text);
    // A negative integer ridge is read as a number, and refused as one.
    let e = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": -1}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0}"#,
    )
    .check()
    .unwrap_err();
    assert!(e.contains("ridge") && e.contains("-1"), "{e}");
}

/// Two groups of `n` rows each, keyed `0` then `1`, with `y = 3 x0 + x1`
/// (and the noise of [`rows`] left out), so a monotone close closes group 0
/// on the first row of group 1.
fn two_groups(n: usize, seed: u64) -> DataFrame {
    let mut df = rows(2 * n, seed);
    let x0: Vec<f64> = floats(df.column("x0").unwrap().as_materialized_series())
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let x1: Vec<f64> = floats(df.column("x1").unwrap().as_materialized_series())
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let y: Vec<f64> = x0.iter().zip(&x1).map(|(a, b)| 3.0 * a + b).collect();
    df.put(Series::new("y".into(), y)).unwrap();
    df.put(Series::new(
        "g".into(),
        (0..2 * n as i64).map(|i| i / n as i64).collect::<Vec<_>>(),
    ))
    .unwrap();
    df
}

/// The values of row `i` of a `list[f64]` column, `None` for a null row.
fn list_row(df: &DataFrame, name: &str, i: usize) -> Option<Vec<Option<f64>>> {
    df.column(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .list()
        .unwrap()
        .get_as_series(i)
        .map(|s| floats(&s))
}

/// A closed `marginal` row carries its pairs -- the moments, the
/// correlation, the slope, the t-statistics, the serial and the binned
/// scalars -- pair for pair and value for value what `marginal()` reports
/// on the same rows, as `test_a_marginal_spec_closes_with_its_pairs` and
/// `test_a_closed_marginal_row_carries_its_lags_and_bins` hold in Python.
/// The closed row is the only readout of a closed group, so a column it
/// dropped or emptied would be lost.
#[test]
fn a_closed_marginal_row_carries_the_pairs_marginal_reports() {
    let n = 60;
    let df = two_groups(n, 21);
    let model = r#""model": {"type": "marginal", "lags": [1, 2], "serial_rule": "truncated",
        "bin_edges": [[0.0], [-0.5, 0.5]]}, "targets": ["y"], "features": ["x0", "x1"],
        "half_life": 20.0"#;
    let mut closing = Bank::new(vec![spec(&format!(
        r#"{{"name": "m", {model}, "group": "g", "group_close": "monotone"}}"#
    ))])
    .unwrap();
    closing.fit_predict(&df).unwrap();
    let closed = closing.closed_groups(None, true).unwrap();
    assert_eq!(closed.height(), 1, "group 0 closed: {closed}");

    let mut plain = Bank::new(vec![spec(&format!(r#"{{"name": "m", {model}}}"#))]).unwrap();
    plain.fit_predict(&df.slice(0, n)).unwrap();
    let want = plain.marginal(0, None).unwrap();
    assert_eq!(want.height(), 2, "{want}");

    for col in [
        "weight_sum",
        "n_kish",
        "mean_x",
        "var_x",
        "mean_y",
        "var_y",
        "cov",
        "corr",
        "beta",
        "t_stat",
        "n_serial",
        "t_stat_serial",
        "phi_x",
        "phi_y",
        "split_gain",
        "split_at",
        "split_gain_t",
    ] {
        let w = floats(want.column(col).unwrap().as_materialized_series());
        assert_eq!(
            list_row(&closed, &format!("pair_{col}"), 0),
            Some(w),
            "{col}"
        );
    }
    // The values themselves, not only their agreement: the slope of `y` on
    // `x0` carries its 3, `n_kish` is a count of rows that are there, and
    // truncation fits no decay.
    let beta = list_row(&closed, "pair_beta", 0).unwrap();
    assert!((beta[0].unwrap() - 3.0).abs() < 0.5, "{beta:?}");
    let kish = list_row(&closed, "pair_n_kish", 0).unwrap();
    assert!(kish.iter().all(|k| k.unwrap() > 10.0), "{kish:?}");
    assert_eq!(
        list_row(&closed, "pair_phi_x", 0),
        Some(vec![None, None]),
        "truncated fits no decay"
    );
    let gain = list_row(&closed, "pair_split_gain", 0).unwrap();
    assert!(gain[0].unwrap() > 0.5, "x0 carries the target: {gain:?}");
}

/// Two closing marginals, one with lags and bins and one without: on the
/// row of the one that did not ask, the lag and bin columns are null --
/// the columns its `marginal()` would not have -- and on the other's they
/// hold its values, as `test_the_lag_and_bin_blocks_follow_the_specs_that_asked`
/// holds in Python.
#[test]
fn a_closed_marginal_row_nulls_the_blocks_its_spec_did_not_ask_for() {
    let n = 30;
    let df = two_groups(n, 22);
    let common = r#""targets": ["y"], "features": ["x0", "x1"], "half_life": 20.0,
        "group": "g", "group_close": "monotone""#;
    let mut bank = Bank::new(vec![
        spec(&format!(
            r#"{{"name": "plain", "model": {{"type": "marginal"}}, {common}}}"#
        )),
        spec(&format!(
            r#"{{"name": "asked", "model": {{"type": "marginal", "lags": [1],
                "serial_rule": "truncated", "bin_edges": [[0.0], [0.0]]}}, {common}}}"#
        )),
    ])
    .unwrap();
    bank.fit_predict(&df).unwrap();
    let closed = bank.closed_groups(None, true).unwrap();
    assert_eq!(closed.height(), 2, "group 0 closed in each spec: {closed}");
    let names: Vec<Option<&str>> = closed
        .column("spec")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .collect();
    let plain = names.iter().position(|s| *s == Some("plain")).unwrap();
    let asked = names.iter().position(|s| *s == Some("asked")).unwrap();
    assert!(list_row(&closed, "pair_corr", plain).is_some());
    for col in [
        "pair_lag_corr_xx",
        "pair_lag_corr_yy",
        "pair_lag_corr_xy",
        "pair_lag_corr_yx",
        "pair_bin_edges",
        "pair_bin_n",
    ] {
        let lists = closed.column(col).unwrap().list().unwrap().clone();
        assert!(lists.get_as_series(plain).is_none(), "{col} on plain");
        let got = lists.get_as_series(asked).expect(col);
        assert_eq!(got.len(), 2, "{col}: one per pair");
    }
    for col in [
        "pair_n_serial",
        "pair_t_stat_serial",
        "pair_split_gain",
        "pair_split_at",
    ] {
        assert_eq!(list_row(&closed, col, plain), None, "{col} on plain");
        let got = list_row(&closed, col, asked).expect(col);
        assert_eq!(got.len(), 2, "{col}: one per pair");
        assert!(got.iter().all(Option::is_some), "{col}: {got:?}");
    }
    // The asked row's lag-1 autocorrelation of `x0` is the one `marginal()`
    // reports for the same rows.
    let mut alone = Bank::new(vec![spec(
        r#"{"name": "asked", "model": {"type": "marginal", "lags": [1],
            "serial_rule": "truncated", "bin_edges": [[0.0], [0.0]]}, "targets": ["y"], "features": ["x0", "x1"],
            "half_life": 20.0}"#,
    )])
    .unwrap();
    alone.fit_predict(&df.slice(0, n)).unwrap();
    let want = alone.marginal(0, None).unwrap();
    assert_eq!(
        list_row(&closed, "pair_n_serial", asked),
        Some(floats(
            want.column("n_serial").unwrap().as_materialized_series()
        ))
    );
}

/// The averaged forecast over a grid of three ridges for each of two
/// targets is each target's own slots weighed by `exp(-eta (sigma² /
/// sigma²_best - 1))`, normalized, from the bank's own columns -- as
/// `test_the_weights_are_each_error_over_the_best` holds in Python. Every
/// slot counts, not only the first of each target, and the second target
/// reads its own slots, not the first target's.
#[test]
fn the_averaged_forecast_weighs_each_of_a_targets_own_slots() {
    let n = 400;
    // Noisy targets, so that no slot's error dwarfs another's and each
    // weighs in: `rows`' own `y` is so nearly exact that the best slot
    // takes every weight, and the average is that slot's forecast alone.
    let mut df = rows(n, 23);
    let noise = rows(n, 24);
    let col = |df: &DataFrame, c: &str| -> Vec<f64> {
        floats(df.column(c).unwrap().as_materialized_series())
            .into_iter()
            .map(Option::unwrap)
            .collect()
    };
    let (x0, x1) = (col(&df, "x0"), col(&df, "x1"));
    let (e1, e2) = (col(&noise, "x0"), col(&noise, "x1"));
    let y: Vec<f64> = (0..n).map(|i| 2.0 * x0[i] + 3.0 * e1[i]).collect();
    let y2: Vec<f64> = (0..n).map(|i| 5.0 - x1[i] + 2.0 * e2[i]).collect();
    df.put(Series::new("y".into(), y)).unwrap();
    df.put(Series::new("y2".into(), y2)).unwrap();
    let eta = 0.7;
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": [1e-8, 1.0, 100.0],
            "max_rows_between_solves": 1},
            "targets": ["y", "y2"], "features": ["x0", "x1"], "half_life": 50.0,
            "min_weight": 5.0, "emit_averaged": true, "average_eta": 0.7, "emit_sigma": true}"#,
    );
    let out = Bank::new(vec![s]).unwrap().fit_predict(&df).unwrap();
    for t in ["y", "y2"] {
        let slots = ["r1e-8", "r1", "r100"];
        let preds: Vec<Vec<Option<f64>>> = slots
            .iter()
            .map(|r| floats(&field(&out, 0, &format!("pred_{t}__{r}"))))
            .collect();
        let sigs: Vec<Vec<Option<f64>>> = slots
            .iter()
            .map(|r| floats(&field(&out, 0, &format!("sigma_{t}__{r}"))))
            .collect();
        let avg = floats(&field(&out, 0, &format!("pred_{t}__averaged")));
        let (mut checked, mut apart) = (0, 0);
        for i in 0..n {
            let ok: Vec<(f64, f64)> = (0..slots.len())
                .filter_map(|k| match (sigs[k][i], preds[k][i]) {
                    (Some(s), Some(p)) if s.is_finite() && p.is_finite() => Some((s * s, p)),
                    _ => None,
                })
                .collect();
            if ok.len() < slots.len() {
                continue;
            }
            let best = ok.iter().map(|o| o.0).fold(f64::INFINITY, f64::min);
            let w: Vec<f64> = ok
                .iter()
                .map(|o| (-eta * (o.0 / best - 1.0)).exp())
                .collect();
            let want = w.iter().zip(&ok).map(|(w, o)| w * o.1).sum::<f64>() / w.iter().sum::<f64>();
            let got = avg[i].unwrap_or_else(|| panic!("{t}: row {i} has no average"));
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{t} row {i}: {got} vs {want}"
            );
            checked += 1;
            // The slots disagree here, so a forecast of the first slot
            // alone would not pass for the average.
            if (want - ok[0].1).abs() > 1e-6 * want.abs().max(1.0) {
                apart += 1;
            }
        }
        assert!(checked > 300, "{t}: {checked} rows with every slot");
        assert!(apart > 300, "{t}: the slots agree on all but {apart} rows");
    }
}
