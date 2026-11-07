//! Each refusal of a value or a column of the wrong type carries the kind
//! `is_wrong_type` reads, and no other refusal does (task 355): the Python
//! extension raises `polars_online.WrongTypeError` for it, a `TypeError`,
//! where every other refusal is a `ValueError`. The text is checked too, so
//! a test that found the kind on some other error of the same call would
//! fail here.

use online_polars::{
    Bank, RefreshCols, RefreshTime, Spec, WindowsConfig, WindowsRun, chunk_from_frame,
    is_wrong_type,
};
use polars::prelude::*;

const N: usize = 12;

fn spec_with(extra: &str) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": ["x0"]{extra}}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// [`spec_with`] at a half-life of 10 clock units.
fn spec(extra: &str) -> Spec {
    spec_with(&format!(r#", "half_life": 10.0{extra}"#))
}

fn base() -> DataFrame {
    let t: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let x0: Vec<f64> = (0..N).map(|i| (i % 5) as f64).collect();
    let y: Vec<f64> = (0..N).map(|i| (i % 3) as f64).collect();
    df!("t" => t, "x0" => x0, "y" => y).unwrap()
}

fn with(df: DataFrame, exprs: Vec<Expr>) -> DataFrame {
    df.lazy().with_columns(exprs).collect().unwrap()
}

fn seconds() -> Expr {
    (col("t").cast(DataType::Int64) * lit(1_000_000_000i64))
        .cast(DataType::Datetime(TimeUnit::Nanoseconds, None))
}

/// The bank's refusal of `df` under `spec`, through `fit_predict`, after
/// checking that the polars adapter alone gives the same error.
fn bank_refusal(spec: Spec, df: &DataFrame) -> PolarsError {
    let adapter =
        chunk_from_frame(df, std::slice::from_ref(&spec)).expect_err("the adapter took the frame");
    let e = Bank::new(vec![spec])
        .unwrap()
        .fit_predict(df)
        .expect_err("the bank took the frame");
    assert_eq!(adapter.to_string(), e.to_string());
    e
}

fn wrong(e: &PolarsError, says: &str) {
    assert!(is_wrong_type(e), "not marked as a wrong type: {e}");
    assert!(e.to_string().contains(says), "{e}");
}

fn not_wrong(e: &PolarsError, says: &str) {
    assert!(!is_wrong_type(e), "marked as a wrong type: {e}");
    assert!(e.to_string().contains(says), "{e}");
}

#[test]
fn a_text_column_read_as_a_number_is_a_wrong_type() {
    let text = |c: &str| with(base(), vec![col(c).cast(DataType::String)]);
    let e = bank_refusal(spec(""), &text("x0"));
    wrong(
        &e,
        r#"spec "m": feature column "x0" must be a number, but it is String"#,
    );
    let e = bank_refusal(spec(""), &text("y"));
    wrong(
        &e,
        r#"target column "y" must be a number, but it is String"#,
    );
    let df = with(base(), vec![lit("a").alias("w")]);
    let e = bank_refusal(spec(r#", "weight": "w""#), &df);
    wrong(
        &e,
        r#"weight column "w" must be a number, but it is String"#,
    );
    let e = bank_refusal(spec(r#", "clock": "t", "gap_cap": 5.0"#), &text("t"));
    wrong(&e, r#"clock column "t" must be a number, but it is String"#);
}

#[test]
fn a_key_with_no_text_or_order_is_a_wrong_type() {
    let df = with(base(), vec![concat_list([col("x0")]).unwrap().alias("g")]);
    let e = bank_refusal(spec(r#", "group": "g""#), &df);
    wrong(
        &e,
        r#"group column "g" is List(Float64), which cannot be read as text"#,
    );
    let df = with(base(), vec![col("x0").alias("g")]);
    let e = bank_refusal(spec(r#", "group": "g", "group_close": "monotone""#), &df);
    wrong(
        &e,
        r#"needs a group column it can order, but "g" is Float64"#,
    );
}

#[test]
fn a_clock_of_a_type_no_clock_takes_is_a_wrong_type() {
    let clocked = r#", "clock": "t", "gap_cap": 5.0"#;
    let df = with(base(), vec![col("t").cast(DataType::Int128)]);
    let e = bank_refusal(spec(clocked), &df);
    wrong(
        &e,
        r#"spec "m": clock column "t" is Int128, wider than the Int64 an integer clock is held in"#,
    );
    let df = with(
        base(),
        vec![
            (col("t").cast(DataType::Int64) * lit(1_000_000_000i64))
                .cast(DataType::Time)
                .alias("t"),
        ],
    );
    let e = bank_refusal(spec(clocked), &df);
    wrong(&e, r#"clock column "t" is a time of day"#);
}

#[test]
fn a_temporal_column_read_as_a_number_is_a_wrong_type() {
    let temporal = r#", "clock": "t", "half_life": "10s", "gap_cap": "5s""#;
    let df = with(base(), vec![seconds().alias("t")]);
    let text = format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": ["x0", "t"]{temporal}}}"#
    );
    let e = bank_refusal(serde_json::from_str(&text).unwrap(), &df);
    wrong(
        &e,
        r#"it reads column "t" as a feature, which must be a number"#,
    );
    // A formula target reading a temporal column that is not its clock.
    let df = with(base(), vec![seconds().alias("d")]);
    let text = r#"{"name": "m", "model": {"type": "ewridge"}, "half_life": 10.0,
        "targets": [{"name": "z", "formula": ["-",
            ["rewm_mean", ["col", "y"], {"half_life": 5.0, "window_size": 10.0}], ["col", "d"]]}],
        "features": ["x0"]}"#;
    let e = bank_refusal(serde_json::from_str(text).unwrap(), &df);
    wrong(
        &e,
        r#"target column "d" is Datetime("ns"), but a formula target reads numbers"#,
    );
}

#[test]
fn a_label_clock_of_another_type_than_the_clock_is_a_wrong_type() {
    let df = with(base(), vec![lit("a").alias("lc")]);
    let e = bank_refusal(
        spec(r#", "clock": "t", "gap_cap": 5.0, "label_clock": "lc""#),
        &df,
    );
    wrong(
        &e,
        r#"label_clock column "lc" is String, and clock column "t" is Float64"#,
    );
}

/// The refusals the same calls make for another reason keep their kind.
#[test]
fn a_refusal_of_a_value_or_a_combination_is_not_a_wrong_type() {
    // A duration on a number clock: two settings that do not go together.
    let e = bank_refusal(
        spec_with(r#", "clock": "t", "half_life": "10s", "gap_cap": "5s""#),
        &base(),
    );
    not_wrong(&e, "is a duration, but clock column \"t\" is Float64");
    // A cap finer than a temporal clock's step: a value out of range.
    let df = with(
        base(),
        vec![
            (col("t").cast(DataType::Int64) * lit(1_000i64))
                .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                .alias("t"),
        ],
    );
    let e = bank_refusal(
        spec_with(r#", "clock": "t", "half_life": "10s", "gap_cap": "1us""#),
        &df,
    );
    not_wrong(&e, "less than a millisecond");
    // A column the frame has not got.
    let e = Bank::new(vec![spec(r#", "weight": "w""#)])
        .unwrap()
        .fit_predict(&base())
        .unwrap_err();
    assert!(matches!(e, PolarsError::ColumnNotFound(_)), "{e}");
    assert!(!is_wrong_type(&e));
}

fn windows(json: &str) -> WindowsConfig {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("{json}: {e}"))
}

#[test]
fn a_window_run_marks_its_wrong_types() {
    let text = with(base(), vec![col("x0").cast(DataType::String)]);
    let r = WindowsRun::new(
        windows(
            r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x0"], {"half_life": 5.0}]}]}"#,
        ),
        text.schema(),
    )
    .err()
    .expect("a text input taken");
    assert!(r.is_wrong_type(), "{r}");
    assert!(
        r.contains("with_windows: f: ewm_mean's input must be a number, but it is String"),
        "{r}"
    );
    let r = WindowsRun::new(
        windows(r#"{"formulas": [{"name": "f", "tree": ["increment", ["col", "x0"]]}]}"#),
        text.schema(),
    )
    .err()
    .expect("a text increment taken");
    assert!(r.is_wrong_type(), "{r}");
    assert!(
        r.contains("an increment's input must be a number or a temporal column, but it is String"),
        "{r}"
    );
    // Another refusal of the same call is not.
    let r = WindowsRun::new(windows(r#"{"formulas": []}"#), base().schema())
        .err()
        .expect("no formula taken");
    assert!(!r.is_wrong_type(), "{r}");
    // An Int128 clock is refused as the rows are read.
    let wide = with(base(), vec![col("t").cast(DataType::Int128)]);
    let mut run = WindowsRun::new(
        windows(
            r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x0"], {"half_life": 5.0}]}],
                "clock": "t", "gap_cap": 5.0}"#,
        ),
        wide.schema(),
    )
    .unwrap();
    let e = run.feed(&wide, None).unwrap_err();
    wrong(
        &e,
        r#"with_windows: clock column "t" is Int128, wider than the Int64"#,
    );
}

#[test]
fn refresh_time_marks_its_wrong_types() {
    let df = |clock: DataType, value: DataType| {
        df!("series" => ["a", "b", "a"], "t" => [1i64, 2, 3], "v" => [1.0, 2.0, 3.0])
            .unwrap()
            .lazy()
            .with_columns([col("t").cast(clock), col("v").cast(value)])
            .collect()
            .unwrap()
    };
    let cols = RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: None,
        keep: &[],
    };
    let names = vec!["a".to_string(), "b".to_string()];
    let refuse = |d: DataFrame| {
        RefreshTime::new(names.clone(), false)
            .unwrap()
            .feed(&d, &cols)
            .unwrap_err()
    };
    let e = refuse(df(DataType::String, DataType::Float64));
    wrong(
        &e,
        r#"po.stream.refresh_time: clock column "t" must be a number or temporal"#,
    );
    let e = refuse(df(DataType::Int128, DataType::Float64));
    wrong(&e, r#"po.stream.refresh_time: clock column "t" is Int128"#);
    let e = refuse(df(DataType::Int64, DataType::String));
    wrong(
        &e,
        r#"po.stream.refresh_time: value column "v" must be a number"#,
    );
    let e = refuse(df(DataType::Time, DataType::Float64));
    wrong(
        &e,
        r#"po.stream.refresh_time: clock column "t" is a time of day"#,
    );
}
