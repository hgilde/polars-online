//! Task 194 (review round 4, docs/PLAN.md §18): what the bank keeps between
//! chunks -- the form of each key column (N22) -- where `coef` rides
//! without a cadence (S2), the clock range in the clock's own dtype (N18)
//! and the readers' keys (N21). `tests/test_state_frames.py` holds the
//! frames as Python reads them.

use online_polars::{
    ArrowChunk, ArrowCol, Bank, ChunkOut, Float64Array, GroupKey, Int64Array, Spec, Stream,
    Utf8ViewArray,
};
use polars::prelude::*;

fn spec(json: &str) -> Spec {
    serde_json::from_str(json).unwrap()
}

/// An `ew_ridge` on `x`, keyed by `g`, with `extra` its further
/// `"key": value` pairs.
fn keyed_spec(extra: &str) -> Spec {
    spec(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge", "max_rows_between_solves": 1}},
            "targets": ["y"], "features": ["x"], "half_life": 10.0, "min_weight": 1.0,
            "group": "g"{extra}}}"#
    ))
}

fn rows(n: usize, key: impl Fn(usize) -> AnyValue<'static>) -> DataFrame {
    let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 11.0).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 0.5).collect();
    let g: Vec<AnyValue> = (0..n).map(key).collect();
    df!("x" => x, "y" => y)
        .unwrap()
        .hstack(&[Column::new(
            "g".into(),
            Series::from_any_values("g".into(), &g, true).unwrap(),
        )])
        .unwrap()
}

/// Review round 4, PC1 and PA7: a key is its value's text, so an Int64 key
/// `1` and a Float64 key `1.0` were two groups, and a column cast between
/// chunks started every group over in silence. The bank keeps each key
/// column's form from its first chunk, in the state too, and refuses
/// another by name, as the windows state does
/// (`windows_frame.rs::a_state_refuses_a_group_column_of_another_dtype`).
#[test]
fn a_group_column_of_another_form_is_refused_by_name() {
    let s = keyed_spec("");
    let ints = rows(8, |i| AnyValue::Int64((i % 2) as i64));
    let floats = rows(8, |i| AnyValue::Float64((i % 2) as f64));
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    bank.fit_predict(&ints).unwrap();
    let saved = bank.save_bytes().unwrap();
    let err = bank.fit_predict(&floats).unwrap_err().to_string();
    assert!(
        err.contains(r#"group column "g" was i64 and is now f64"#),
        "{err}"
    );
    assert_eq!(
        bank.save_bytes().unwrap(),
        saved,
        "a refused chunk moves nothing"
    );
    // Kept in the state, and read by `predict`.
    let loaded = Bank::load_bytes(&saved, Some(std::slice::from_ref(&s))).unwrap();
    let err = loaded.predict(&floats).unwrap_err().to_string();
    assert!(err.contains("was i64 and is now f64"), "{err}");
    // The same form in another width goes on.
    let narrow = rows(8, |i| AnyValue::Int32((i % 2) as i32));
    let mut bank = Bank::load_bytes(&saved, Some(&[s])).unwrap();
    bank.fit_predict(&narrow).unwrap();
    assert_eq!(bank.groups()[0].len(), 2, "one group per key");
}

/// An Arrow caller's chunk carries no polars dtype; the form is the one
/// the chunk holds the key in: an integer array, or text.
#[test]
fn an_arrow_chunk_keeps_the_form_its_key_array_has() {
    let s = keyed_spec("");
    let names = vec!["x", "y", "g"];
    let x = Float64Array::from_slice([0.1, 0.2, 0.3, 0.4]);
    let y = Float64Array::from_slice([0.7, 0.9, 1.1, 1.3]);
    let chunk = |g: ArrowCol| {
        ArrowChunk::new(
            4,
            vec![
                ("x", ArrowCol::F64(x.clone())),
                ("y", ArrowCol::F64(y.clone())),
                ("g", g),
            ],
            names.clone(),
        )
        .unwrap()
    };
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict_arrow(&chunk(ArrowCol::I64(Int64Array::from_slice([1, 2, 1, 2]))))
        .unwrap();
    let text = Utf8ViewArray::from_slice([Some("1"), Some("2"), Some("1"), Some("2")]);
    let err = bank
        .fit_predict_arrow(&chunk(ArrowCol::Str(text)))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(r#"group column "g" was i64 and is now str"#),
        "{err}"
    );
}

/// A session read as `"1"` and then `"1.0"` was a session change on every
/// group: under `session_gap = "reset"` a cold model (PC-battery2).
#[test]
fn a_session_column_of_another_form_is_refused_by_name() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"],
            "half_life": 10.0, "session": "g", "session_gap": "reset"}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&rows(8, |_| AnyValue::Int64(1))).unwrap();
    let err = bank
        .fit_predict(&rows(8, |_| AnyValue::Float64(1.0)))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(r#"session column "g" was i64 and is now f64"#),
        "{err}"
    );
}

/// A column of nulls with no type of its own carries no form: it is the
/// null key whatever the column becomes.
#[test]
fn a_null_typed_key_column_records_no_form() {
    let mut bank = Bank::new(vec![keyed_spec("")]).unwrap();
    bank.fit_predict(&rows(4, |_| AnyValue::Null)).unwrap();
    bank.fit_predict(&rows(4, |i| AnyValue::Int64(i as i64)))
        .unwrap();
    let err = bank
        .fit_predict(&rows(4, |i| AnyValue::String(["a", "b", "c", "d"][i])))
        .unwrap_err()
        .to_string();
    assert!(err.contains("was i64 and is now str"), "{err}");
}

/// `n` rows of `x0`, `x1`, `y`, `w`, no clock.
fn wide(n: usize) -> DataFrame {
    let x0: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 13.0 - 0.5).collect();
    let x1: Vec<f64> = (0..n).map(|i| ((i * 5) % 17) as f64 / 17.0 - 0.5).collect();
    let y: Vec<f64> = x0.iter().zip(&x1).map(|(a, b)| 2.0 * a - b).collect();
    df!("x0" => x0, "x1" => x1, "y" => y, "w" => vec![1.0; n]).unwrap()
}

fn ridge() -> Spec {
    spec(
        r#"{"name": "m", "model": {"type": "ewridge", "ridge": 1e-6, "max_rows_between_solves": 1},
            "targets": ["y"], "features": ["x0", "x1"], "weight": "w", "half_life": 500.0,
            "min_weight": 5.0}"#,
    )
}

fn coef_rows(out: &[Column]) -> Vec<usize> {
    let coef = out[0].struct_().unwrap().field_by_name("coef").unwrap();
    (0..coef.len())
        .filter(|&i| coef.get(i).unwrap() != AnyValue::Null)
        .collect()
}

/// Review round 4, PB2: without a cadence `coef` rides on the group's last
/// accepted row of the chunk, where a chunk whose last row of the group was
/// skipped carried none. The last accepted row may sit in an earlier run
/// than the chunk's last ([`ChunkOut::run_rows`]): a narrow `ew_ridge` runs
/// 65,520 rows at a time, and here the last run is every row skipped.
#[test]
fn coef_rides_on_the_last_accepted_row_across_runs() {
    let s = ridge();
    let stream = Stream::new(&s).unwrap();
    let run = ChunkOut::run_rows(&s, stream.n_models(), stream.n_slots());
    let n = 2 * run + 100;
    let skip_from = 2 * run;
    let mut df = wide(n);
    let x0: Vec<Option<f64>> = (0..n)
        .map(|i| (i < skip_from).then(|| ((i * 7) % 13) as f64 / 13.0 - 0.5))
        .collect();
    df.with_column(Column::new("x0".into(), x0)).unwrap();
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    let out = bank.fit_predict(&df).unwrap();
    assert_eq!(coef_rows(&out), vec![skip_from - 1]);
    // The chunk's coefficients are the fit after that row, as `coef()` has it.
    let field = out[0].struct_().unwrap().field_by_name("coef").unwrap();
    let AnyValue::List(row) = field.get(skip_from - 1).unwrap() else {
        panic!("a list");
    };
    let read: Vec<f64> = bank.coef(0, None).unwrap()[0].coef.clone().expect("solved");
    assert_eq!(row.f64().unwrap().to_vec_null_aware().left().unwrap(), read);
    // A chunk of skipped rows alone writes none.
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&df.slice(0, 50)).unwrap();
    let none = bank.fit_predict(&df.slice(skip_from as i64, 20)).unwrap();
    assert_eq!(coef_rows(&none), Vec::<usize>::new());
}

/// Review round 4, N18 (SF1, AP13): the clock range in `summary`, `groups`
/// and the closed frame is in the clock column's own dtype, exactly, as
/// `scored_clock` is. A double of seconds since 1970 was 64 ns off at 2024.
#[test]
fn the_clock_range_is_the_clock_columns_own_to_the_nanosecond() {
    let base: i64 = 1_704_187_800_126_456_001;
    let t: Vec<i64> = (0..6).map(|i| base + 1_234_567 * i).collect();
    let clock = Series::new("t".into(), t.clone())
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    let df = df!(
        "g" => [0i64, 0, 0, 1, 1, 1],
        "x" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "y" => [0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
    )
    .unwrap()
    .hstack(&[clock.into()])
    .unwrap();
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"],
            "group": "g", "group_close": "monotone", "clock": "t", "half_life": "1d",
            "gap_cap": "10d"}"#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&df).unwrap();
    let ns = |c: &Column| -> Vec<Option<i64>> {
        assert_eq!(
            c.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None),
            "{}",
            c.name()
        );
        let raw = c.as_materialized_series().to_physical_repr().into_owned();
        let ca = raw.i64().unwrap();
        (0..ca.len()).map(|i| ca.get(i)).collect()
    };
    let summary = bank.summary(0, None).unwrap();
    assert_eq!(ns(summary.column("clock_min").unwrap()), vec![Some(t[3])]);
    assert_eq!(ns(summary.column("clock_max").unwrap()), vec![Some(t[5])]);
    assert_eq!(ns(summary.column("last_clock").unwrap()), vec![Some(t[5])]);
    let groups = bank.groups_table(&[0]).unwrap();
    assert_eq!(ns(groups.column("last_clock").unwrap()), vec![Some(t[5])]);
    let closed = bank.closed_groups(None, true).unwrap();
    assert_eq!(ns(closed.column("clock_min").unwrap()), vec![Some(t[0])]);
    assert_eq!(ns(closed.column("clock_max").unwrap()), vec![Some(t[2])]);
    assert_eq!(
        closed.column("rows_fed").unwrap().dtype(),
        &DataType::UInt64
    );
}

/// Task 200: an integer clock's range comes out in its own integer dtype,
/// exactly, in `summary`, `groups` and the closed frame, and in
/// `scored_clock`: an `Int64` of epoch nanoseconds near 1.79e18, which a
/// double would round to 256, an `Int32` and a `UInt32`. A chunk whose clock
/// is a float, or an integer of another width, is refused by name, before
/// anything moves, in `fit_predict` and in `predict`.
#[test]
fn an_integer_clock_is_held_and_shown_in_its_own_dtype() {
    let s = spec(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"],
            "group": "g", "group_close": "monotone", "clock": "t", "half_life": 50.0,
            "gap_cap": 1e6, "emit_clocks": true}"#,
    );
    let frame = |t: Series| {
        df!(
            "g" => [0i64, 0, 0, 1, 1, 1],
            "x" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "y" => [0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
        )
        .unwrap()
        .hstack(&[t.into()])
        .unwrap()
    };
    let base: i64 = 1_790_000_000_000_000_000;
    let offs = [0i64, 1, 8, 108, 109, 116];
    let cases = [
        (DataType::Int64, base),
        (DataType::Int32, 2_000_000_000),
        (DataType::UInt32, 4_000_000_000),
    ];
    for (dtype, at) in cases {
        let t: Vec<i64> = offs.iter().map(|o| at + o).collect();
        let df = frame(Series::new("t".into(), t.clone()).cast(&dtype).unwrap());
        let mut bank = Bank::new(vec![s.clone()]).unwrap();
        let out = bank.fit_predict(&df).unwrap();
        let ints = |c: &Column| -> Vec<Option<i64>> {
            assert_eq!(c.dtype(), &dtype, "{}", c.name());
            let ca = c.cast(&DataType::Int64).unwrap();
            ca.i64().unwrap().iter().collect()
        };
        let summary = bank.summary(0, None).unwrap();
        assert_eq!(ints(summary.column("clock_min").unwrap()), vec![Some(t[3])]);
        assert_eq!(
            ints(summary.column("last_clock").unwrap()),
            vec![Some(t[5])]
        );
        let groups = bank.groups_table(&[0]).unwrap();
        assert_eq!(ints(groups.column("last_clock").unwrap()), vec![Some(t[5])]);
        let closed = bank.closed_groups(None, true).unwrap();
        assert_eq!(ints(closed.column("clock_min").unwrap()), vec![Some(t[0])]);
        assert_eq!(ints(closed.column("clock_max").unwrap()), vec![Some(t[2])]);
        let fields = out[0].as_materialized_series().struct_().unwrap().clone();
        let scored = fields.field_by_name("scored_clock").unwrap().into_column();
        let want: Vec<Option<i64>> = t.iter().map(|v| Some(*v)).collect();
        assert_eq!(ints(&scored), want);
    }
    let t: Vec<i64> = offs.iter().map(|o| base + o).collect();
    let df = frame(Series::new("t".into(), t));
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&df.slice(0, 3)).unwrap();
    let saved = bank.save_bytes().unwrap();
    for (dtype, want) in [
        (DataType::Float64, "was i64 and is now f64"),
        (DataType::Int32, "was i64 and is now i32"),
    ] {
        let shifted = Series::new("t".into(), offs[3..].to_vec())
            .cast(&dtype)
            .unwrap();
        let mut other = df.slice(3, 3);
        other.with_column(shifted.into()).unwrap();
        let err = bank.fit_predict(&other).unwrap_err().to_string();
        assert!(
            err.contains(&format!(r#"clock column "t" {want}"#)),
            "{err}"
        );
        let err = bank.predict(&other).unwrap_err().to_string();
        assert!(err.contains(want), "{err}");
        assert_eq!(
            bank.save_bytes().unwrap(),
            saved,
            "a refused chunk moves nothing"
        );
    }
    bank.fit_predict(&df.slice(3, 3)).unwrap();
}

/// Review round 4, N21 (PA10): a reader narrows to keys, the null group
/// among them, which a string could not name; an integer column's keys are
/// listed as numbers, the null group first, where they sorted as text.
#[test]
fn the_readers_take_keys_and_list_integer_keys_as_numbers() {
    let keys: Vec<AnyValue> = (0..22)
        .map(|i| match i % 11 {
            0 => AnyValue::Null,
            k => AnyValue::Int64(k as i64),
        })
        .collect();
    let mut bank = Bank::new(vec![keyed_spec("")]).unwrap();
    bank.fit_predict(&rows(22, |i| keys[i].clone())).unwrap();
    let order: Vec<Option<String>> = bank.groups()[0].iter().map(|(k, ..)| k.0.clone()).collect();
    let want: Vec<Option<String>> = std::iter::once(None)
        .chain((1..=10).map(|k| Some(k.to_string())))
        .collect();
    assert_eq!(order, want);
    let null = [GroupKey(None)];
    let summary = bank.summary(0, Some(&null)).unwrap();
    assert_eq!(summary.height(), 1);
    assert_eq!(summary.column("group").unwrap().null_count(), 1);
    let some = [
        GroupKey(Some("10".into())),
        GroupKey(None),
        GroupKey(Some("2".into())),
    ];
    let coef: Vec<Option<String>> = bank
        .coef(0, Some(&some))
        .unwrap()
        .into_iter()
        .map(|c| c.group.0)
        .collect();
    assert_eq!(coef, vec![None, Some("2".into()), Some("10".into())]);
}
