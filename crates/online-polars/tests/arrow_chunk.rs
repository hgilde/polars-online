//! The bank fed as Arrow (docs/PLAN.md task 86).
//!
//! The point of the Arrow entry points is that a caller holding Arrow arrays
//! can feed the bank without building a `DataFrame` first. The golden suites
//! reach `fit_predict_arrow` only through the polars pair, which proves the
//! delegation but not the claim: these build a chunk by hand, from arrays
//! nothing in polars ever touched, and hold the result to what the same data
//! as a frame produces.

use online_polars::{ArrowChunk, ArrowCol, Bank, Spec, chunk_from_frame};
use polars::prelude::*;
use polars_arrow::array::{Float64Array, MutableBinaryViewArray, Utf8ViewArray};

fn spec() -> Spec {
    serde_json::from_str(
        r#"{
            "name": "m",
            "model": {"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "halflife": 60.0,
            "max_dclock": 30.0,
            "weight": "w",
            "group": "g",
            "min_periods": 5.0
        }"#,
    )
    .unwrap()
}

/// The same numbers, held once, so the frame and the chunk cannot drift.
struct Data {
    g: Vec<&'static str>,
    t: Vec<f64>,
    x0: Vec<Option<f64>>,
    x1: Vec<f64>,
    y: Vec<Option<f64>>,
    w: Vec<f64>,
}

/// Two groups, nulls in a feature and in the target, a clock that advances
/// per group -- the shape the bank tests use elsewhere.
fn data(n: usize) -> Data {
    let mut s = 1234u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut d = Data {
        g: Vec::new(),
        t: Vec::new(),
        x0: Vec::new(),
        x1: Vec::new(),
        y: Vec::new(),
        w: Vec::new(),
    };
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let g = i % 2;
        d.g.push(if g == 0 { "g0" } else { "g1" });
        clocks[g] += 1.0 + lcg().abs() * 5.0;
        d.t.push(clocks[g]);
        let (a, b) = (lcg(), lcg());
        d.x0.push((i % 17 != 5).then_some(a));
        d.x1.push(b);
        d.y.push((i % 23 != 7).then_some(if g == 0 { 2.0 * a - b } else { -a } + 0.01 * lcg()));
        d.w.push(0.5 + lcg().abs());
    }
    d
}

fn frame(d: &Data) -> DataFrame {
    df!(
        "g" => d.g.clone(),
        "t" => d.t.clone(),
        "x0" => d.x0.clone(),
        "x1" => d.x1.clone(),
        "y" => d.y.clone(),
        "w" => d.w.clone(),
    )
    .unwrap()
}

fn text(vals: &[&str]) -> Utf8ViewArray {
    let mut b = MutableBinaryViewArray::<str>::with_capacity(vals.len());
    for v in vals {
        b.push(Some(*v));
    }
    b.into()
}

fn nums(vals: &[f64]) -> Float64Array {
    Float64Array::from_iter(vals.iter().map(|v| Some(*v)))
}

fn opt_nums(vals: &[Option<f64>]) -> Float64Array {
    Float64Array::from_iter(vals.iter().copied())
}

/// The chunk built by hand: no frame, no dtype inference, no cast.
fn chunk(d: &Data) -> ArrowChunk {
    let names: Vec<PlSmallStr> = ["g", "t", "x0", "x1", "y", "w"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    let cols: Vec<(PlSmallStr, ArrowCol)> = vec![
        ("g".into(), ArrowCol::Str(text(&d.g))),
        ("t".into(), ArrowCol::F64(nums(&d.t))),
        ("x0".into(), ArrowCol::F64(opt_nums(&d.x0))),
        ("x1".into(), ArrowCol::F64(nums(&d.x1))),
        ("y".into(), ArrowCol::F64(opt_nums(&d.y))),
        ("w".into(), ArrowCol::F64(nums(&d.w))),
    ];
    ArrowChunk::new(d.t.len(), cols, names).unwrap()
}

/// Every struct array equals the struct column the frame path produced,
/// nulls included.
fn same(from_frame: &[Column], from_arrow: Vec<online_polars::StructArray>) {
    assert_eq!(from_frame.len(), from_arrow.len());
    assert!(!from_arrow.is_empty(), "the bank returned nothing");
    for (c, st) in from_frame.iter().zip(from_arrow) {
        let s = Series::from_arrow(c.name().clone(), Box::new(st)).unwrap();
        let want = c.as_materialized_series();
        assert_eq!(want.len(), s.len());
        assert!(
            want.equals_missing(&s),
            "the Arrow path and the frame path disagree on {:?}",
            c.name()
        );
    }
}

#[test]
fn a_hand_built_arrow_chunk_learns_what_the_same_frame_does() {
    let d = data(500);
    let mut by_frame = Bank::new(vec![spec()]).unwrap();
    let mut by_arrow = Bank::new(vec![spec()]).unwrap();
    let want = by_frame.fit_predict(&frame(&d)).unwrap();
    let got = by_arrow.fit_predict_arrow(&chunk(&d)).unwrap();
    same(&want, got);
}

#[test]
fn the_frame_path_is_the_arrow_path_behind_the_adapter() {
    let d = data(300);
    let df = frame(&d);
    let specs = vec![spec()];
    let mut direct = Bank::new(vec![spec()]).unwrap();
    let mut adapted = Bank::new(vec![spec()]).unwrap();
    let want = direct.fit_predict(&df).unwrap();
    let got = adapted
        .fit_predict_arrow(&chunk_from_frame(&df, &specs).unwrap())
        .unwrap();
    same(&want, got);
}

#[test]
fn predict_arrow_scores_what_predict_does() {
    let d = data(400);
    let df = frame(&d);
    // Two banks taught identically, then scored down the two paths.
    let mut by_frame = Bank::new(vec![spec()]).unwrap();
    let mut by_arrow = Bank::new(vec![spec()]).unwrap();
    by_frame.fit_predict(&df).unwrap();
    by_arrow.fit_predict_arrow(&chunk(&d)).unwrap();
    let want = by_frame.predict(&df).unwrap();
    let got = by_arrow.predict_arrow(&chunk(&d)).unwrap();
    same(&want, got);
}

/// A chunk whose columns disagree about how many rows there are is refused
/// when it is built, not read half-way through.
#[test]
fn a_ragged_chunk_is_refused() {
    let d = data(20);
    let cols: Vec<(PlSmallStr, ArrowCol)> = vec![
        ("t".into(), ArrowCol::F64(nums(&d.t))),
        ("x0".into(), ArrowCol::F64(nums(&d.x1[..5]))),
    ];
    let names: Vec<PlSmallStr> = ["t", "x0"].iter().map(|s| (*s).into()).collect();
    let err = ArrowChunk::new(d.t.len(), cols, names).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("x0"), "{text}");
    assert!(text.contains("rows"), "{text}");
}
