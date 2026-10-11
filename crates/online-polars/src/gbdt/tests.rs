//! The parts of the component that need neither library: segments, the
//! reader and the sample. The libraries' own routes are held to by the
//! Python suite (`tests/test_gbdt.py`), where both are installed.

use std::num::NonZeroUsize;

use polars::prelude::*;

use super::frames::{Frame, Num, Reader, segments};
use super::sample::{Planted, Sampler};
use super::*;

fn layout(p: usize, weight: bool) -> Layout {
    Layout {
        names: (0..p).map(|j| format!("x{j}")).collect(),
        kinds: vec![Kind::Float; p],
        label: "y".into(),
        weight: weight.then(|| "w".into()),
    }
}

fn frame(rows: std::ops::Range<i64>) -> DataFrame {
    let x0: Vec<Option<f64>> = rows
        .clone()
        .map(|i| (i % 7 != 3).then_some(i as f64 * 0.5))
        .collect();
    let x1: Vec<i32> = rows.clone().map(|i| (i % 11) as i32).collect();
    let y: Vec<f64> = rows.map(|i| i as f64).collect();
    df!("x0" => x0, "x1" => x1, "y" => y).unwrap()
}

fn pool() -> rayon::ThreadPool {
    conversion_pool(2).unwrap()
}

fn df_of(f: Frame) -> DataFrame {
    match f {
        Frame::Df(df) => df,
        Frame::Cols { .. } => panic!("a polars frame"),
    }
}

#[test]
fn segments_cut_where_any_column_s_chunks_end_and_copy_nothing() {
    // x0 in chunks of 5 and 7, x1 and y in one: segments of 5 and 7.
    let mut a = frame(0..5);
    a.vstack_mut(&frame(5..12)).unwrap();
    let b = frame(0..12);
    let df = DataFrame::new(
        12,
        vec![
            a.column("x0").unwrap().clone(),
            b.column("x1").unwrap().clone(),
            b.column("y").unwrap().clone(),
        ],
    )
    .unwrap();
    let segs = segments(&df, &layout(2, false)).unwrap();
    assert_eq!(segs.iter().map(|s| s.n).collect::<Vec<_>>(), vec![5, 7]);
    let x1 = b.column("x1").unwrap().as_materialized_series().chunks()[0].clone();
    match (segs[1].num(1), Num::of(x1.as_ref()).unwrap()) {
        (Num::I32(s), Num::I32(whole)) => {
            assert_eq!(
                s.values().as_ptr() as usize,
                whole.values().as_ptr() as usize + 5 * 4,
                "a slice of the frame's own buffer"
            );
        }
        _ => panic!("x1 is Int32"),
    }
    // Nulls by validity, starting at bit 0 in every segment.
    for (seg, r0) in segs.iter().zip([0usize, 5]) {
        let v = seg.num(0);
        for r in 0..seg.n {
            let i = r0 + r;
            assert_eq!(v.f32_at(r).is_nan(), i % 7 == 3, "row {i}");
        }
        assert_eq!(v.validity().unwrap().as_slice().1, 0);
    }
}

#[test]
fn a_column_of_the_wrong_kind_is_refused_by_name() {
    let df = df!("x0" => ["a", "b"], "x1" => [1i32, 2], "y" => [1.0, 2.0]).unwrap();
    let e = segments(&df, &layout(2, false)).err().unwrap().to_string();
    assert!(e.contains("`x0`") && e.contains("str"), "{e}");
    // The label must arrive as Float64.
    let df = df!("x0" => [1.0, 2.0], "x1" => [1i32, 2], "y" => [1i64, 2]).unwrap();
    let e = segments(&df, &layout(2, false)).err().unwrap().to_string();
    assert!(e.contains("`y`"), "{e}");
}

#[test]
fn rows_f32_fill_every_type_and_nan_for_null() {
    let df = frame(0..9000);
    let segs = segments(&df, &layout(2, false)).unwrap();
    let seg = &segs[0];
    let mut buf = vec![0f32; seg.n * 2];
    for j in 0..2 {
        seg.num(j).fill_f32(&mut buf[2 * 4096..], 2, j, 4096);
    }
    for r in 4096..seg.n {
        let i = r as i64;
        let x0 = buf[r * 2];
        assert_eq!(x0.is_nan(), i % 7 == 3);
        if i % 7 != 3 {
            assert_eq!(x0, (i as f64 * 0.5) as f32);
        }
        assert_eq!(buf[r * 2 + 1], (i % 11) as f32);
    }
}

#[test]
fn the_reader_reads_every_pass_whole_and_restarts_part_way() {
    let frames: Vec<DataFrame> = (0..10).map(|b| frame(b * 100..(b + 1) * 100)).collect();
    let mut r = Reader::new(Box::new(VecFrames(frames)), 2);
    let mut rows = 0;
    while let Some(f) = r.next().unwrap() {
        rows += df_of(f).height();
    }
    assert_eq!(rows, 1000);
    assert!(r.next().unwrap().is_none(), "the end stays the end");
    r.restart();
    let first = df_of(r.next().unwrap().unwrap());
    assert_eq!(first.column("y").unwrap().f64().unwrap().get(0), Some(0.0));
    r.restart();
    let mut rows = 0;
    while let Some(f) = r.next().unwrap() {
        rows += df_of(f).height();
    }
    assert_eq!(rows, 1000);
    assert_eq!(r.passes, 3);
}

struct Failing;
impl Frames for Failing {
    fn pass(&mut self, out: &mut dyn FnMut(Frame) -> bool) -> Result<(), GbdtError> {
        out(Frame::Df(frame(0..3)));
        Err(GbdtError::msg("the source broke"))
    }
}

#[test]
fn a_source_s_error_reaches_the_build() {
    let mut r = Reader::new(Box::new(Failing), 1);
    assert!(r.next().unwrap().is_some());
    let e = r.next().err().unwrap();
    assert_eq!(e.to_string(), "the source broke");
}

#[test]
fn a_plan_is_read_on_the_streaming_engine_once_per_pass() {
    let plan = frame(0..10_000).lazy();
    let mut r = Reader::new(
        Box::new(PlanFrames::new(plan, NonZeroUsize::new(1000).unwrap())),
        2,
    );
    for _ in 0..2 {
        let mut ys: Vec<f64> = Vec::new();
        while let Some(f) = r.next().unwrap() {
            let df = df_of(f);
            ys.extend(df.column("y").unwrap().f64().unwrap().into_no_null_iter());
        }
        assert_eq!(ys, (0..10_000).map(|i| i as f64).collect::<Vec<_>>());
        r.restart();
    }
}

#[test]
fn the_sample_is_the_same_rows_however_the_stream_is_cut() {
    let lay = layout(2, false);
    let pool = pool();
    let draw = |cut: i64| {
        let mut s = Sampler::new(lay.kinds.clone(), 500, 7, false);
        let mut at = 0;
        while at < 20_000 {
            let end = (at + cut).min(20_000);
            s.push_frame(&frame(at..end), &lay, &pool).unwrap();
            at = end;
        }
        assert_eq!(s.rows_seen(), 20_000);
        assert_eq!(s.rows_kept(), 500);
        s.ref_matrix(64, &pool)
    };
    let one = draw(20_000);
    assert_eq!(one, draw(1));
    assert_eq!(one, draw(333));
    // A short stream is kept whole.
    let mut s = Sampler::new(lay.kinds.clone(), 500, 7, false);
    s.push_frame(&frame(0..40), &lay, &pool).unwrap();
    assert_eq!(s.rows_kept(), 40);
    match &s.planted(256, &pool)[1] {
        Planted::Values(v) => assert_eq!(v, &(0..11).map(|c| c as f32).collect::<Vec<_>>()),
        other => panic!("{other:?}"),
    }
}

/// `frame(rows)` with a weight column `w`, `w(i)` for row `i`.
fn weighted(rows: std::ops::Range<i64>, w: impl Fn(i64) -> f64) -> DataFrame {
    let mut df = frame(rows.clone());
    let w: Vec<f64> = rows.map(w).collect();
    df.with_column(Column::new("w".into(), w)).unwrap();
    df
}

fn draw(lay: &Layout, k: usize, cut: i64, n: i64, w: &dyn Fn(i64) -> f64) -> Sampler {
    let pool = pool();
    let mut s = Sampler::new(lay.kinds.clone(), k, 7, lay.weight.is_some());
    let mut at = 0;
    while at < n {
        let end = (at + cut).min(n);
        let df = if lay.weight.is_some() {
            weighted(at..end, w)
        } else {
            frame(at..end)
        };
        s.push_frame(&df, lay, &pool).unwrap();
        at = end;
    }
    s
}

#[test]
fn a_weighted_sample_is_the_same_rows_however_the_stream_is_cut() {
    let lay = layout(2, true);
    let pool = pool();
    let w = |i: i64| 0.25 + (i % 13) as f64 + if i % 97 == 0 { 300.0 } else { 0.0 };
    let one = draw(&lay, 500, 20_000, 20_000, &w).ref_matrix(64, &pool);
    assert_eq!(one, draw(&lay, 500, 1, 20_000, &w).ref_matrix(64, &pool));
    assert_eq!(one, draw(&lay, 500, 333, 20_000, &w).ref_matrix(64, &pool));
}

#[test]
fn equal_weights_draw_and_plant_the_unweighted_sample_to_the_bit() {
    let pool = pool();
    let plain = draw(&layout(2, false), 500, 333, 20_000, &|_| 1.0).ref_matrix(64, &pool);
    for x in [1.0, 3.7, 1e-3, 1e300] {
        let s = draw(&layout(2, true), 500, 333, 20_000, &|_| x);
        assert_eq!(plain, s.ref_matrix(64, &pool), "{x}");
    }
    // A stream shorter than the sample too.
    let plain = draw(&layout(2, false), 500, 7, 300, &|_| 1.0).ref_matrix(64, &pool);
    assert_eq!(
        plain,
        draw(&layout(2, true), 500, 7, 300, &|_| 0.5).ref_matrix(64, &pool)
    );
}

#[test]
fn a_row_heavier_than_tau_is_always_kept_and_weight_zero_never_counts() {
    let lay = layout(2, true);
    let pool = pool();
    // Rows 1000, 2000, .., 19000 carry a weight far above tau: all 19 kept.
    // x0 is i / 2 (a null at i % 7 == 3), so those rows hold every edge.
    let s = draw(&lay, 200, 999, 20_000, &|i| {
        if i % 1000 == 0 && i > 0 { 1e9 } else { 1.0 }
    });
    assert_eq!(s.rows_kept(), 200);
    match &s.planted(5, &pool)[0] {
        // 16 of them hold a value (x0 is null at rows 4000, 11000 and
        // 18000), 500, 1000, 1500, 2500, .., each nearly 1/16 of the weight:
        // the edges at 1/5 .. 4/5 of it fall on the 4th, 7th, 10th and 13th
        // of them, and the 183 light rows' weight, under 1e-4 of the total,
        // cannot move an edge.
        Planted::Values(v) => assert_eq!(&v[1..], &[2500.0, 4000.0, 6000.0, 7500.0]),
        other => panic!("{other:?}"),
    }
    // Weight 0 on every row of x0 >= 100 (i >= 200): those rows place no
    // edge, though the stream is shorter than the sample.
    let s = draw(&lay, 5_000, 333, 2_000, &|i| {
        if i >= 200 { 0.0 } else { 2.0 }
    });
    assert_eq!(s.rows_kept(), 2_000);
    match &s.planted(16, &pool)[0] {
        Planted::Values(v) => assert!(v.iter().all(|&e| e < 100.0), "{v:?}"),
        other => panic!("{other:?}"),
    }
    // No weight at all: one value, as for a feature never seen.
    let s = draw(&lay, 50, 333, 2_000, &|_| 0.0);
    assert_eq!(s.planted(16, &pool)[0], Planted::Values(vec![0.0]));
}

#[test]
fn the_sample_refuses_a_negative_weight_by_row() {
    let lay = layout(2, true);
    let pool = pool();
    let mut s = Sampler::new(lay.kinds.clone(), 50, 7, true);
    s.push_frame(&weighted(0..100, |_| 1.0), &lay, &pool)
        .unwrap();
    let e = s
        .push_frame(
            &weighted(100..200, |i| if i == 150 { -1.0 } else { 1.0 }),
            &lay,
            &pool,
        )
        .unwrap_err();
    assert!(e.to_string().contains("holds -1.0 at row 150"), "{e}");
}

#[test]
fn a_category_plants_its_levels() {
    let lay = Layout {
        kinds: vec![Kind::Float, Kind::Category { levels: 4 }],
        ..layout(2, false)
    };
    let pool = pool();
    let mut s = Sampler::new(lay.kinds.clone(), 100, 0, false);
    s.push_frame(&frame(0..1000), &lay, &pool).unwrap();
    let (m, rows) = s.ref_matrix(256, &pool);
    let cat: Vec<f32> = (0..rows).map(|r| m[r * 2 + 1]).collect();
    assert_eq!(&cat[..4], &[0.0, 1.0, 2.0, 3.0]);
    assert!(
        cat[4..].iter().all(|&c| c == 3.0),
        "padded with the last level"
    );
}

mod ipc {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use polars_arrow::array::{BooleanArray, DictionaryArray, PrimitiveArray, Utf8ViewArray};
    use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
    use polars_arrow::io::ipc::write::{FileWriter, WriteOptions};
    use polars_arrow::record_batch::RecordBatchT;

    use super::super::frames::{Frame, Num, segments_of};
    use super::super::*;

    /// One record batch: a dictionary column over `dict`, a Boolean and an
    /// Int64 label.
    fn write(path: &Path, dict: &[&str], keys: Vec<Option<u8>>) {
        let n = keys.len();
        let values = Utf8ViewArray::from_slice_values(dict).boxed();
        let cat = DictionaryArray::try_from_keys(PrimitiveArray::<u8>::from(keys), values, false)
            .unwrap();
        let b = BooleanArray::from_slice((0..n).map(|i| i % 2 == 0).collect::<Vec<_>>());
        let y = PrimitiveArray::<i64>::from_vec((0..n as i64).collect());
        let schema: ArrowSchema = [
            Field::new("cat".into(), cat.dtype().clone(), true),
            Field::new("b".into(), ArrowDataType::Boolean, true),
            Field::new("y".into(), ArrowDataType::Int64, true),
        ]
        .into_iter()
        .map(|f| (f.name.clone(), f))
        .collect();
        let schema = Arc::new(schema);
        let batch = RecordBatchT::new(n, schema.clone(), vec![cat.boxed(), b.boxed(), y.boxed()]);
        let mut w = FileWriter::try_new(
            File::create(path).unwrap(),
            schema,
            None,
            WriteOptions { compression: None },
        )
        .unwrap();
        w.write(&batch, None).unwrap();
        w.finish().unwrap();
    }

    fn layout() -> Layout {
        Layout {
            names: vec!["b".into(), "cat".into()],
            kinds: vec![Kind::Bool, Kind::Category { levels: 3 }],
            label: "y".into(),
            weight: None,
        }
    }

    fn read(path: &Path) -> Result<Vec<Frame>, GbdtError> {
        let cats = vec![None, Some(vec!["a".into(), "b".into(), "c".into()])];
        let mut src = IpcFrames::new(vec![path.to_path_buf()], layout(), cats)?;
        let mut out = Vec::new();
        src.pass(&mut |f| {
            out.push(f);
            true
        })?;
        Ok(out)
    }

    #[test]
    fn a_record_batch_arrives_in_the_layout_s_order_as_the_libraries_read_it() {
        let dir = std::env::temp_dir().join(format!("gbdt-ipc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A dictionary that is the Enum's categories in order hands over its
        // keys; one in another order has each value looked up.
        for (name, dict, want) in [
            (
                "identity.arrow",
                ["a", "b", "c"],
                [0.0f32, 1.0, f32::NAN, 2.0],
            ),
            ("permuted.arrow", ["c", "a", "b"], [2.0, 0.0, f32::NAN, 1.0]),
        ] {
            let path = dir.join(name);
            write(&path, &dict, vec![Some(0), Some(1), None, Some(2)]);
            let frames = read(&path).unwrap();
            assert_eq!(frames.len(), 1);
            let segs = segments_of(frames.into_iter().next().unwrap(), &layout()).unwrap();
            let seg = &segs[0];
            let b: Vec<f32> = (0..4).map(|i| seg.num(0).f32_at(i)).collect();
            assert_eq!(b, [1.0, 0.0, 1.0, 0.0], "{name}: the Boolean as 0 and 1");
            let cat: Vec<f32> = (0..4).map(|i| seg.num(1).f32_at(i)).collect();
            for (g, w) in cat.iter().zip(want) {
                assert!(*g == w || (g.is_nan() && w.is_nan()), "{name}: {cat:?}");
            }
            match seg.num(2) {
                Num::F64(y) => assert_eq!(y.values().as_slice(), &[0.0, 1.0, 2.0, 3.0]),
                _ => panic!("{name}: the label as Float64"),
            }
        }
        let path = dir.join("unknown.arrow");
        write(&path, &["a", "z", "c"], vec![Some(0), Some(1)]);
        let e = read(&path).err().unwrap().to_string();
        assert!(e.contains("`cat` holds \"z\""), "{e}");
        let e = IpcFrames::new(vec![dir.join("absent.arrow")], layout(), vec![None, None])
            .unwrap()
            .pass(&mut |_| true)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("absent.arrow"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
