//! The margin column (task 343): read row-major from each kind of array it
//! may arrive as, refused by name, laid out class-major for LightGBM, and
//! read from an Arrow IPC file. Each library's trees around a margin are
//! held to its own route by `tests/test_gbdt_margin.py`.

use polars::prelude::*;
use polars_arrow::array::{Array, FixedSizeListArray, ListArray, PrimitiveArray};
use polars_arrow::datatypes::{ArrowDataType, Field};
use polars_arrow::offset::OffsetsBuffer;

use super::frames::{Frame, Segment, WidthFrom, column_name, margin_rows, segments, segments_of};
use super::lgb::class_major;
use super::*;

/// One feature `x0`, the label `y`, and the margin `m`.
fn layout() -> Layout {
    Layout {
        names: vec!["x0".into()],
        kinds: vec![Kind::Float],
        label: "y".into(),
        weight: None,
        margin: Some("m".into()),
    }
}

fn floats(v: &[f64]) -> Box<dyn Array> {
    PrimitiveArray::from_slice(v).boxed()
}

/// A record batch of `n` rows of `x0` and `y` beside the margin `m`.
fn batch(m: Box<dyn Array>) -> Frame {
    let n = m.len();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Frame::Cols {
        n,
        cols: vec![floats(&x), floats(&x), m],
    }
}

/// A list array of `rows` (each `None` for a null row), its values `T`.
fn list<T: polars_arrow::types::NativeType>(rows: &[Option<Vec<Option<T>>>]) -> ListArray<i64> {
    let mut offsets = vec![0i64];
    let mut values: Vec<Option<T>> = Vec::new();
    for r in rows {
        values.extend(r.iter().flatten().cloned());
        offsets.push(values.len() as i64);
    }
    let child = PrimitiveArray::<T>::from(values).boxed();
    let validity = rows.iter().map(Option::is_some).collect();
    let dtype = ListArray::<i64>::default_datatype(child.dtype().clone());
    ListArray::new(
        dtype,
        OffsetsBuffer::try_from(offsets).unwrap(),
        child,
        Some(validity),
    )
}

/// Every segment's margin, row-major, the width it set, or the first error.
fn read(frame: Frame, width: Option<usize>) -> Result<(Vec<f64>, Option<usize>), GbdtError> {
    let lay = layout();
    let mut w = width.map(|k| (k, WidthFrom::Reference));
    let (mut all, mut buf) = (Vec::new(), Vec::new());
    let mut row = 0;
    for seg in segments_of(frame, &lay)? {
        margin_rows(&seg, &lay, "base_margin", "ref", row, &mut w, &mut buf)?;
        all.extend_from_slice(&buf);
        row += seg.n as u64;
    }
    Ok((all, w.map(|w| w.0)))
}

fn read_df(df: &DataFrame) -> Vec<(Vec<f64>, usize)> {
    let lay = layout();
    let mut w = None;
    let mut buf = Vec::new();
    let mut row = 0;
    segments(df, &lay)
        .unwrap()
        .iter()
        .map(|seg: &Segment| {
            margin_rows(seg, &lay, "base_margin", "ref", row, &mut w, &mut buf).unwrap();
            row += seg.n as u64;
            (buf.clone(), seg.n)
        })
        .collect()
}

#[test]
fn a_margin_is_read_row_major_from_every_kind_it_arrives_as() {
    // A number a row, of any numeric type.
    let (v, k) = read(batch(floats(&[0.5, -1.0, 2.0])), None).unwrap();
    assert_eq!((v, k), (vec![0.5, -1.0, 2.0], Some(1)));
    let ints = PrimitiveArray::<i16>::from_slice([3, -4]).boxed();
    assert_eq!(read(batch(ints), None).unwrap().0, vec![3.0, -4.0]);
    // A list of float32, two values a row.
    let l = list::<f32>(&[
        Some(vec![Some(1.5), Some(2.0)]),
        Some(vec![Some(-0.25), Some(4.0)]),
    ]);
    let (v, k) = read(batch(l.boxed()), None).unwrap();
    assert_eq!((v, k), (vec![1.5, 2.0, -0.25, 4.0], Some(2)));
    // A fixed-size list, three values a row, sliced past its first row.
    let child = floats(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let dtype = ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::Float64, true)),
        3,
    );
    let fixed = FixedSizeListArray::new(dtype, 3, child, None)
        .sliced(1, 2)
        .boxed();
    let (v, k) = read(batch(fixed), None).unwrap();
    assert_eq!((v, k), (vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0], Some(3)));
}

#[test]
fn a_list_margin_in_chunks_is_read_from_each_chunk_s_own_rows() {
    // Two chunks of a List column: the second segment's offsets point
    // into the second chunk's values, not from 0.
    let rows = |a: usize, b: usize| -> DataFrame {
        let m: Vec<Series> = (a..b)
            .map(|i| Series::new("".into(), [i as f64, -(i as f64)]))
            .collect();
        let x: Vec<f64> = (a..b).map(|i| i as f64).collect();
        df!("x0" => x.clone(), "y" => x, "m" => m).unwrap()
    };
    let mut df = rows(0, 3);
    df.vstack_mut(&rows(3, 5)).unwrap();
    let got = read_df(&df);
    assert_eq!(got.len(), 2, "a segment a chunk");
    assert_eq!(got[0], (vec![0.0, -0.0, 1.0, -1.0, 2.0, -2.0], 3));
    assert_eq!(got[1], (vec![3.0, -3.0, 4.0, -4.0], 2));
    // polars' Array, which a fixed-size list carries.
    let arr = df
        .column("m")
        .unwrap()
        .cast(&DataType::Array(Box::new(DataType::Float64), 2))
        .unwrap();
    let mut df2 = df.clone();
    df2.with_column(arr).unwrap();
    let flat: Vec<f64> = read_df(&df2).into_iter().flat_map(|s| s.0).collect();
    assert_eq!(
        flat,
        vec![0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    );
}

#[test]
fn a_margin_is_refused_by_name_at_a_null_a_non_finite_value_or_another_width() {
    let err = |f: Frame, w: Option<usize>| read(f, w).unwrap_err().to_string();
    let null = "po.gbdt.xgboost_matrix: the base_margin column `m` has a null at row 1 (counting \
                from 0), and every row needs a margin. Replace the null before the build, with 0 \
                where the row has no prediction to start from.";
    let m = PrimitiveArray::<f64>::from([Some(1.0), None]).boxed();
    assert_eq!(err(batch(m), None), null);
    let l = list::<f64>(&[Some(vec![Some(1.0)]), None]);
    assert_eq!(err(batch(l.boxed()), None), null, "a null row");
    let l = list::<f64>(&[
        Some(vec![Some(1.0), Some(2.0)]),
        Some(vec![Some(1.0), None]),
    ]);
    assert_eq!(err(batch(l.boxed()), None), null, "a null value in a row");
    assert_eq!(
        err(batch(floats(&[0.0, 0.0, f64::NEG_INFINITY])), None),
        "po.gbdt.xgboost_matrix: the base_margin column `m` holds -inf at row 2 (counting from \
         0), and a margin must be finite. Replace it before the build, with 0 where the row has \
         no prediction to start from."
    );
    let l = list::<f64>(&[Some(vec![Some(f64::NAN)])]);
    assert!(err(batch(l.boxed()), None).contains("holds NaN at row 0"));
    let l = list::<f64>(&[
        Some(vec![Some(1.0), Some(2.0)]),
        Some(vec![Some(1.0), Some(2.0)]),
        Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
    ]);
    assert_eq!(
        err(batch(l.clone().boxed()), None),
        "po.gbdt.xgboost_matrix: the base_margin column `m` holds 3 values at row 2 (counting from 0) and 2 \
         at row 0, and every row needs the same number. Give every row one value, or one for each \
         class."
    );
    assert_eq!(
        err(batch(l.boxed()), Some(1)),
        "po.gbdt.xgboost_matrix: the base_margin column `m` holds 2 values at row 0 (counting from 0), and \
         the ref's base_margin holds 1 value a row. Give every row as many values as the ref's \
         base_margin has."
    );
    assert_eq!(
        err(batch(floats(&[1.0])), Some(3)),
        "po.gbdt.xgboost_matrix: the base_margin column `m` holds 1 value at row 0 (counting from 0), and \
         the ref's base_margin holds 3 values a row. Give every row as many values as the ref's \
         base_margin has."
    );
    let l = list::<f64>(&[Some(vec![]), Some(vec![Some(1.0)])]);
    assert_eq!(
        err(batch(l.boxed()), None),
        "po.gbdt.xgboost_matrix: the base_margin column `m` holds no values at row 0 (counting from 0), and a \
         margin needs one. Give every row one value, or one for each class."
    );
    // The reference's width admits rows of that width.
    let l = list::<f64>(&[Some(vec![Some(1.0), Some(2.0)])]);
    assert_eq!(
        read(batch(l.boxed()), Some(2)).unwrap(),
        (vec![1.0, 2.0], Some(2))
    );
}

#[test]
fn the_width_set_by_the_first_row_holds_over_later_segments() {
    let mut a = df!("x0" => [0.0, 1.0], "y" => [0.0, 1.0], "m" => [0.5, 0.25]).unwrap();
    let b = df!("x0" => [2.0], "y" => [2.0], "m" => [0.75]).unwrap();
    a.vstack_mut(&b).unwrap();
    let got = read_df(&a);
    assert_eq!(got, vec![(vec![0.5, 0.25], 2), (vec![0.75], 1)]);
}

#[test]
fn a_frame_with_a_margin_counts_it_and_takes_a_list_only_there() {
    let lay = layout();
    assert_eq!(lay.width(), 3);
    assert_eq!(lay.margin_at(), Some(2));
    assert_eq!(column_name(&lay, 2), "m");
    let weighted = Layout {
        weight: Some("w".into()),
        ..layout()
    };
    assert_eq!((weighted.width(), weighted.margin_at()), (4, Some(3)));
    assert_eq!(
        (column_name(&weighted, 2), column_name(&weighted, 3)),
        ("w", "m")
    );
    assert_eq!(
        Layout {
            margin: None,
            ..layout()
        }
        .margin_at(),
        None
    );
    // A list is a margin, never a feature or a label.
    let l = || list::<f64>(&[Some(vec![Some(1.0)])]).boxed();
    for cols in [
        vec![l(), floats(&[0.0]), floats(&[0.0])],
        vec![floats(&[0.0]), l(), floats(&[0.0])],
    ] {
        let e = segments_of(Frame::Cols { n: 1, cols }, &lay)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("not a numeric column the libraries read"), "{e}");
    }
    // A margin of strings is refused by name.
    let s = polars_arrow::array::Utf8ViewArray::from_slice_values(["a"]).boxed();
    let e = segments_of(
        Frame::Cols {
            n: 1,
            cols: vec![floats(&[0.0]), floats(&[0.0]), s],
        },
        &lay,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(e.contains("column `m` arrived as"), "{e}");
    let e = segments(&df!("x0" => [0.0], "y" => [0.0]).unwrap(), &lay)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("(the features, the label, the margin)"), "{e}");
}

#[test]
fn class_major_lays_each_class_s_rows_together() {
    // Two rows of three classes into a dataset of four rows, at row 1.
    let mut dst = vec![0.0; 12];
    class_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 4, 1, &mut dst);
    assert_eq!(
        dst,
        vec![0.0, 1.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0, 6.0, 0.0]
    );
}

#[test]
fn an_arrow_ipc_file_hands_over_its_margin() {
    use polars_arrow::datatypes::ArrowSchema;
    use polars_arrow::io::ipc::write::{FileWriter, WriteOptions};
    use polars_arrow::record_batch::RecordBatchT;
    use std::sync::Arc;

    let dir = std::env::temp_dir().join(format!("gbdt-margin-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, m: Box<dyn Array>| {
        let x = floats(&[0.0, 1.0]);
        let schema: ArrowSchema = [
            Field::new("x0".into(), ArrowDataType::Float64, true),
            Field::new("y".into(), ArrowDataType::Float64, true),
            Field::new("m".into(), m.dtype().clone(), true),
        ]
        .into_iter()
        .map(|f| (f.name.clone(), f))
        .collect();
        let schema = Arc::new(schema);
        let path = dir.join(name);
        let mut w = FileWriter::try_new(
            std::fs::File::create(&path).unwrap(),
            schema.clone(),
            None,
            WriteOptions { compression: None },
        )
        .unwrap();
        w.write(&RecordBatchT::new(2, schema, vec![x.clone(), x, m]), None)
            .unwrap();
        w.finish().unwrap();
        path
    };
    let read_file = |path| -> Result<(Vec<f64>, Option<usize>), GbdtError> {
        let mut src = IpcFrames::new(vec![path], layout(), vec![None])?;
        let mut frames = Vec::new();
        src.pass(&mut |f| {
            frames.push(f);
            true
        })?;
        read(frames.pop().unwrap(), None)
    };
    let ints = write(
        "ints.arrow",
        PrimitiveArray::<i32>::from_slice([7, -8]).boxed(),
    );
    assert_eq!(read_file(ints).unwrap(), (vec![7.0, -8.0], Some(1)));
    let l = list::<f32>(&[
        Some(vec![Some(0.5), Some(1.0)]),
        Some(vec![Some(2.0), Some(3.0)]),
    ]);
    let lists = write("lists.arrow", l.boxed());
    assert_eq!(
        read_file(lists).unwrap(),
        (vec![0.5, 1.0, 2.0, 3.0], Some(2))
    );
    let b = polars_arrow::array::BooleanArray::from_slice([true, false]).boxed();
    let bools = write("bools.arrow", b);
    let e = read_file(bools).unwrap_err().to_string();
    assert!(
        e.contains("column `m` is Boolean in the Arrow IPC file"),
        "{e}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
