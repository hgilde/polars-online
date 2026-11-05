//! A limited feed returns exactly its limit and holds the rest of its last
//! row's points (task 321): a slice of the output, then a run resumed from
//! the state saved there, give the one run's grid wherever the slice ends.

use polars::prelude::*;

use super::held::{Cell, Held};
use super::{RefreshCols, RefreshTime};

/// Four series ticking in a seeded order, two groups, so with `pairs` one
/// tick often completes several of the six pairs' points at once. `i` is
/// the row, carried as a `keep` column, and `g` the group.
fn ticks(n: usize) -> DataFrame {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let names = ["a", "b", "c", "d"];
    let (mut series, mut t, mut v, mut g, mut i) = (vec![], vec![], vec![], vec![], vec![]);
    for row in 0..n {
        let r = next();
        series.push(names[(r % 4) as usize]);
        t.push(row as f64 * 0.5);
        v.push(((r >> 8) % 1000) as f64 / 10.0);
        g.push(if (r >> 20) % 5 == 0 { "y" } else { "x" });
        i.push(row as i64);
    }
    df!("series" => series, "t" => t, "v" => v, "g" => g, "i" => i).unwrap()
}

fn names() -> Vec<String> {
    ["a", "b", "c", "d"].map(str::to_string).to_vec()
}

const KEEP: &[String] = &[];

fn cols<'a>(group: bool, keep: &'a [String]) -> RefreshCols<'a> {
    RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: group.then_some("g"),
        keep,
    }
}

/// One run as the Python source drives it: `df` from row `from` in chunks of
/// `chunk` rows, each fed the points still wanted, stopping once `limit` are
/// out; at the input's end a frame of no rows takes what is still held. The
/// output, the state saved, and the input row the next run starts at.
fn run(
    state: Option<&[u8]>,
    pairs: bool,
    df: &DataFrame,
    from: usize,
    chunk: usize,
    limit: Option<usize>,
    c: &RefreshCols<'_>,
) -> (DataFrame, Vec<u8>, usize) {
    let mut rt = match state {
        Some(b) => RefreshTime::load_bytes(b, names(), pairs, c.group.is_some()).unwrap(),
        None => RefreshTime::new(names(), pairs).unwrap(),
    };
    let rest = df.slice(from as i64, df.height() - from);
    // The schema from a sampler of its own, as the source declares it: a
    // feed of this one would return what it holds.
    let mut out = RefreshTime::new(names(), pairs)
        .unwrap()
        .feed(&rest.clear(), c)
        .unwrap();
    let (mut seen, mut at, mut satisfied) = (0, 0, false);
    while at < rest.height() {
        let part = rest.slice(at as i64, chunk);
        at += part.height();
        let got = rt.feed_limited(&part, c, limit.map(|l| l - seen)).unwrap();
        assert!(limit.is_none_or(|l| seen + got.height() <= l));
        seen += got.height();
        out.vstack_mut(&got).unwrap();
        if limit.is_some_and(|l| seen >= l) {
            satisfied = true;
            break;
        }
    }
    if !satisfied {
        let got = rt
            .feed_limited(&rest.clear(), c, limit.map(|l| l - seen))
            .unwrap();
        out.vstack_mut(&got).unwrap();
    }
    (out, rt.save_bytes().unwrap(), from + rt.rows_fed)
}

/// Successive runs of `step` points each, every one resumed from the state
/// the one before saved, give the one run's grid; so does a single slice
/// ending at each point, then a run resumed after it. Joint and pairs,
/// grouped and not, with `keep`, in chunks of 1, 3 and the whole input.
#[test]
fn slices_and_resumed_runs_give_the_one_runs_grid() {
    let df = ticks(160);
    let keep = vec!["i".to_string()];
    for pairs in [false, true] {
        for group in [false, true] {
            let c = cols(group, &keep);
            let (whole, whole_state, _) = run(None, pairs, &df, 0, df.height(), None, &c);
            if pairs {
                // Several points a tick, so slices end inside ticks.
                let i = whole.column("i").unwrap().i64().unwrap();
                let shared = (1..i.len()).filter(|&r| i.get(r) == i.get(r - 1)).count();
                assert!(shared > 20, "{shared} points share a tick");
            }
            for chunk in [1, 3, df.height()] {
                for step in [1, 2, 5] {
                    let (mut parts, mut state, mut from) = (Vec::new(), None, 0);
                    loop {
                        let (got, s, next) =
                            run(state.as_deref(), pairs, &df, from, chunk, Some(step), &c);
                        let h = got.height();
                        parts.push(got);
                        (state, from) = (Some(s), next);
                        if h < step {
                            break;
                        }
                    }
                    let mut all = parts.remove(0);
                    for p in parts {
                        all.vstack_mut(&p).unwrap();
                    }
                    assert!(all.equals_missing(&whole), "{pairs} {group} {chunk} {step}");
                    assert_eq!(
                        state.unwrap(),
                        whole_state,
                        "{pairs} {group} {chunk} {step}"
                    );
                }
                for n in 0..=whole.height() {
                    let (head, s, next) = run(None, pairs, &df, 0, chunk, Some(n), &c);
                    assert!(head.equals_missing(&whole.head(Some(n))));
                    let (tail, _, _) = run(Some(&s), pairs, &df, next, chunk, None, &c);
                    let mut all = head;
                    all.vstack_mut(&tail).unwrap();
                    assert!(all.equals_missing(&whole), "{pairs} {group} {chunk} {n}");
                }
            }
        }
    }
}

/// The held points come out first in the same run too, with the next
/// chunk's points after them, and a feed limited below what is held
/// returns only held points and reads no row.
#[test]
fn a_feed_returns_the_held_points_first() {
    let df = ticks(160);
    let c = cols(false, KEEP);
    let mut one = RefreshTime::new(names(), true).unwrap();
    let whole = one.feed(&df, &c).unwrap();
    let t = whole.column("time_refresh").unwrap().f64().unwrap();
    // A tick completing at least three points: hold two of them.
    let r = (2..t.len())
        .find(|&r| t.get(r) == t.get(r - 1) && t.get(r) == t.get(r - 2))
        .expect("a tick completing three points");
    let mut rt = RefreshTime::new(names(), true).unwrap();
    let head = rt.feed_limited(&df, &c, Some(r - 1)).unwrap();
    assert_eq!(head.height(), r - 1);
    let held = rt.held.as_ref().unwrap().points.len();
    assert!(held >= 2, "{held}");
    let fed = rt.rows_fed;
    let rest = df.slice(fed as i64, df.height() - fed);
    let one_more = rt.feed_limited(&rest, &c, Some(1)).unwrap();
    assert_eq!(
        rt.rows_fed, fed,
        "no row read while held points satisfy the limit"
    );
    assert_eq!(rt.held.as_ref().unwrap().points.len(), held - 1);
    let tail = rt.feed(&rest, &c).unwrap();
    assert!(rt.held.is_none());
    let mut all = head;
    all.vstack_mut(&one_more).unwrap();
    all.vstack_mut(&tail).unwrap();
    assert!(all.equals_missing(&whole));
}

/// Every dtype's value comes back from its [`Cell`], through the state's
/// msgpack, as the column had it: the plain forms, a null, and the one-value
/// Arrow IPC of any other dtype.
#[test]
fn every_dtype_a_row_holds_comes_back_through_the_state() {
    let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
    let list = Series::new("list".into(), [Series::new("".into(), [1i64, 2, 3])]);
    let strukt = StructChunked::from_series(
        "strukt".into(),
        1,
        [
            Series::new("p".into(), [1.5f64]),
            Series::new("q".into(), ["text"]),
        ]
        .iter(),
    )
    .unwrap()
    .into_series();
    let cols: Vec<Column> = vec![
        Column::new("bool".into(), [true]),
        Column::new("i8".into(), [-3i8]),
        Column::new("i32".into(), [i32::MIN]),
        Column::new("i64".into(), [i64::MIN]),
        Column::new("u8".into(), [200u8]),
        Column::new("u64".into(), [u64::MAX]),
        Column::new("f32".into(), [0.1f32]),
        Column::new("f64".into(), [-0.0f64]),
        Column::new("str".into(), ["a string"]),
        Column::new("null_int".into(), [None::<i64>]),
        Column::new("date".into(), [19_000i32])
            .cast(&DataType::Date)
            .unwrap(),
        Int64Chunked::new("dt".into(), [1_700_000_000_123_456i64])
            .into_datetime(TimeUnit::Microseconds, tz)
            .into_column(),
        Column::new("dt_ns".into(), [1_700_000_000_123_456_789i64])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap(),
        Column::new("dur".into(), [-5i64])
            .cast(&DataType::Duration(TimeUnit::Milliseconds))
            .unwrap(),
        Column::new("time".into(), [3_600_000_000_001i64])
            .cast(&DataType::Time)
            .unwrap(),
        Column::new("cat".into(), ["level"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap(),
        list.into_column(),
        strukt.into_column(),
    ];
    let row = DataFrame::new(1, cols).unwrap();
    // The row inside a longer frame, so the slice is a row of a column.
    let mut frame = row.clone();
    frame.vstack_mut(&row).unwrap();
    frame.vstack_mut(&row).unwrap();
    let names: Vec<&str> = row.get_column_names().iter().map(|n| n.as_str()).collect();
    let held = Held::new(&frame, &names, 1, vec![(0, vec![1, 2], vec![0.5, 1.5])]).unwrap();
    let arrow = held
        .row
        .iter()
        .filter(|(_, c)| matches!(c, Cell::Arrow(_)))
        .map(|(n, _)| n.as_str())
        .collect::<Vec<_>>();
    assert_eq!(arrow, ["cat", "list", "strukt"]);
    let dt = held
        .row
        .iter()
        .find(|(n, _)| n == "dt")
        .map(|(_, c)| c.clone());
    assert_eq!(
        dt,
        Some(Cell::Datetime(
            1_700_000_000_123_456,
            super::held::Unit::Us,
            Some("Europe/Amsterdam".into())
        )),
        "{:?}",
        frame.column("dt").unwrap().dtype()
    );
    let bytes = rmp_serde::to_vec_named(&held).unwrap();
    let back: Held = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(back, held);
    assert_eq!(rmp_serde::to_vec_named(&back).unwrap(), bytes);
    let got = back.frame(&row, &names).unwrap();
    assert!(got.equals_missing(&row), "{got}\n{row}");
}

/// A held row resumed on another unit of a temporal clock, or another
/// width of integer, takes the input's dtype; one that cannot be cast, or
/// a column the state did not keep, is refused by name.
#[test]
fn a_held_row_takes_the_inputs_dtypes_or_is_refused_by_name() {
    let saved = df!("t" => [1_500i64], "k" => ["seven"]).unwrap();
    let mut saved = saved
        .lazy()
        .with_column(col("t").cast(DataType::Datetime(TimeUnit::Milliseconds, None)))
        .collect()
        .unwrap();
    saved.rechunk_mut();
    let held = Held::new(&saved, &["t", "k"], 0, vec![]).unwrap();
    let input = df!("t" => [0i64], "k" => ["x"])
        .unwrap()
        .lazy()
        .with_column(col("t").cast(DataType::Datetime(TimeUnit::Nanoseconds, None)))
        .collect()
        .unwrap();
    let got = held.frame(&input, &["t", "k"]).unwrap();
    assert_eq!(
        got.column("t").unwrap().dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, None)
    );
    let ns = got.column("t").unwrap().cast(&DataType::Int64).unwrap();
    assert_eq!(ns.i64().unwrap().get(0), Some(1_500_000_000));
    // A zoned clock on another unit keeps its zone.
    let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
    let zoned = |unit, v: i64| {
        DataFrame::new(
            1,
            vec![
                Int64Chunked::new("t".into(), [v])
                    .into_datetime(unit, tz.clone())
                    .into_column(),
            ],
        )
        .unwrap()
    };
    let held_zoned = Held::new(&zoned(TimeUnit::Milliseconds, 1_500), &["t"], 0, vec![]).unwrap();
    let got = held_zoned
        .frame(&zoned(TimeUnit::Nanoseconds, 0), &["t"])
        .unwrap();
    assert!(got.equals_missing(&zoned(TimeUnit::Nanoseconds, 1_500_000_000)));
    assert_eq!(
        got.column("t").unwrap().dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, tz.clone())
    );
    // A naive clock resumed on a zoned one takes the zone, the instant kept.
    let got = held
        .frame(&zoned(TimeUnit::Milliseconds, 0), &["t"])
        .unwrap();
    assert!(got.equals_missing(&zoned(TimeUnit::Milliseconds, 1_500)));
    assert_eq!(
        got.column("t").unwrap().dtype(),
        &DataType::Datetime(TimeUnit::Milliseconds, tz.clone())
    );
    let ints = df!("t" => [0i64], "k" => [0i32]).unwrap();
    let e = held.frame(&ints, &["k"]).unwrap_err().to_string();
    assert!(
        e.contains("\"k\" is a str") && e.contains("cannot hold"),
        "{e}"
    );
    let e = held.frame(&input, &["t", "z"]).unwrap_err().to_string();
    assert!(e.contains("saved without a \"z\" column"), "{e}");
}

/// A state holding points is version 2; a damaged held set is refused.
#[test]
fn a_state_holding_points_is_version_two_and_checked_on_load() {
    let df = ticks(160);
    let c = cols(false, KEEP);
    let mut rt = RefreshTime::new(names(), true).unwrap();
    let whole = rt.feed(&df, &c).unwrap();
    let t = whole.column("time_refresh").unwrap().f64().unwrap();
    let r = (1..t.len()).find(|&r| t.get(r) == t.get(r - 1)).unwrap();
    let mut rt = RefreshTime::new(names(), true).unwrap();
    rt.feed_limited(&df, &c, Some(r)).unwrap();
    let bytes = rt.save_bytes().unwrap();
    let mut file: super::RefreshFile = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(file.version, super::REFRESH_VERSION_HELD);
    let load = |f: &super::RefreshFile| {
        RefreshTime::load_bytes(&rmp_serde::to_vec_named(f).unwrap(), names(), true, false)
    };
    assert!(load(&file).is_ok());
    let good = file.held.clone();
    file.version = super::REFRESH_VERSION;
    assert_eq!(
        load(&file).err().unwrap(),
        "refresh_time: the state is damaged: version 1 with held points"
    );
    file.version = super::REFRESH_VERSION_HELD;
    for damage in [
        |h: &mut Held| h.points.clear(),
        |h: &mut Held| h.points[0].0 = 6,
        |h: &mut Held| h.points[0].1.push(1),
        |h: &mut Held| {
            h.points[0].2.pop();
        },
    ] {
        let mut h = good.clone().unwrap();
        damage(&mut h);
        file.held = Some(h);
        let e = load(&file).err().unwrap();
        assert!(
            e.contains("its held points are not points of 6 grids of 2 series"),
            "{e}"
        );
    }
}
