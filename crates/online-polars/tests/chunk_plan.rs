//! Edge cases of the chunk plan (docs/PERFORMANCE.md P9-P11), on both sides
//! of `PAR_MIN_ROWS`.
//!
//! The bank lays a chunk out group after group (P9), builds each output
//! field in parallel (P10), and reads a chunk's columns in parallel, arrow
//! chunk by arrow chunk, bucketing an integer group key by value (P11) --
//! from `PAR_MIN_ROWS` rows up, on the calling thread below. None of that
//! may show. A frame gives the same output and leaves the same state however
//! its groups interleave, however its columns are chunked in memory, whatever
//! numeric dtype they arrive in, and on whichever side of the threshold each
//! chunk falls; an error names the frame's row, not the layout's; and a
//! refused chunk still leaves the bank untouched. tests/bank.rs pins the same
//! invariants on small frames, where every step runs on one thread.

use online_polars::{Bank, GroupKey, PAR_MIN_ROWS, Spec};
use polars::prelude::*;

/// A spec over the fixture's columns: `group` names the key column (none for
/// an ungrouped spec), `halflife` is the JSON value, `extra` more
/// `"key": value,` pairs -- `session_gap` among them, since the fixture has
/// a session column.
fn spec(name: &str, group: Option<&str>, halflife: &str, extra: &str) -> Spec {
    let g = group.map_or(String::new(), |g| format!(r#""group": "{g}","#));
    serde_json::from_str(&format!(
        r#"{{
            "name": "{name}",
            "model": {{"type": "ew_ridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": ["y"],
            "features": ["x0", "x1"],
            "clock": "t",
            "halflife": {halflife},
            "max_dclock": 30.0,
            "weight": "w",
            "session": "sess",
            {g}
            {extra}
            "min_periods": 5.0
        }}"#
    ))
    .unwrap()
}

/// The grouped specs every test runs, keyed on `group`: a plain one, whose
/// `coef` lands on each chunk's last row per group and so depends on the
/// chunking; one with every optional output on, two halflives (instances run
/// in parallel) and `coef_every = 1`, so that every one of its fields is
/// chunk-invariant; and the coupled drift path, where a break in either
/// instance resets both and the rows go one at a time, with a session
/// change resetting the stream as well.
fn grouped_specs(group: &str) -> Vec<Spec> {
    vec![
        spec("plain", Some(group), "60.0", r#""session_gap": 10.0,"#),
        spec(
            "full",
            Some(group),
            "[30.0, 120.0]",
            r#""session_gap": 10.0, "coef_every": 1, "emit_sigma": true,
               "emit_resid_z": true, "emit_metrics": true,
               "resid_quantiles": [0.5, 0.9], "emit_autocorr": true, "conformal": 0.9,
               "emit_drift": true, "emit_selected": true, "emit_averaged": true,"#,
        ),
        spec(
            "coupled",
            Some(group),
            "[30.0, 120.0]",
            r#""session_gap": "reset", "coef_every": 1, "emit_drift": true,
               "drift_action": "reset", "drift_threshold": 2.0,"#,
        ),
    ]
}

/// The grouped specs plus an ungrouped one, whose layout is always the
/// frame's own and which shares the chunk's task pool with the others.
fn all_specs(group: &str) -> Vec<Spec> {
    let mut specs = grouped_specs(group);
    specs.push(spec(
        "solo",
        None,
        "60.0",
        r#""session_gap": 10.0, "coef_every": 1,"#,
    ));
    specs
}

/// A stream whose groups interleave row by row, so that the bank has to lay
/// the chunk out. Group sizes are skewed -- a few big groups, a thin tail --
/// and their first-seen order is not their sorted order; two rows are groups
/// of their own (one of them the last row of the frame) and every 101st row
/// has a null key. `g` is the String key and `gi` an Int32 key for the same
/// partition; `id` is the frame row. A feature, the target and the weight
/// have nulls, the weight has zeros, the session changes at group-specific
/// rows and is null now and then; `xi` and `xb` are an integer and a Boolean
/// feature for the dtype test.
fn interleaved(n: usize, n_groups: usize, seed: u64) -> DataFrame {
    assert!(n_groups <= 97, "the label map is a bijection below 97");
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(12345);
    let mut unit = move || -> f64 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut id = Vec::with_capacity(n);
    let mut g: Vec<Option<String>> = Vec::with_capacity(n);
    let mut gi: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut x0: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut xi: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut xb: Vec<Option<bool>> = Vec::with_capacity(n);
    let mut y: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut w: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut sess: Vec<Option<String>> = Vec::with_capacity(n);
    // One clock per group: the `n_groups` skewed ones, two solos, the null key.
    let mut clocks = vec![0.0f64; n_groups + 3];
    for i in 0..n {
        let u = unit();
        let slot = if i == n / 3 || i + 1 == n {
            n_groups + usize::from(i + 1 == n)
        } else if i % 101 == 50 {
            n_groups + 2
        } else {
            // Cubed: most rows land in the first few groups.
            ((u * u * u) * n_groups as f64) as usize
        };
        let (label, key) = if slot == n_groups + 2 {
            (None, None)
        } else if slot >= n_groups {
            let k = slot - n_groups;
            (Some(format!("solo{k}")), Some(-1 - k as i32))
        } else {
            // 62 is coprime with 97, so distinct slots get distinct labels
            // and the first-seen order is not the sorted one.
            let m = (slot * 62) % 97;
            (Some(format!("g{m}")), Some(m as i32))
        };
        id.push(i as i64);
        g.push(label);
        gi.push(key);
        // Occasionally a jump past `max_dclock`, so the cap is exercised.
        clocks[slot] += 1.0 + 5.0 * unit() + if i % 331 == 100 { 200.0 } else { 0.0 };
        t.push(clocks[slot]);
        let a = 2.0 * unit() - 1.0;
        let b = 2.0 * unit() - 1.0;
        let c = (unit() * 7.0) as i32;
        let d = unit() > 0.5;
        x0.push(if i % 17 == 5 { None } else { Some(a) });
        x1.push(b);
        xi.push(if i % 29 == 11 { None } else { Some(c) });
        xb.push(if i % 53 == 30 { None } else { Some(d) });
        y.push(if i % 23 == 7 {
            None
        } else {
            let noise = 0.01 * (2.0 * unit() - 1.0);
            Some(2.0 * a - b + 0.5 * f64::from(c) + 0.3 * f64::from(u8::from(d)) + noise)
        });
        w.push(if i % 43 == 20 {
            None
        } else if i % 41 == 9 {
            Some(0.0)
        } else {
            Some(0.5 + unit())
        });
        sess.push(if i % 37 == 13 {
            None
        } else {
            Some(format!("s{}", (clocks[slot] / 150.0).floor()))
        });
    }
    let df = df!(
        "id" => id,
        "g" => g,
        "gi" => gi,
        "t" => t,
        "x0" => x0,
        "x1" => x1,
        "xi" => xi,
        "xb" => xb,
        "y" => y,
        "w" => w,
        "sess" => sess,
    )
    .unwrap();
    // The fixture must be what the tests say it is: some group's rows are not
    // one run (so the layout is a permutation, not the identity), a group has
    // one row, there are null keys and null inputs.
    let keys: Vec<Option<&str>> = df.column("g").unwrap().str().unwrap().iter().collect();
    let runs = 1 + keys.windows(2).filter(|p| p[0] != p[1]).count();
    let distinct = keys.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(runs > distinct, "n={n}: the groups must interleave");
    assert!(keys.contains(&Some("solo1")) && keys.last() == Some(&Some("solo1")));
    if n > 50 {
        assert!(keys.contains(&None), "n={n}: a null key");
    }
    df
}

/// The frame's own `g` mask for one key (`None` for the null key).
fn is_key(df: &DataFrame, key: Option<&str>) -> BooleanChunked {
    df.column("g")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|v| Some(v == key))
        .collect()
}

/// A String-keyed bank's groups respelled with the fixture's Int32 keys,
/// in the order [`Bank::groups`] would list them: a group's `g` and `gi`
/// values pair up row by row, the null key stays null.
fn as_int_keys(
    groups: &[(GroupKey, u64, Option<f64>)],
    df: &DataFrame,
) -> Vec<(GroupKey, u64, Option<f64>)> {
    let g = df.column("g").unwrap().str().unwrap();
    let gi = df.column("gi").unwrap().i32().unwrap();
    let mut int_of = std::collections::HashMap::new();
    for (s, i) in g.iter().zip(gi.iter()) {
        match (s, i) {
            (Some(s), Some(i)) => {
                let seen = int_of.insert(s.to_string(), i.to_string());
                assert!(seen.is_none_or(|old| old == i.to_string()));
            }
            (None, None) => {}
            _ => panic!("the keys are null on different rows"),
        }
    }
    let mut v: Vec<_> = groups
        .iter()
        .map(|(k, c, t)| (GroupKey(k.as_str().map(|s| int_of[s].clone())), *c, *t))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// `df` sorted by group, each group's rows in their frame order: the layout
/// the bank would build, as a frame. Its `id` column says where each row
/// came from.
fn by_group(df: &DataFrame) -> DataFrame {
    df.sort(
        ["g"],
        SortMultipleOptions::default().with_maintain_order(true),
    )
    .unwrap()
}

/// `df` as several arrow chunks per column: slices of uneven length -- one
/// row, a stretch with no null in it, a long one -- each with its own offset
/// into the original buffer, stacked without rechunking.
fn fragmented(df: &DataFrame) -> DataFrame {
    let n = df.height();
    // Between rows 6 and 22 the fixture has no null `x0` (they fall on
    // i % 17 == 5), so [6, 10) and [10, 11) are null-free pieces and take
    // the memcpy path; the rest take the null-aware one.
    let mut cuts = vec![6, 10, 11, 1000, PAR_MIN_ROWS - 1, PAR_MIN_ROWS + 1, 9000];
    cuts.retain(|&c| c < n);
    let mut parts = Vec::new();
    let mut start = 0;
    for c in cuts {
        parts.push(df.slice(start as i64, c - start));
        start = c;
    }
    parts.push(df.slice(start as i64, n - start));
    let mut acc = parts.remove(0);
    for p in parts {
        acc.vstack_mut(&p).unwrap();
    }
    assert_eq!(acc.height(), n);
    for c in acc.columns() {
        assert!(
            c.as_materialized_series().n_chunks() > 1,
            "{}: not fragmented",
            c.name()
        );
    }
    acc
}

/// One chunk through a fresh bank.
fn run(specs: Vec<Spec>, df: &DataFrame) -> (Bank, DataFrame) {
    let mut bank = Bank::new(specs).unwrap();
    let out = DataFrame::new(df.height(), bank.fit_predict(df).unwrap()).unwrap();
    (bank, out)
}

/// `out` with `id` alongside, put back into frame order.
fn realigned(out: DataFrame, id: &Column) -> DataFrame {
    let mut out = out;
    out.with_column(id.clone()).unwrap();
    out.sort(["id"], SortMultipleOptions::default())
        .unwrap()
        .drop("id")
        .unwrap()
}

fn assert_series(a: &Series, b: &Series, what: &str) {
    if a.equals_missing(b) {
        return;
    }
    assert_eq!(a.len(), b.len(), "{what}: length");
    let eq = a.equal_missing(b).unwrap();
    let i = eq.iter().position(|v| v != Some(true)).unwrap();
    panic!(
        "{what}: differs first at row {i}: {:?} vs {:?}",
        a.get(i).unwrap(),
        b.get(i).unwrap()
    );
}

/// Every field of every output column, bit for bit (NaN equals NaN, null
/// equals null), naming the first field and row that differ.
fn assert_same(a: &DataFrame, b: &DataFrame, what: &str) {
    assert_eq!(
        a.get_column_names(),
        b.get_column_names(),
        "{what}: columns"
    );
    for (ca, cb) in a.columns().iter().zip(b.columns()) {
        let (sa, sb) = (ca.as_materialized_series(), cb.as_materialized_series());
        if let DataType::Struct(_) = sa.dtype() {
            let fa = sa.struct_().unwrap().fields_as_series();
            let fb = sb.struct_().unwrap().fields_as_series();
            assert_eq!(fa.len(), fb.len(), "{what}: {} fields", ca.name());
            for (x, y) in fa.iter().zip(&fb) {
                assert_series(x, y, &format!("{what}: {}.{}", ca.name(), x.name()));
            }
        } else {
            assert_series(sa, sb, &format!("{what}: {}", ca.name()));
        }
    }
}

/// The struct columns without their `coef` fields, for a comparison across
/// chunkings of a spec whose `coef` follows the chunks.
fn without_coef(df: &DataFrame, spec: &str) -> DataFrame {
    let mut out = df.clone();
    let col = out.column(spec).unwrap().as_materialized_series().clone();
    let fields: Vec<Series> = col
        .struct_()
        .unwrap()
        .fields_as_series()
        .into_iter()
        .filter(|f| !f.name().starts_with("coef"))
        .collect();
    let s = StructChunked::from_series(spec.into(), col.len(), fields.iter())
        .unwrap()
        .into_series();
    out.with_column(s.into_column()).unwrap();
    out
}

fn sizes() -> [usize; 5] {
    [
        64,
        PAR_MIN_ROWS - 1,
        PAR_MIN_ROWS,
        PAR_MIN_ROWS + 1,
        3 * PAR_MIN_ROWS + 17,
    ]
}

/// P9: the layout is invisible. A frame with its groups interleaved, the
/// same rows sorted by group (the layout the bank would build, so the bank
/// sees the identity), and each group fed alone all give the same output
/// row for row, `coef` included -- and leave the same state behind, byte
/// for byte. The Int32 key gives the same partition as the String one and
/// so the same everything.
#[test]
fn layout_is_invisible() {
    for n in sizes() {
        let df = interleaved(n, 40, 7);
        let (bank, out) = run(grouped_specs("g"), &df);

        let sorted = by_group(&df);
        let (bank_sorted, out_sorted) = run(grouped_specs("g"), &sorted);
        assert_same(
            &out,
            &realigned(out_sorted, sorted.column("id").unwrap()),
            &format!("n={n}: interleaved vs sorted by group"),
        );
        assert_eq!(bank.groups(), bank_sorted.groups(), "n={n}: groups");
        assert_eq!(
            bank.save_bytes().unwrap(),
            bank_sorted.save_bytes().unwrap(),
            "n={n}: the state after the chunk depends on the layout"
        );

        let (bank_int, out_int) = run(grouped_specs("gi"), &df);
        assert_same(&out, &out_int, &format!("n={n}: String key vs Int32 key"));
        // The keys are spelled differently, so the state bytes cannot be
        // compared; the groups can, once respelled.
        for (si, groups) in bank.groups().iter().enumerate() {
            assert_eq!(
                as_int_keys(groups, &df),
                bank_int.groups()[si],
                "n={n}: spec {si}: groups by Int32 key"
            );
        }

        // Each group alone: the biggest, a single-row one, the null key.
        let biggest = bank.groups()[0]
            .iter()
            .max_by_key(|(_, c, _)| *c)
            .map(|(k, ..)| k.as_str().map(str::to_string))
            .unwrap();
        for key in [biggest.as_deref(), Some("solo1"), None] {
            let mask = is_key(&df, key);
            assert!(mask.sum().unwrap_or(0) > 0, "n={n}: {key:?} present");
            let alone = df.filter(&mask).unwrap();
            let (_, out_alone) = run(grouped_specs("g"), &alone);
            assert_same(
                &out.filter(&mask).unwrap(),
                &out_alone,
                &format!("n={n}: group {key:?} alone"),
            );
        }
    }
}

/// P11: a column arriving as several arrow chunks -- some with nulls, some
/// without, one of a single row, each a slice with an offset -- reads the
/// same as one contiguous chunk, under both layouts and with either key
/// dtype, and leaves the same state.
#[test]
fn columns_read_in_pieces_read_the_same() {
    for n in sizes() {
        let df = interleaved(n, 40, 11);
        for (layout, frame) in [("interleaved", df.clone()), ("sorted", by_group(&df))] {
            let pieces = fragmented(&frame);
            for key in ["g", "gi"] {
                let (bank, out) = run(all_specs(key), &frame);
                let (bank_pieces, out_pieces) = run(all_specs(key), &pieces);
                assert_same(
                    &out,
                    &out_pieces,
                    &format!("n={n} {layout} key={key}: in pieces"),
                );
                assert_eq!(
                    bank.save_bytes().unwrap(),
                    bank_pieces.save_bytes().unwrap(),
                    "n={n} {layout} key={key}: state after the chunk"
                );
            }
        }
    }
}

/// P11's per-chunk reader casts each column to Float64 first: an integer,
/// Float32, Boolean or Null-typed feature, target or weight, in pieces and
/// interleaved, gives what the same values as a contiguous Float64 column
/// give.
#[test]
fn other_dtypes_read_like_float64() {
    let n = PAR_MIN_ROWS + 1000;
    let df = interleaved(n, 40, 5);
    let specs = || {
        let mut s = all_specs("g");
        for sp in &mut s {
            sp.features = vec!["x0".into(), "x1".into(), "xi".into(), "xb".into()];
        }
        s
    };
    let as_f64 = |name: &str| {
        df.column(name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .with_name(name.into())
    };
    // The baseline: every column already Float64, one chunk.
    let mut base = df.clone();
    for name in ["xi", "xb"] {
        base.with_column(as_f64(name)).unwrap();
    }
    let (bank, out) = run(specs(), &base);

    let mut cases: Vec<(String, DataFrame)> = Vec::new();
    cases.push(("Int32 xi, Boolean xb".into(), df.clone()));
    for dtype in [
        DataType::Int8,
        DataType::UInt8,
        DataType::Int64,
        DataType::Float32,
    ] {
        let mut d = df.clone();
        d.with_column(df.column("xi").unwrap().cast(&dtype).unwrap())
            .unwrap();
        cases.push((format!("{dtype} xi"), d));
    }
    for (what, frame) in cases {
        for (how, frame) in [
            ("contiguous", frame.clone()),
            ("in pieces", fragmented(&frame)),
        ] {
            let (bank_case, out_case) = run(specs(), &frame);
            assert_same(&out, &out_case, &format!("{what}, {how}"));
            assert_eq!(
                bank.save_bytes().unwrap(),
                bank_case.save_bytes().unwrap(),
                "{what}, {how}: state"
            );
        }
    }

    // A Null-typed column is an all-null column: as a target every row is
    // predict-only, as a weight every row is skipped. Same as the Float64
    // all-null column either way.
    for name in ["y", "w"] {
        let all_null = |dtype: DataType| {
            let mut d = df.clone();
            d.with_column(Series::full_null(name.into(), n, &dtype).into_column())
                .unwrap();
            d
        };
        let (_, want) = run(specs(), &all_null(DataType::Float64));
        for (how, frame) in [
            ("contiguous", all_null(DataType::Null)),
            ("in pieces", fragmented(&all_null(DataType::Null))),
        ] {
            let (_, got) = run(specs(), &frame);
            assert_same(&want, &got, &format!("Null-typed {name}, {how}"));
        }
    }
}

/// `df` fed in chunks of `len` rows to a fresh bank, the outputs stacked.
fn feed(df: &DataFrame, len: usize) -> DataFrame {
    let mut bank = Bank::new(all_specs("g")).unwrap();
    let n = df.height();
    let mut acc: Option<DataFrame> = None;
    let mut i = 0;
    while i < n {
        let chunk = df.slice(i as i64, len.min(n - i));
        let out = DataFrame::new(chunk.height(), bank.fit_predict(&chunk).unwrap()).unwrap();
        match &mut acc {
            None => acc = Some(out),
            Some(a) => {
                a.vstack_mut(&out).unwrap();
            }
        }
        i += chunk.height();
    }
    acc.unwrap()
}

/// Chunk invariance across the threshold: chunks of `PAR_MIN_ROWS - 1`,
/// `PAR_MIN_ROWS` and `PAR_MIN_ROWS + 1` rows -- the last two read and
/// assembled in parallel, the first on one thread -- give what one chunk
/// and what small chunks give, for every field of the specs whose `coef`
/// cadence is per row, and for everything but `coef` on the one whose
/// cadence is per chunk.
#[test]
fn the_threshold_is_not_a_seam() {
    let n = 3 * PAR_MIN_ROWS + 17;
    let df = interleaved(n, 40, 13);
    let one = feed(&df, n);
    for len in [PAR_MIN_ROWS - 1, PAR_MIN_ROWS, PAR_MIN_ROWS + 1, 1000, 61] {
        let chunked = feed(&df, len);
        assert_same(
            &without_coef(&one, "plain"),
            &without_coef(&chunked, "plain"),
            &format!("chunks of {len}"),
        );
    }
}

/// `df` with one value of a Float64 column replaced.
fn with_value(df: &DataFrame, name: &str, row: usize, v: Option<f64>) -> DataFrame {
    let mut vals: Vec<Option<f64>> = df.column(name).unwrap().f64().unwrap().iter().collect();
    vals[row] = v;
    let mut d = df.clone();
    d.with_column(Column::new(name.into(), vals)).unwrap();
    d
}

/// The frame rows of `row`'s group, in order.
fn rows_of_the_group(df: &DataFrame, row: usize) -> Vec<usize> {
    let keys = df.column("g").unwrap().str().unwrap();
    let key = keys.get(row);
    (0..df.height()).filter(|&i| keys.get(i) == key).collect()
}

/// Under a permuted layout the columns are read in group order, so a bad
/// value's position in what is checked is not its row in the frame. The
/// error must name the frame's row, and always the same one, whichever
/// thread found it first: a null clock, a negative weight, and a backwards
/// clock under `on_clock_reset = "error"`, whether the earlier row of the
/// group is in the same chunk or in the bank's state. And when a clock and a
/// weight are both bad, the clock is what is reported, every time -- the
/// column checks run in parallel, the errors surface in a fixed order.
#[test]
fn errors_name_the_frame_row_under_any_layout() {
    let n = PAR_MIN_ROWS + 500;
    let df = interleaved(n, 40, 17);
    // A row deep in the frame, not in the first-seen group, with an earlier
    // row of its own group in the frame and the same session as that row
    // (a session change is allowed to take the clock backwards).
    let keys = df.column("g").unwrap().str().unwrap();
    let sess = df.column("sess").unwrap().str().unwrap();
    let previous_in_group = |frame: &DataFrame, i: usize| -> Option<usize> {
        let group = rows_of_the_group(frame, i);
        let at = group.iter().position(|&r| r == i).unwrap();
        (at > 0).then(|| group[at - 1])
    };
    let row = (PAR_MIN_ROWS + 321..n)
        .find(|&i| {
            keys.get(i).is_some()
                && keys.get(i) != keys.get(0)
                && previous_in_group(&df, i).is_some_and(|p| sess.get(p) == sess.get(i))
        })
        .unwrap();
    let strict = || {
        let mut specs = grouped_specs("g");
        for s in &mut specs {
            s.on_clock_reset = online_core::OnClockReset::Error;
        }
        specs
    };
    let names = |err: PolarsError, what: &str| {
        let msg = err.to_string();
        assert!(
            msg.contains(&format!("row {row}")) && msg.contains(what),
            "{what}: {msg}"
        );
        msg
    };

    let mut bank = Bank::new(strict()).unwrap();
    let before = bank.save_bytes().unwrap();
    names(
        bank.fit_predict(&with_value(&df, "t", row, None))
            .unwrap_err(),
        "clock",
    );
    names(
        bank.fit_predict(&with_value(&df, "w", row, Some(-0.5)))
            .unwrap_err(),
        "negative",
    );
    // Backwards within the chunk: below the group's previous row.
    let prev = previous_in_group(&df, row).unwrap();
    let t_prev = df.column("t").unwrap().f64().unwrap().get(prev).unwrap();
    let backwards = with_value(&df, "t", row, Some(t_prev - 1.0));
    names(bank.fit_predict(&backwards).unwrap_err(), "goes backwards");
    assert_eq!(
        bank.save_bytes().unwrap(),
        before,
        "a refused chunk changed the bank"
    );
    assert!(
        bank.groups().iter().all(Vec::is_empty),
        "a refused chunk left groups behind"
    );

    // Both bad at once, the weight earlier in the frame than the clock and
    // in another group: the clock wins, and keeps winning.
    let other = (0..row)
        .find(|&i| keys.get(i).is_some() && keys.get(i) != keys.get(row))
        .unwrap();
    let both = with_value(&with_value(&df, "t", row, None), "w", other, Some(-1.0));
    for _ in 0..20 {
        let msg = bank.fit_predict(&both).unwrap_err().to_string();
        assert!(msg.contains("clock") && !msg.contains("negative"), "{msg}");
    }

    // Backwards against the state: the first chunk trains, the second's
    // `row2` is the first of its group there, in the session the bank
    // remembers for the group, and sits below the clock it remembers.
    // Refused, naming the second chunk's row; the bank is as it was, and
    // the corrected chunk then feeds as into a clean bank.
    let first = df.clone();
    let second = interleaved(n, 40, 18);
    let keys2 = second.column("g").unwrap().str().unwrap();
    let row2 = (PAR_MIN_ROWS..n)
        .find(|&i| {
            keys2.get(i).is_some()
                && rows_of_the_group(&second, i)[0] == i
                && keys.iter().any(|k| k == keys2.get(i))
        })
        .unwrap();
    let last_in_first = *rows_of_the_group(
        &df,
        (0..n).find(|&i| keys.get(i) == keys2.get(row2)).unwrap(),
    )
    .last()
    .unwrap();
    let mut sess2: Vec<Option<String>> = second
        .column("sess")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|v| v.map(str::to_string))
        .collect();
    sess2[row2] = sess.get(last_in_first).map(str::to_string);
    // The second frame's clocks start from zero again, so every group's
    // first row there is already behind the state; shift them past it and
    // put `row2` alone back below.
    let shift = df.column("t").unwrap().f64().unwrap().max().unwrap() + 1.0;
    let mut t2: Vec<Option<f64>> = second
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(|v| v.map(|v| v + shift))
        .collect();
    let mut good = second.clone();
    good.with_column(Column::new("sess".into(), sess2)).unwrap();
    good.with_column(Column::new("t".into(), t2.clone()))
        .unwrap();
    t2[row2] = Some(0.5);
    let mut bad = good.clone();
    bad.with_column(Column::new("t".into(), t2)).unwrap();

    let mut bank = Bank::new(strict()).unwrap();
    bank.fit_predict(&first).unwrap();
    let trained = bank.save_bytes().unwrap();
    let msg = bank.fit_predict(&bad).unwrap_err().to_string();
    assert!(
        msg.contains("goes backwards") && msg.contains(&format!("row {row2}")),
        "{msg}"
    );
    assert_eq!(
        bank.save_bytes().unwrap(),
        trained,
        "a refused chunk changed the bank"
    );
    let out = DataFrame::new(n, bank.fit_predict(&good).unwrap()).unwrap();
    let mut clean = Bank::new(strict()).unwrap();
    clean.fit_predict(&first).unwrap();
    let want = DataFrame::new(n, clean.fit_predict(&good).unwrap()).unwrap();
    assert_same(&want, &out, "after a refused chunk");
}

/// `predict` on a frame that interleaves groups the bank knows with groups
/// it has never seen, on both sides of the threshold: the known groups'
/// rows score as they would with the unknown rows filtered out, or with
/// the frame sorted by group; the unknown rows are null in every field.
/// Without a target or weight column the predictions and `n_eff` are the
/// same, and the bank is not touched by any of it.
#[test]
fn predict_skips_unseen_groups_in_any_layout() {
    let train = interleaved(PAR_MIN_ROWS + 1000, 30, 21);
    let mut bank = Bank::new(all_specs("g")).unwrap();
    bank.fit_predict(&train).unwrap();
    let trained = bank.save_bytes().unwrap();
    let seen: std::collections::HashSet<Option<String>> = bank.groups()[0]
        .iter()
        .map(|(k, ..)| k.as_str().map(str::to_string))
        .collect();

    for n in [300, PAR_MIN_ROWS + 700] {
        // 45 groups where the bank saw 30: the labels of the 15 more are new.
        let score = interleaved(n, 45, 22);
        let keys = score.column("g").unwrap().str().unwrap();
        let known: BooleanChunked = keys
            .iter()
            .map(|k| Some(seen.contains(&k.map(str::to_string))))
            .collect();
        let n_known = known.sum().unwrap() as usize;
        assert!(0 < n_known && n_known < n, "n={n}: both kinds of group");

        let out = DataFrame::new(n, bank.predict(&score).unwrap()).unwrap();
        let filtered = score.filter(&known).unwrap();
        let out_filtered = DataFrame::new(n_known, bank.predict(&filtered).unwrap()).unwrap();
        assert_same(
            &out.filter(&known).unwrap(),
            &out_filtered,
            &format!("n={n}: known rows vs the frame without the unknown"),
        );
        let sorted = by_group(&score);
        let out_sorted = DataFrame::new(n, bank.predict(&sorted).unwrap()).unwrap();
        assert_same(
            &out,
            &realigned(out_sorted, sorted.column("id").unwrap()),
            &format!("n={n}: interleaved vs sorted by group"),
        );
        let unknown = !&known;
        for col in out.columns() {
            let s = col.as_materialized_series();
            if s.name() == "solo" {
                continue; // ungrouped: every row is its one stream's
            }
            for f in s.struct_().unwrap().fields_as_series() {
                let f = f.filter(&unknown).unwrap();
                assert_eq!(
                    f.null_count(),
                    f.len(),
                    "n={n}: {}.{} on an unknown group",
                    s.name(),
                    f.name()
                );
            }
        }

        // No target, no weight: `pred` and `n_eff` as before.
        let bare = score.drop("y").unwrap().drop("w").unwrap();
        let out_bare = DataFrame::new(n, bank.predict(&bare).unwrap()).unwrap();
        for (a, b) in out.columns().iter().zip(out_bare.columns()) {
            let (sa, sb) = (a.as_materialized_series(), b.as_materialized_series());
            let fa = sa.struct_().unwrap().fields_as_series();
            let fb = sb.struct_().unwrap().fields_as_series();
            for (x, y) in fa.iter().zip(&fb) {
                if x.name().starts_with("pred_") || x.name() == "n_eff" {
                    assert_series(
                        x,
                        y,
                        &format!("n={n}: {}.{} without y and w", sa.name(), x.name()),
                    );
                }
            }
        }
        assert_eq!(
            bank.save_bytes().unwrap(),
            trained,
            "n={n}: predict touched the bank"
        );
    }
}

/// The bank's own view of the fixture's groups: the Int32 key and the
/// String key name the same partition, with the null key apart.
#[test]
fn the_fixture_s_two_keys_partition_alike() {
    let df = interleaved(PAR_MIN_ROWS, 40, 3);
    let gap = r#""session_gap": 10.0,"#;
    let (bank_s, _) = run(vec![spec("m", Some("g"), "60.0", gap)], &df);
    let (bank_i, _) = run(vec![spec("m", Some("gi"), "60.0", gap)], &df);
    assert_eq!(as_int_keys(&bank_s.groups()[0], &df), bank_i.groups()[0]);
    assert!(
        bank_s.groups()[0]
            .iter()
            .any(|(k, ..)| *k == GroupKey(None))
    );
    assert!(
        bank_i.groups()[0]
            .iter()
            .any(|(k, ..)| *k == GroupKey(None))
    );
    assert!(
        bank_i.groups()[0]
            .iter()
            .any(|(k, ..)| k.as_str() == Some("-2"))
    );
}
