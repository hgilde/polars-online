//! A later block of the plain loop that begins on a key the core has not made
//! is that key's, not the first row's group's (docs/PLAN.md task 431).
//!
//! The plain loop takes a chunk's rows in blocks of 1,024. With one group made,
//! a block after the first began wherever the last ended, and took its rows as
//! the group of the chunk's first row whatever their key: a chunk of more than
//! 1,025 rows of one key followed by another key not yet seen ran the new key's
//! rows through the first key's windows. The streams here are equal with the
//! loop on and off.
use online_polars::{WindowsConfig, WindowsRun};
use polars::prelude::*;
use serde_json::json;

/// The operator: a running mean, or a forward one over a window.
fn run(plain: bool, forward: bool, keys: &[i64], chunks: &[usize]) -> (Vec<Option<f64>>, u64) {
    let n = keys.len();
    let ts: Vec<i64> = (0..n as i64).map(|i| i * 1_000_000_000).collect();
    let mid: Vec<f64> = (0..n).map(|i| 100.0 + (i % 7) as f64).collect();
    let config: WindowsConfig = serde_json::from_value(json!({
        "formulas": [{"name": "m", "tree": if forward {
            json!(["rewm_mean", ["col", "mid"], {"half_life": "5s", "window_size": "20s"}])
        } else {
            json!(["ewm_mean", ["col", "mid"], {"half_life": "5s"}])
        }}],
        "clock": "ts",
        "gap_cap": "5m",
        "group": "g",
    }))
    .unwrap();
    let frame = |a: usize, z: usize| {
        DataFrame::new_infer_height(vec![
            Column::new("ts".into(), &ts[a..z])
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap(),
            Column::new("mid".into(), &mid[a..z]),
            Column::new("g".into(), &keys[a..z]),
        ])
        .unwrap()
    };
    let mut r = WindowsRun::new(config, frame(0, 1).schema()).unwrap();
    r.set_plain_run(plain);
    let mut out = vec![];
    let mut at = 0;
    for &len in chunks {
        let to = (at + len).min(n);
        let o = r.feed(&frame(at, to), None).unwrap();
        out.extend(o.column("m").unwrap().f64().unwrap().iter());
        at = to;
    }
    let taken = r.plain_rows();
    out.extend(
        r.finish()
            .unwrap()
            .column("m")
            .unwrap()
            .f64()
            .unwrap()
            .iter(),
    );
    (out, taken)
}

#[test]
fn a_later_block_of_another_key_is_that_keys_not_the_first_rows() {
    // The first row of a stream is the general loop's, the first block of the
    // plain loop is the next 1,024 rows, and the second begins at row 1,025:
    // key 0 through row 1,024 and key 1 from row 1,025 on.
    let keys: Vec<i64> = (0..1100).map(|i| i64::from(i >= 1025)).collect();
    for forward in [false, true] {
        let (general, none) = run(false, forward, &keys, &[1100]);
        let (plain, taken) = run(true, forward, &keys, &[1100]);
        assert_eq!(none, 0);
        assert!(taken > 1000, "the plain loop took {taken} rows");
        assert_eq!(plain.len(), general.len());
        for (i, (p, g)) in plain.iter().zip(&general).enumerate() {
            assert_eq!(
                p.map(f64::to_bits),
                g.map(f64::to_bits),
                "forward {forward}, row {i}: {p:?} against {g:?}"
            );
        }
    }
}

#[test]
fn a_group_with_thousands_of_rows_in_one_pass_is_taken_in_segments_to_the_bit() {
    // Two keys in runs of 3,000 rows, then interleaved: a pass over both groups
    // holds a group's rows by the thousand, which are run in segments.
    let keys: Vec<i64> = (0..24000)
        .map(|i| if i < 18000 { (i / 3000) % 2 } else { i % 2 })
        .collect();
    for (forward, chunks) in [false, true].into_iter().flat_map(|f| {
        [vec![24000], vec![5000, 7000, 12000], vec![1000]]
            .into_iter()
            .map(move |c| (f, c))
    }) {
        let (general, none) = run(false, forward, &keys, &chunks);
        let (plain, taken) = run(true, forward, &keys, &chunks);
        assert_eq!(none, 0);
        assert!(taken > 500, "the plain loop took {taken} rows");
        assert_eq!(plain.len(), general.len());
        for (i, (p, g)) in plain.iter().zip(&general).enumerate() {
            assert_eq!(
                p.map(f64::to_bits),
                g.map(f64::to_bits),
                "forward {forward}, chunks {chunks:?}, row {i}: {p:?} against {g:?}"
            );
        }
    }
}
