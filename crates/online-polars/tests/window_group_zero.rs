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

/// The clock of a stream: a datetime, an integer column of milliseconds, or
/// none, the row count.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Clock {
    Datetime,
    Int,
    Rows,
}

/// The operator: a running mean, or a forward one over a window.
fn run(plain: bool, forward: bool, keys: &[i64], chunks: &[usize]) -> (Vec<Option<f64>>, u64) {
    run_on(Clock::Datetime, plain, forward, keys, chunks)
}

fn run_on(
    clock: Clock,
    plain: bool,
    forward: bool,
    keys: &[i64],
    chunks: &[usize],
) -> (Vec<Option<f64>>, u64) {
    let n = keys.len();
    let ts: Vec<i64> = (0..n as i64).map(|i| i * 1_000_000_000).collect();
    let mid: Vec<f64> = (0..n).map(|i| 100.0 + (i % 7) as f64).collect();
    let t = |secs: f64| match clock {
        Clock::Datetime => json!(format!("{secs}s")),
        Clock::Int => json!(secs * 1000.0),
        Clock::Rows => json!(secs),
    };
    let mut config = json!({
        "formulas": [{"name": "m", "tree": if forward {
            json!(["rewm_mean", ["col", "mid"], {"half_life": t(5.0), "window_size": t(20.0)}])
        } else {
            json!(["ewm_mean", ["col", "mid"], {"half_life": t(5.0)}])
        }}],
        "group": "g",
    });
    if clock != Clock::Rows {
        config["clock"] = json!("ts");
        config["gap_cap"] = t(300.0);
    }
    let config: WindowsConfig = serde_json::from_value(config).unwrap();
    let frame = |a: usize, z: usize| {
        let mut cols = vec![];
        match clock {
            Clock::Datetime => cols.push(
                Column::new("ts".into(), &ts[a..z])
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap(),
            ),
            Clock::Int => {
                let ms: Vec<i64> = ts[a..z].iter().map(|v| v / 1_000_000).collect();
                cols.push(Column::new("ts".into(), ms));
            }
            Clock::Rows => {}
        }
        cols.push(Column::new("mid".into(), &mid[a..z]));
        cols.push(Column::new("g".into(), &keys[a..z]));
        DataFrame::new_infer_height(cols).unwrap()
    };
    let mut r = WindowsRun::new(config, frame(0, 1).schema()).unwrap();
    r.set_plain_run(plain);
    let mut out = vec![];
    let mut at = 0;
    for &len in chunks.iter().cycle() {
        if at >= n {
            break;
        }
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

#[test]
fn a_later_block_of_another_key_is_that_keys_on_an_integer_and_a_row_count_clock() {
    // The same stream on the other clocks the loop takes: a row-count clock
    // takes it by `count_block`, an integer one by `plain_block`.
    let keys: Vec<i64> = (0..3200)
        .map(|i| i64::from((1025..2100).contains(&i)))
        .collect();
    for clock in [Clock::Int, Clock::Rows] {
        for forward in [false, true] {
            // A forward window under groups needs a clock column.
            if clock == Clock::Rows && forward {
                continue;
            }
            let (general, none) = run_on(clock, false, forward, &keys, &[3200]);
            let (plain, taken) = run_on(clock, true, forward, &keys, &[3200]);
            assert_eq!(none, 0);
            assert!(taken > 2000, "{clock:?}: the plain loop took {taken} rows");
            assert_eq!(plain.len(), general.len());
            for (i, (p, g)) in plain.iter().zip(&general).enumerate() {
                assert_eq!(
                    p.map(f64::to_bits),
                    g.map(f64::to_bits),
                    "{clock:?}, forward {forward}, row {i}: {p:?} against {g:?}"
                );
            }
        }
    }
}

#[test]
fn a_pass_over_groups_on_an_integer_and_a_row_count_clock_is_taken_in_segments_to_the_bit() {
    let keys: Vec<i64> = (0..24000)
        .map(|i| if i < 18000 { (i / 3000) % 2 } else { i % 2 })
        .collect();
    for clock in [Clock::Int, Clock::Rows] {
        for forward in [false, true] {
            // A forward window under groups needs a clock column.
            if clock == Clock::Rows && forward {
                continue;
            }
            for chunks in [vec![24000], vec![5000, 7000, 12000], vec![1000]] {
                let (general, none) = run_on(clock, false, forward, &keys, &chunks);
                let (plain, taken) = run_on(clock, true, forward, &keys, &chunks);
                assert_eq!(none, 0);
                assert!(taken > 500, "{clock:?}: the plain loop took {taken} rows");
                for (i, (p, g)) in plain.iter().zip(&general).enumerate() {
                    assert_eq!(
                        p.map(f64::to_bits),
                        g.map(f64::to_bits),
                        "{clock:?}, forward {forward}, chunks {chunks:?}, row {i}"
                    );
                }
            }
        }
    }
}

/// A stream of `keys` keys in rotation, a row a second: each key's rows are
/// `keys` seconds apart, so past the cap of 300 seconds each one starts its
/// group over.
fn rotation(n: usize, keys: i64) -> Vec<i64> {
    (0..n as i64).map(|i| i % keys).collect()
}

#[test]
fn a_row_that_restarts_its_group_is_taken_by_the_plain_loop_to_the_bit() {
    // 100 keys (every row 100 s after its key's last: no restart, the level),
    // 400 keys (400 s: every row restarts), 10,000 keys (all rows restart),
    // on a datetime and an integer clock, backward and forward, with chunks
    // that cut between a restart row and the row after it.
    for (n, keys) in [(30_000usize, 400i64), (45_000, 10_000), (12_000, 100)] {
        let k = rotation(n, keys);
        for clock in [Clock::Datetime, Clock::Int] {
            for forward in [false, true] {
                for chunks in [vec![n], vec![4097, 1, 999, 20_000], vec![65_536]] {
                    let (general, none) = run_on(clock, false, forward, &k, &chunks);
                    let (plain, taken) = run_on(clock, true, forward, &k, &chunks);
                    assert_eq!(none, 0);
                    if keys >= 400 {
                        assert!(
                            taken as usize > n / 2,
                            "{clock:?} {keys} keys: the plain loop took {taken} of {n} rows"
                        );
                    }
                    assert_eq!(plain.len(), general.len());
                    for (i, (p, g)) in plain.iter().zip(&general).enumerate() {
                        assert_eq!(
                            p.map(f64::to_bits),
                            g.map(f64::to_bits),
                            "{clock:?}, {keys} keys, forward {forward}, chunks {chunks:?}, row {i}: {p:?} against {g:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_gap_past_the_cap_in_one_group_is_taken_by_the_plain_loop_to_the_bit() {
    // One key, a row a second with a gap of 301 s every 50th row: each gap is
    // a restart of the one group (the stream's cap and the group's are the
    // same step).
    let n = 6_000;
    let keys = vec![0i64; n];
    let ts: Vec<i64> = {
        let mut t = 0i64;
        (0..n)
            .map(|i| {
                t += if i % 50 == 49 { 301 } else { 1 };
                t * 1_000_000_000
            })
            .collect()
    };
    for forward in [false, true] {
        for grouped in [false, true] {
            let config: WindowsConfig = serde_json::from_value(json!({
                "formulas": [{"name": "m", "tree": if forward {
                    json!(["rewm_mean", ["col", "mid"], {"half_life": "5s", "window_size": "20s"}])
                } else {
                    json!(["ewm_mean", ["col", "mid"], {"half_life": "5s"}])
                }}],
                "clock": "ts",
                "gap_cap": "5m",
            }))
            .map(|mut c: serde_json::Value| {
                if grouped {
                    c["group"] = json!("g");
                }
                c
            })
            .map(|c| serde_json::from_value(c).unwrap())
            .unwrap();
            let frame = |a: usize, z: usize| {
                let mut cols = vec![
                    Column::new("ts".into(), &ts[a..z])
                        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                        .unwrap(),
                    Column::new(
                        "mid".into(),
                        (a..z).map(|i| 100.0 + (i % 7) as f64).collect::<Vec<f64>>(),
                    ),
                ];
                if grouped {
                    cols.push(Column::new("g".into(), &keys[a..z]));
                }
                DataFrame::new_infer_height(cols).unwrap()
            };
            let go = |plain: bool, chunks: &[usize]| {
                let mut r = WindowsRun::new(config.clone(), frame(0, 1).schema()).unwrap();
                r.set_plain_run(plain);
                let (mut out, mut at) = (vec![], 0);
                for &len in chunks.iter().cycle() {
                    if at >= n {
                        break;
                    }
                    let to = (at + len).min(n);
                    let o = r.feed(&frame(at, to), None).unwrap();
                    out.extend(o.column("m").unwrap().f64().unwrap().iter());
                    at = to;
                }
                let taken = r.plain_rows();
                let tail = r.finish().unwrap();
                out.extend(tail.column("m").unwrap().f64().unwrap().iter());
                (out, taken, r.save_bytes().unwrap())
            };
            for chunks in [vec![n], vec![49, 1, 1, 48, 9999], vec![7]] {
                let (general, none, gs) = go(false, &chunks);
                let (plain, taken, ps) = go(true, &chunks);
                assert_eq!(none, 0);
                assert!(taken as usize > n / 2, "the plain loop took {taken} of {n}");
                assert_eq!(gs, ps, "forward {forward}, grouped {grouped}: the state");
                for (i, (p, g)) in plain.iter().zip(&general).enumerate() {
                    assert_eq!(
                        p.map(f64::to_bits),
                        g.map(f64::to_bits),
                        "forward {forward}, grouped {grouped}, chunks {chunks:?}, row {i}"
                    );
                }
            }
        }
    }
}
