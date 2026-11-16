//! A window's far-edge row weighs nothing in a mean (docs/PLAN.md task 430).
//!
//! A row exactly one window back from a row is in that row's window under
//! `closed = "left"` or `"both"` looking back, `"right"` or `"both"` looking
//! ahead, and its held interval ends at it: no mass inside the window, so a
//! mean or a variance whose only valued row it is, is null. Membership was
//! decided from the two rows' raw clocks and the mass from the policy time
//! (seconds from the stretch's first row), where the same distance rounds a
//! last bit either way, so the row weighed 1e-17 or nothing depending on
//! where the stretch began. The edge here is exact integer nanoseconds, so no
//! policy-time arithmetic is shared with the operators; `tests/
//! test_window_far_edge.py` sweeps the same inputs through the Python call
//! against an oracle for the general case.

use online_polars::{WindowsConfig, WindowsRun};
use polars::prelude::*;
use serde_json::{Value, json};

const MS: i64 = 1_000_000;

/// Every output of a run over `frame`, the rows closed at the end included.
fn run(config: Value, frame: &DataFrame) -> DataFrame {
    let config: WindowsConfig = serde_json::from_value(config).unwrap();
    let mut run = WindowsRun::new(config, frame.schema()).unwrap();
    let mut out = run.feed(frame, None).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    out
}

fn formula(name: &str, op: &str, h: &Value, w: &Value, closed: &str, bias: bool) -> Value {
    let mut o = json!({"half_life": h, "window_size": w, "closed": closed, "partial": "keep"});
    if op == "ewm_var" {
        o["bias"] = json!(bias);
    }
    json!({"name": name, "tree": [op, ["col", "x"], o]})
}

fn nulls(out: &DataFrame, name: &str) -> Vec<bool> {
    out.column(name)
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(|v| v.is_none())
        .collect()
}

/// A datetime frame of `(stamp in ms, value)` rows.
fn frame(rows: &[(i64, Option<f64>)]) -> DataFrame {
    let ts: Vec<i64> = rows.iter().map(|r| r.0 * MS).collect();
    let x: Vec<Option<f64>> = rows.iter().map(|r| r.1).collect();
    DataFrame::new_infer_height(vec![
        Column::new("ts".into(), ts)
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap(),
        Column::new("x".into(), x),
    ])
    .unwrap()
}

#[test]
fn the_row_one_window_back_weighs_nothing_whenever_the_stretch_began() {
    // The window and the edge row's stamp, in ms; the stretch began at each
    // start, so the policy time of every row after it rounds differently.
    for (w, a) in [
        (200, 100),
        (300, 400),
        (700, 1500),
        (1100, 400),
        (1100, 2999),
    ] {
        for s in [0, 3, 7, 50, 99] {
            for h in [json!("inf"), json!("700ms")] {
                let w_text = json!(format!("{w}ms"));
                for closed in ["both", "left"] {
                    // Row 1 holds the only value; row 2 is exactly a window
                    // after it; row 3 is past it.
                    let rows = [(s, None), (a, Some(5.0)), (a + w, None), (a + w + 1, None)];
                    let out = run(
                        json!({
                            "formulas": [
                                formula("m", "ewm_mean", &h, &w_text, closed, false),
                                formula("v", "ewm_var", &h, &w_text, closed, true),
                            ],
                            "clock": "ts", "gap_cap": "1h",
                        }),
                        &frame(&rows),
                    );
                    let at = format!("{closed} h={h} window {w} edge {a} start {s}");
                    // Under "both" the row at the stamp is in its own window
                    // and holds the whole window's mass; under "left" it is
                    // not in it.
                    let own = closed == "left";
                    assert_eq!(nulls(&out, "m"), [true, own, true, true], "mean {at}");
                    assert_eq!(nulls(&out, "v"), [true, own, true, true], "var {at}");
                }
                for closed in ["both", "right"] {
                    // A forward window: the valued row is exactly a window
                    // after the row before it, and a later row closes it.
                    let rows = [(s, None), (a, None), (a + w, Some(5.0)), (a + w + 1, None)];
                    let out = run(
                        json!({
                            "formulas": [formula("m", "rewm_mean", &h, &w_text, closed, false)],
                            "clock": "ts", "gap_cap": "1h",
                        }),
                        &frame(&rows),
                    );
                    let at = format!("ahead {closed} h={h} window {w} edge {a} start {s}");
                    assert_eq!(nulls(&out, "m"), [true; 4], "{at}");
                }
            }
        }
    }
}

#[test]
fn the_reviews_input_is_null_at_both_starts() {
    for s in [0, 3] {
        let rows = [(s, None), (400, Some(5.0)), (1500, None), (1501, None)];
        let out = run(
            json!({
                "formulas": [formula("m", "ewm_mean", &json!("1s"), &json!("1100ms"), "both", false)],
                "clock": "ts", "gap_cap": "1h",
            }),
            &frame(&rows),
        );
        let m: Vec<Option<f64>> = out.column("m").unwrap().f64().unwrap().iter().collect();
        assert_eq!(m, [None, Some(5.0), None, None], "start {s}");
    }
}
