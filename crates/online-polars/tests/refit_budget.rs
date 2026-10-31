//! A spec's `refit_budget` (docs/PLAN.md task 277): each group the bank
//! first sees reserves the most its Huber refit buffers can hold, in the
//! order its rows first arrive, until the next reservation would pass the
//! budget; that group and every later new one give their refits up, are
//! counted in `summary()`'s `refit_given_up`, and raise one notice. The
//! reservations are state, and depend on that order alone, whatever the
//! chunking (hard rule 3).

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// Groups `0..groups`, interleaved, `rows` rows each, one feature, two
/// targets: the rows of group `g` first arrive at row `g`.
fn frame(groups: usize, rows: usize) -> DataFrame {
    let mut s = 3u64;
    let mut r = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let n = groups * rows;
    let (mut g, mut x, mut y, mut z) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let a = r();
        g.push((i % groups) as i64);
        x.push(a);
        y.push(1000.0 + 2.0 * a + 0.5 * r() + if a > 0.0 && r() > 0.5 { 50.0 } else { 0.0 });
        z.push(-a + 0.5 * r());
    }
    df!("g" => g, "x" => x, "y" => y, "z" => z).unwrap()
}

fn spec(budget: &str, rows: u32) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {{"type": "huber", "huber_refit_rows": {rows}, "refit_budget": {budget}}},
            "targets": ["y", "z"], "features": ["x"], "group": "g", "half_life": "inf", "coef_every": 0}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn feed(bank: &mut Bank, df: &DataFrame, chunk: usize) -> DataFrame {
    let mut out: Option<DataFrame> = None;
    let mut at = 0;
    while at < df.height() {
        let len = chunk.min(df.height() - at);
        let piece = df.slice(at as i64, len);
        let got = DataFrame::new(len, bank.fit_predict(&piece).unwrap()).unwrap();
        match out.as_mut() {
            None => out = Some(got),
            Some(o) => {
                o.vstack_mut(&got).unwrap();
            }
        }
        at += len;
    }
    out.unwrap()
}

/// `refit_given_up` per group, in key order.
fn given_up(bank: &Bank) -> Vec<i64> {
    let s = bank.summary(0, None).unwrap();
    let keys: Vec<i64> = s
        .column("group")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|k| k.and_then(|k| k.parse().ok()).unwrap_or(0))
        .collect();
    let counts: Vec<i64> = s
        .column("refit_given_up")
        .unwrap()
        .i64()
        .unwrap()
        .iter()
        .map(Option::unwrap)
        .collect();
    let mut pairs: Vec<(i64, i64)> = keys.into_iter().zip(counts).collect();
    pairs.sort();
    pairs.into_iter().map(|p| p.1).collect()
}

/// A budget of 1 MiB at 50 rows a target: each group of two targets
/// reserves `2 · 8 · 50 · (1 + 4 + 16)` = 16,800 bytes, so 62 groups fit
/// and the 38 after them give their refits up, both targets each, counted
/// and noticed once. The groups that refit have refit by the end.
#[test]
fn the_groups_past_the_budget_give_their_refits_up() {
    let df = frame(100, 60);
    let mut bank = Bank::new(vec![spec("1.0", 50)]).unwrap();
    feed(&mut bank, &df, df.height());
    let each = 2 * online_polars::online_core::refit_reservation(1, 50);
    assert_eq!(each, 16_800);
    let fit = (1024 * 1024) / each;
    assert_eq!(fit, 62);
    let counts = given_up(&bank);
    assert_eq!(counts[..fit], vec![0; fit][..]);
    assert_eq!(counts[fit..], vec![2; 100 - fit][..]);
    let notices = bank.take_notices();
    let refit: Vec<&String> = notices
        .iter()
        .filter(|n| n.contains("refit_budget"))
        .collect();
    assert_eq!(refit.len(), 1, "{notices:?}");
    assert!(refit[0].contains("at group \"62\""), "{}", refit[0]);
    // No bound: every group refits.
    let mut free = Bank::new(vec![spec("\"inf\"", 50)]).unwrap();
    feed(&mut free, &df, df.height());
    assert_eq!(given_up(&free), vec![0; 100]);
    assert!(
        free.take_notices()
            .iter()
            .all(|n| !n.contains("refit_budget"))
    );
}

/// Which groups give up depends on the order their rows first arrive, not
/// on where the chunks end: one chunk, chunks of 7 and 600 rows, and a
/// save and a load in between give the same output and the same counts.
#[test]
fn the_budget_is_the_same_at_any_chunking_and_across_a_save() {
    let df = frame(100, 60);
    let mut one = Bank::new(vec![spec("1.0", 50)]).unwrap();
    let whole = feed(&mut one, &df, df.height());
    for chunk in [7, 600] {
        let mut b = Bank::new(vec![spec("1.0", 50)]).unwrap();
        assert!(
            feed(&mut b, &df, chunk).equals_missing(&whole),
            "chunks of {chunk}"
        );
        assert_eq!(given_up(&b), given_up(&one));
    }
    // Saved after the first 30 groups have arrived: the rest reserve against
    // what the file kept.
    let mut a = Bank::new(vec![spec("1.0", 50)]).unwrap();
    let head = feed(&mut a, &df.slice(0, 30), 30);
    let bytes = a.save_bytes().unwrap();
    let mut b = Bank::load_bytes(&bytes, Some(&[spec("1.0", 50)])).unwrap();
    let tail = feed(&mut b, &df.slice(30, df.height() - 30), 1000);
    let mut both = head;
    both.vstack_mut(&tail).unwrap();
    assert!(both.equals_missing(&whole));
    assert_eq!(given_up(&b), given_up(&one));
}

#[test]
fn a_budget_of_no_mib_is_refused() {
    let text = r#"{"name": "m", "model": {"type": "huber", "refit_budget": 0.0},
        "targets": ["y"], "features": ["x"], "half_life": "inf"}"#;
    let s: Spec = serde_json::from_str(text).unwrap();
    let e = Bank::new(vec![s]).err().unwrap();
    assert!(e.contains("refit_budget must be > 0 MiB"), "{e}");
    let text = r#"{"name": "m", "model": {"type": "sgd", "refit_budget": 8.0},
        "targets": ["y"], "features": ["x"], "half_life": "inf"}"#;
    let s: Spec = serde_json::from_str(text).unwrap();
    let e = Bank::new(vec![s]).err().unwrap();
    assert!(e.contains("refit_budget"), "{e}");
}

/// A group given its refits up keeps them given up through a drift reset:
/// the model the reset rebuilds holds no buffer the bank did not reserve.
/// One group, a budget smaller than its reservation, and a break that
/// fires the detector; against no bound, where the rebuilt model refits.
#[test]
fn a_reset_keeps_a_group_given_up() {
    let mut s = 7u64;
    let mut r = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let n = 1600;
    let (mut t, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![]);
    let mut clock = 0.0;
    for i in 0..n {
        clock += 1.0 + 0.5 * r();
        let (a, b) = (r(), r());
        let (sign, level) = if i < n / 2 { (1.0, 0.3) } else { (-1.0, 6.3) };
        t.push(clock);
        x0.push(a);
        x1.push(b);
        y.push(level + sign * (2.0 * a - 1.5 * b) + 0.1 * r());
    }
    let df = df!("t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap();
    let make = |budget: &str| -> Spec {
        let text = format!(
            r#"{{"name": "m", "model": {{"type": "huber", "huber_refit_rows": 100, "refit_budget": {budget}}},
                "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0,
                "half_life": 200.0, "emit_drift": true, "drift_threshold": 20.0,
                "drift_action": "reset"}}"#
        );
        serde_json::from_str(&text).unwrap()
    };
    let mut tight = Bank::new(vec![make("0.001")]).unwrap();
    let out = feed(&mut tight, &df, df.height());
    let drift = out
        .column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("drift_y")
        .unwrap();
    let fired = (0..n)
        .find(|&i| drift.bool().unwrap().get(i) == Some(true))
        .expect("the detector fires");
    assert!(fired >= n / 2 && fired + 200 < n, "{fired}");
    assert_eq!(
        given_up(&tight),
        vec![1],
        "the rebuilt model gave its refit up too"
    );
    let mut free = Bank::new(vec![make("\"inf\"")]).unwrap();
    feed(&mut free, &df, df.height());
    assert_eq!(given_up(&free), vec![0]);
}

fn sgd_spec(budget: &str, loss: &str) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {{"type": "sgd", "loss": "{loss}", "huber_refit_rows": 50, "refit_budget": {budget}}},
            "targets": ["y", "z"], "features": ["x"], "group": "g", "half_life": "inf", "coef_every": 0}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// `sgd(loss = "huber")` reserves and gives up as `huber` does: the same
/// 16,800 bytes a group, so the same 62 groups refit. Another loss has no
/// refit and so no budget.
#[test]
fn sgds_huber_loss_holds_the_same_budget() {
    let df = frame(100, 60);
    let mut bank = Bank::new(vec![sgd_spec("1.0", "huber")]).unwrap();
    feed(&mut bank, &df, df.height());
    let counts = given_up(&bank);
    assert_eq!(counts[..62], vec![0; 62][..]);
    assert_eq!(counts[62..], vec![2; 38][..]);
    assert_eq!(sgd_spec("1.0", "huber").model.refit_budget_mib(), Some(1.0));
    assert_eq!(sgd_spec("\"inf\"", "huber").model.refit_budget_mib(), None);
    let squared: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "sgd"}, "targets": ["y"], "features": ["x"], "half_life": "inf"}"#,
    )
    .unwrap();
    assert_eq!(squared.model.refit_budget_mib(), None);
    assert_eq!(spec("2.5", 50).model.refit_budget_mib(), Some(2.5));
}

/// A bank file whose reservation is not a count of bytes, or names a spec
/// the caller did not give, is refused as damaged.
#[test]
fn a_damaged_reservation_is_refused() {
    let df = frame(10, 20);
    let mut bank = Bank::new(vec![spec("1.0", 50)]).unwrap();
    feed(&mut bank, &df, df.height());
    let bytes = bank.save_bytes().unwrap();
    let file = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let with = |f: &dyn Fn(&mut Vec<rmpv::Value>)| -> Vec<u8> {
        let mut v = file.clone();
        let rmpv::Value::Map(map) = &mut v else {
            panic!("a bank file is a map")
        };
        let (_, reserved) = map
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some("refit_reserved"))
            .expect("the file keeps the reservation");
        let rmpv::Value::Array(entries) = reserved else {
            panic!("the reservations are an array")
        };
        let rmpv::Value::Array(entry) = &mut entries[0] else {
            panic!("a reservation is a pair")
        };
        f(entry);
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        out
    };
    let specs = [spec("1.0", 50)];
    assert!(Bank::load_bytes(&with(&|_| {}), Some(&specs)).is_ok());
    for (what, bad) in [
        ("negative", with(&|e| e[1] = rmpv::Value::F64(-1.0))),
        ("NaN", with(&|e| e[1] = rmpv::Value::F64(f64::NAN))),
        (
            "infinite",
            with(&|e| e[1] = rmpv::Value::F64(f64::INFINITY)),
        ),
        ("another spec", with(&|e| e[0] = rmpv::Value::from(3))),
    ] {
        let e = Bank::load_bytes(&bad, Some(&specs))
            .err()
            .unwrap_or_else(|| panic!("{what}"));
        assert!(e.contains("refit reservation"), "{what}: {e}");
    }
}

/// A session restart under `session_gap = "reset"` rebuilds each group's
/// stream, and the group keeps what the budget gave it when the bank first
/// saw it: the two groups admitted refit again in the new session, and the
/// two past the budget stay given up. Each group reserves
/// `2 · 8 · 20 · 21` = 6,720 bytes; the budget holds two and a half.
#[test]
fn a_session_restart_keeps_each_group_as_the_budget_left_it() {
    let base = frame(4, 60);
    let n = base.height();
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let s: Vec<&str> = (0..n).map(|i| if i < n / 2 { "a" } else { "b" }).collect();
    let df = base
        .hstack(&[Column::new("t".into(), t), Column::new("s".into(), s)])
        .unwrap();
    let each = 2 * online_polars::online_core::refit_reservation(1, 20);
    assert_eq!(each, 6_720);
    let budget = 2.5 * each as f64 / (1024.0 * 1024.0);
    let text = format!(
        r#"{{"name": "m", "model": {{"type": "huber", "huber_refit_rows": 20, "refit_budget": {budget}}},
            "targets": ["y", "z"], "features": ["x"], "group": "g", "half_life": "inf",
            "coef_every": 0, "clock": "t", "gap_cap": 10.0, "session": "s",
            "session_gap": "reset"}}"#
    );
    let spec: Spec = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
    let mut bank = Bank::new(vec![spec]).unwrap();
    feed(&mut bank, &df, n);
    let resets = bank.summary(0, None).unwrap();
    let resets: i64 = resets
        .column("resets")
        .unwrap()
        .i64()
        .unwrap()
        .sum()
        .unwrap();
    assert_eq!(resets, 4, "each group restarted once");
    assert_eq!(given_up(&bank), vec![0, 0, 2, 2]);
}

/// Task 278: a refit that takes every one of its 200 IRLS passes is counted
/// in the bank's `solve_failures`, for `huber` and `sgd(loss="huber")`
/// alike. The stream is `online_core`'s `slow_refit_stream(687)`, written
/// out again here: 71 rows, a level added to 39% of the rows with `x > 0`,
/// whose IRLS needs 880 passes.
#[test]
fn a_refit_that_runs_to_its_cap_is_a_solve_failure() {
    let mut state: u64 = 688u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 50 + (rnd() * 450.0) as usize;
    let frac = rnd() * 0.45;
    let level = (rnd() * 6.0 - 1.0).exp() * 3.0;
    let (mut x, mut y) = (vec![], vec![]);
    for _ in 0..n {
        let a = rnd() * 2.0 - 1.0;
        let e = (-2.0 * rnd().max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * rnd()).cos();
        let out = if rnd() < frac && a > 0.0 { level } else { 0.0 };
        x.push(a);
        y.push(1.0 + 2.0 * a + 0.5 * e + out);
    }
    assert_eq!(n, 71);
    let df = df!("x" => x, "y" => y).unwrap();
    for (model, extra) in [
        ("huber", ""),
        ("sgd", r#", "loss": "huber", "learning_rate": 0.05"#),
    ] {
        for (rows, want) in [(n, 1), (0, 0)] {
            let text = format!(
                r#"{{"name": "m", "model": {{"type": "{model}", "huber_refit_rows": {rows}{extra}}},
                    "targets": ["y"], "features": ["x"], "half_life": "inf", "min_weight": 3.0}}"#
            );
            let spec: Spec = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
            let mut bank = Bank::new(vec![spec]).unwrap();
            feed(&mut bank, &df, 13);
            let got: u64 = bank.solve_failures()[0].iter().map(|(_, c)| c).sum();
            assert_eq!(got, want, "{model} with huber_refit_rows = {rows}");
        }
    }
}

/// A chunk refused after its new groups reserved their buffers takes the
/// once-a-bank notice back with them: the groups are forgotten, so the
/// notice naming one, and the flag that it was raised, go too (review round
/// 10, S7-5). Here the chunk's groups 62 to 99 cross the budget at 62, and
/// a row of group 0 that steps its clock back refuses the chunk; nothing is
/// noticed, and the chunk fed again without that row names group 62.
#[test]
fn a_refused_chunk_takes_its_refit_notice_back() {
    let df = frame(100, 1);
    let t: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let df = df.hstack(&[Series::new("t".into(), t).into()]).unwrap();
    let text = r#"{"name": "m", "model": {"type": "huber", "huber_refit_rows": 50, "refit_budget": 1.0},
        "targets": ["y", "z"], "features": ["x"], "group": "g", "clock": "t", "gap_cap": 1000.0,
        "half_life": "inf", "coef_every": 0}"#;
    let s: Spec = serde_json::from_str(text).unwrap();
    let mut bank = Bank::new(vec![s]).unwrap();
    bank.fit_predict(&df.slice(0, 62)).unwrap();
    assert!(
        bank.take_notices()
            .iter()
            .all(|n| !n.contains("refit_budget"))
    );
    let back =
        df!("g" => [0i64], "x" => [0.1], "y" => [1000.0], "z" => [0.0], "t" => [-1.0]).unwrap();
    let mut refused = df.slice(62, 38);
    refused.vstack_mut(&back).unwrap();
    let e = bank
        .fit_predict(&refused)
        .expect_err("the step back is refused");
    assert!(e.to_string().contains("clock"), "{e}");
    let notices = bank.take_notices();
    assert!(
        notices.iter().all(|n| !n.contains("refit_budget")),
        "a notice of a forgotten group: {notices:?}"
    );
    bank.fit_predict(&df.slice(62, 38)).unwrap();
    let notices = bank.take_notices();
    let refit: Vec<&String> = notices
        .iter()
        .filter(|n| n.contains("refit_budget"))
        .collect();
    assert_eq!(refit.len(), 1, "{notices:?}");
    assert!(refit[0].contains("at group \"62\""), "{}", refit[0]);
}
