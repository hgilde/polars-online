//! Cross-platform state test (docs/PLAN.md §9 class 7).
//!
//! Writing a state file on one OS and loading it on another must work. CI does
//! the real hand-off by uploading the artifact this test writes on macOS and
//! reading it back on Windows (see .github/workflows/release.yml). Locally, and
//! whenever the artifact is absent, the test still checks the round trip and
//! that the format has no host-dependent parts.

use std::path::PathBuf;

use online_polars::{Bank, Spec};
use polars::prelude::*;

fn spec() -> Spec {
    toml::from_str(
        r#"
name = "m"
targets = ["y"]
features = ["x0", "x1"]
clock = "t"
halflife = 50.0
max_dclock = 10.0
group = "g"
min_periods = 5.0

[model]
type = "ew_ridge"
ridge = 1e-6
max_rows_between_solves = 1
"#,
    )
    .unwrap()
}

fn frame(n: usize) -> DataFrame {
    let mut s = 7u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut g, mut t, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        g.push(if i % 2 == 0 { "a" } else { "b" }.to_string());
        t.push(i as f64);
        let a = lcg();
        let b = lcg();
        x0.push(a);
        x1.push(b);
        y.push(2.0 * a - b);
    }
    df!("g" => g, "t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

fn preds(df: &DataFrame) -> Vec<Option<f64>> {
    let s = df
        .column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("pred_y")
        .unwrap();
    s.f64().unwrap().iter().collect()
}

/// A state written now must load now and continue identically. This is the
/// same code path the cross-OS test exercises; only the writer differs.
#[test]
fn state_round_trips_through_bytes() {
    let df = frame(400);
    let mut a = Bank::new(vec![spec()]).unwrap();
    a.fit_predict(&df.slice(0, 200)).unwrap();
    let bytes = a.save_bytes().unwrap();

    let mut b = Bank::load_bytes(&bytes, Some(&[spec()])).unwrap();
    let second = df.slice(200, 200);
    let out_a = DataFrame::new(second.height(), a.fit_predict(&second).unwrap()).unwrap();
    let out_b = DataFrame::new(second.height(), b.fit_predict(&second).unwrap()).unwrap();
    assert_eq!(preds(&out_a), preds(&out_b));
}

/// msgpack from this crate must be byte-identical for identical input, which is
/// what makes the artifact hand-off between OSes meaningful.
#[test]
fn state_bytes_are_deterministic() {
    let df = frame(200);
    let mut a = Bank::new(vec![spec()]).unwrap();
    let mut b = Bank::new(vec![spec()]).unwrap();
    a.fit_predict(&df).unwrap();
    b.fit_predict(&df).unwrap();
    assert_eq!(a.save_bytes().unwrap(), b.save_bytes().unwrap());
}

/// When CI hands over a state file written on the other OS
/// (`ONLINE_FOREIGN_STATE`), load it and continue the stream; the predictions
/// must match a run that never left this machine.
#[test]
fn loads_a_state_written_on_another_os() {
    let Some(path) = std::env::var_os("ONLINE_FOREIGN_STATE") else {
        eprintln!("ONLINE_FOREIGN_STATE not set; skipping the cross-OS hand-off");
        return;
    };
    let path = PathBuf::from(path);
    let df = frame(400);

    let mut local = Bank::new(vec![spec()]).unwrap();
    local.fit_predict(&df.slice(0, 200)).unwrap();
    let second = df.slice(200, 200);
    let expected = DataFrame::new(second.height(), local.fit_predict(&second).unwrap()).unwrap();

    let mut foreign = Bank::load(&path, Some(&[spec()]))
        .unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    let got = DataFrame::new(second.height(), foreign.fit_predict(&second).unwrap()).unwrap();
    assert_eq!(
        preds(&expected),
        preds(&got),
        "a state written on another OS did not continue identically"
    );
}

/// Writes the state file CI hands to the other OS. Run with
/// `ONLINE_WRITE_STATE=<path> cargo test -p online-polars --test state_portability`.
#[test]
fn writes_the_handoff_state_when_asked() {
    let Some(path) = std::env::var_os("ONLINE_WRITE_STATE") else {
        return;
    };
    let df = frame(400);
    let mut bank = Bank::new(vec![spec()]).unwrap();
    bank.fit_predict(&df.slice(0, 200)).unwrap();
    bank.save(&PathBuf::from(path)).unwrap();
}
