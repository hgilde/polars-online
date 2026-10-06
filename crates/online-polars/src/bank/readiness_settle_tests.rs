//! No solve holds its factor past the end of a run, and a stream's `flush`
//! settles one learned outside it. A file of its own, keeping bank.rs under
//! `tests/test_repo_hygiene.py`'s 250 KB cap for a source file.
use crate::stream::AnyModel;
use crate::{Bank, Spec};
use online_core::OnlineModel;
use polars::prelude::*;

/// A solve's readiness shares wait for a read, holding the solve's
/// factor (docs/PLAN.md task 140), and the end of each run takes them
/// and drops it: after a chunk no group's model holds one. One more row
/// learned outside the stream leaves one held, which the stream's
/// `flush` settles.
#[test]
fn no_solve_holds_its_factor_past_the_end_of_a_run() {
    let spec: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "ew_ridge", "max_rows_between_solves": 1},
                "targets": ["y"], "features": ["x0", "x1"], "group": "g",
                "half_life": 20.0, "coef_every": 1000}"#,
    )
    .unwrap();
    let n = 120;
    let df = df!(
        "g" => (0..n).map(|i| ["a", "b", "c"][i % 3]).collect::<Vec<_>>(),
        "x0" => (0..n).map(|i| ((i * 7) % 11) as f64).collect::<Vec<_>>(),
        "x1" => (0..n).map(|i| ((i * 5) % 13) as f64).collect::<Vec<_>>(),
        "y" => (0..n).map(|i| ((i * 3) % 17) as f64).collect::<Vec<_>>()
    )
    .unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    bank.fit_predict(&df.slice(0, 60)).unwrap();
    bank.fit_predict(&df.slice(60, 60)).unwrap();
    assert_eq!(bank.states[0].len(), 3);
    for stream in bank.states[0].values_mut() {
        let AnyModel::EwRidge(m) = &mut stream.models[0].1 else {
            panic!("not an ewridge");
        };
        assert_eq!(m.pending_readiness(), 0, "a chunk's end left a factor held");
        m.step(&[1.0, 2.0], &[Some(3.0)], 1.0, 1.0);
        assert!(
            m.pending_readiness() > 0,
            "the row's solve left nothing to settle"
        );
        stream.models[0].1.flush(1);
        let AnyModel::EwRidge(m) = &stream.models[0].1 else {
            unreachable!();
        };
        assert_eq!(m.pending_readiness(), 0, "flush settles");
    }
}
