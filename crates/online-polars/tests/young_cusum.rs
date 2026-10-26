//! A windowed CUSUM is near its size from its first rows (docs/PLAN.md task
//! 278, D-B5): its null's share of the plain variance is read as far as its
//! sums have filled, not as the filled share from the first row.
//!
//! Beside an `ewridge` at a half-life of 50 with the breaks at a memory of
//! 200, the filled CUSUM's share is `50 / 250 = 0.2` and the CUSUM of
//! squares' 1/2; a sum a few rows old is all but its plain variance. Divided
//! by the filled shares from the first row, the 150 rows after the
//! diagnostics' readiness on 200 clean streams passed 1.96 on 29.0% (CUSUM)
//! and 14.5% (CUSUM of squares) of rows; read as far as the sums have
//! filled, on 5.4% and 5.6% (measured 2026-10-25). The Python twin
//! (`tests/test_online_diagnostics.py`) holds the same at the default
//! memory; this one is where `cargo mutants` sees the plumbing.

use online_polars::{Bank, Spec};
use polars::prelude::*;

const GROUPS: usize = 200;
const ROWS: usize = 420;
const YOUNG: usize = 150;

/// A seeded generator in `(0, 1)`.
fn uniform(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// `GROUPS` streams of `y = 0.5 + x0 − x1 + e`, every draw Gaussian (Box and
/// Muller's transform).
fn frame() -> DataFrame {
    let mut u = uniform(2785);
    let mut gauss = move || (-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos();
    let n = GROUPS * ROWS;
    let (mut g, mut x0, mut x1, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for k in 0..GROUPS {
        for _ in 0..ROWS {
            let (a, b) = (gauss(), gauss());
            g.push(k as i64);
            x0.push(a);
            x1.push(b);
            y.push(0.5 + a - b + gauss());
        }
    }
    df!("g" => g, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

/// The share of each group's first `YOUNG` values of `field` past 1.96, and
/// how many values that read.
fn young_size(out: &DataFrame, field: &str) -> (f64, usize) {
    let s = out.column("m").unwrap().struct_().unwrap();
    let col = s.field_by_name(field).unwrap();
    let v: Vec<f64> = col
        .f64()
        .unwrap()
        .iter()
        .map(|x| x.unwrap_or(f64::NAN))
        .collect();
    let (mut past, mut read) = (0usize, 0usize);
    for k in 0..GROUPS {
        let rows = &v[k * ROWS..(k + 1) * ROWS];
        let Some(first) = rows.iter().position(|x| x.is_finite()) else {
            continue;
        };
        for x in rows[first..].iter().take(YOUNG).filter(|x| x.is_finite()) {
            read += 1;
            past += usize::from(x.abs() > 1.96);
        }
    }
    (past as f64 / read.max(1) as f64, read)
}

#[test]
fn a_young_windowed_cusum_is_near_its_size() {
    let spec: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x0", "x1"], "half_life": 50.0, "group": "g",
            "min_weight": 10.0, "emit_breaks": true, "breaks_half_life": 200.0}"#,
    )
    .unwrap();
    let df = frame();
    let mut bank = Bank::new(vec![spec]).unwrap();
    let out = DataFrame::new(df.height(), bank.fit_predict(&df).unwrap()).unwrap();
    let sizes: Vec<(&str, f64, usize)> = ["cusum_y", "cusum_sq_y"]
        .into_iter()
        .map(|field| {
            let (size, read) = young_size(&out, field);
            (field, size, read)
        })
        .collect();
    for (field, size, read) in sizes {
        // Every group reads its young rows: the readiness is near row 230.
        assert!(read > GROUPS * (YOUNG - 10), "{field}: {read} values");
        // 5.4% and 5.6% measured; a normal variable passes 1.96 on 5%. The
        // CUSUM's floor holds the correction from overshooting into a
        // conservative test: the plain sum's share, 1, read 2.3% (and 4.8%
        // for the CUSUM of squares, whose young share is near 1 anyway).
        let floor = if field == "cusum_y" { 0.04 } else { 0.03 };
        assert!((floor..0.08).contains(&size), "{field}: {size}");
    }
}

/// A gradient fit absorbs a level at its learning rate, not as a mean at a
/// memory, so its CUSUM keeps the plain sum at every age: `Z1 / sqrt(S2)`
/// over the studentized residuals before the row, unit weights decayed at
/// the breaks' memory, from the first one on -- no share, young or filled.
/// Its CUSUM of squares does take its scale's share, so it is left out.
#[test]
fn a_gradient_fits_cusum_keeps_the_plain_sum() {
    let spec: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "sgd"}, "targets": ["y"],
            "features": ["x0", "x1"], "half_life": 50.0, "min_weight": 10.0,
            "emit_breaks": true}"#,
    )
    .unwrap();
    let df = frame().head(Some(ROWS));
    let mut bank = Bank::new(vec![spec]).unwrap();
    let out = DataFrame::new(df.height(), bank.fit_predict(&df).unwrap()).unwrap();
    let s = out.column("m").unwrap().struct_().unwrap();
    let read = |name: &str| -> Vec<f64> {
        let col = s.field_by_name(name).unwrap();
        col.f64()
            .unwrap()
            .iter()
            .map(|x| x.unwrap_or(f64::NAN))
            .collect()
    };
    let (z, cusum) = (read("studentized_y"), read("cusum_y"));
    let lam = 0.5f64.powf(1.0 / 50.0);
    let first = z
        .iter()
        .position(|v| v.is_finite())
        .expect("a studentized row");
    let (mut z1, mut s2, mut checked) = (0.0, 0.0, 0);
    for t in first..ROWS {
        if t > first {
            let want = z1 / f64::sqrt(s2);
            assert!(
                (cusum[t] - want).abs() <= 1e-10 * want.abs().max(1.0),
                "row {t}: {} vs {want}",
                cusum[t]
            );
            checked += 1;
        }
        (z1, s2) = (z1 * lam + z[t], s2 * lam * lam + 1.0);
    }
    assert!(checked > 100, "{checked} rows checked");
}
