//! Writes args and every implementation's results to out/<fn>_<set>.bin (f64 LE: args, then one column a candidate).
use std::io::Write;
#[path = "../coefs2.rs"]
mod coefs2;
#[path = "../own.rs"]
mod own;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn mkset(n: usize, seed: u64, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| f(lcg(&mut s))).collect()
}
type F = fn(f64) -> f64;
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    let n = 400_000;
    let ln10 = std::f64::consts::LN_10;
    let sets: Vec<(&str, &str, Vec<f64>)> = vec![
        ("exp2", "neg4", mkset(n, 1, |u| -4.0 * u)),
        ("exp2", "neg1100", mkset(n, 2, |u| -1070.0 * u)),
        ("exp2", "tiny", mkset(n, 3, |u| -(libm::pow(10.0, -18.0 * u)))),
        ("exp2", "pos", mkset(n, 4, |u| 1020.0 * u)),
        ("exp", "neg745", mkset(n, 5, |u| -740.0 * u)),
        ("exp", "neg20", mkset(n, 6, |u| -20.0 * u)),
        ("exp", "pm30", mkset(n, 7, |u| 60.0 * u - 30.0)),
        ("exp", "pm1", mkset(n, 8, |u| 2.0 * u - 1.0)),
        ("exp", "tiny", mkset(n, 9, |u| -(libm::pow(10.0, -18.0 * u)))),
        ("expm1", "neg3", mkset(n, 10, |u| -3.0 * u)),
        ("expm1", "neg40", mkset(n, 11, |u| -40.0 * u)),
        ("expm1", "pm1", mkset(n, 12, |u| 2.0 * u - 1.0)),
        ("expm1", "pos5", mkset(n, 13, |u| 5.0 * u)),
        ("expm1", "tiny", mkset(n, 14, |u| -(libm::pow(10.0, -18.0 * u)))),
        ("expm1", "wide", mkset(n, 15, |u| 1400.0 * u - 700.0)),
        ("ln", "unit", mkset(n, 16, |u| u.max(1e-300))),
        ("ln", "around1", mkset(n, 17, |u| 0.5 + 1.5 * u)),
        ("ln", "wide", mkset(n, 18, |u| libm::exp(ln10 * (600.0 * u - 300.0)))),
        ("ln", "near1", mkset(n, 19, |u| 1.0 + (2.0 * u - 1.0) * 1e-5)),
        ("ln", "logspace", mkset(n, 20, |u| libm::exp(-30.0 * u))),
        ("ln1p", "pos10", mkset(n, 21, |u| 10.0 * u)),
        ("ln1p", "neg", mkset(n, 22, |u| -0.999 * u)),
        ("ln1p", "tiny", mkset(n, 23, |u| libm::pow(10.0, -18.0 * u))),
        ("ln1p", "pos1e6", mkset(n, 24, |u| libm::pow(10.0, 6.0 * u))),
    ];
    for (f, set, a) in &sets {
        let impls: Vec<(&str, F)> = match *f {
            "exp2" => vec![("std", |x| x.exp2()), ("libm", |x| libm::exp2(x)), ("pxfm", |x| pxfm::f_exp2(x)), ("own", |x| own::exp2(x))],
            "exp" => vec![("std", |x| x.exp()), ("libm", |x| libm::exp(x)), ("pxfm", |x| pxfm::f_exp(x)), ("own", |x| own::exp(x))],
            "expm1" => vec![("std", |x| x.exp_m1()), ("libm", |x| libm::expm1(x)), ("pxfm", |x| pxfm::f_expm1(x)), ("own", |x| own::exp_m1(x))],
            "ln" => vec![("std", |x| x.ln()), ("libm", |x| libm::log(x)), ("pxfm", |x| pxfm::f_log(x)), ("own", |x| own::ln(x))],
            "ln1p" => vec![("std", |x| x.ln_1p()), ("libm", |x| libm::log1p(x)), ("pxfm", |x| pxfm::f_log1p(x)), ("own", |x| own::ln_1p(x))],
            _ => unreachable!(),
        };
        let mut w = std::fs::File::create(format!("{dir}/{f}_{set}.txt")).unwrap();
        let names: Vec<&str> = impls.iter().map(|i| i.0).collect();
        writeln!(w, "{}", names.join(",")).unwrap();
        let mut bin = std::fs::File::create(format!("{dir}/{f}_{set}.bin")).unwrap();
        for &x in a { bin.write_all(&x.to_le_bytes()).unwrap(); }
        for (_, g) in &impls {
            for &x in a { bin.write_all(&g(x).to_le_bytes()).unwrap(); }
        }
    }
}
