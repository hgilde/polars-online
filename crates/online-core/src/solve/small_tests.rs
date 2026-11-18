//! Task 456: the small factor and the small solve are faer's, to the bit.
//!
//! `cholesky_seq` takes an order up to `SMALL` (12) through `small_llt` and
//! `solve_band` an order up to `SMALL_SOLVE` (4) through `small_solve`; both
//! are faer's own arithmetic written out without faer's scratch and its
//! dispatch. Each case here runs faer's call on the same input and holds
//! every output bit equal, across each threshold (4 and 5, 12 and 13), over
//! matrices that factor, ones that nearly do not, and ones faer refuses.
//! `SMALL_TAKEN` says which path a call took, since the two paths' outputs
//! are the same and a bit comparison cannot tell a threshold that moved.

use super::*;

fn draw(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// `Q diag(λ) Qᵀ` of an order `k`, `λ` geometric from 1 to `cond`, exactly
/// symmetric (faer's QR for `Q`): a positive definite matrix of condition
/// number `cond`, and past about `1e16` one that is singular to rounding.
fn spd(k: usize, cond: f64, s: &mut u64) -> Mat<f64> {
    let g = Mat::from_fn(k, k, |_, _| draw(s));
    let q = g.qr().compute_Q();
    let lam = |i: usize| {
        if k == 1 {
            1.0
        } else {
            cond.powf(i as f64 / (k - 1) as f64)
        }
    };
    let mut a = Mat::from_fn(k, k, |i, j| {
        (0..k).map(|r| q[(i, r)] * lam(r) * q[(j, r)]).sum()
    });
    for i in 0..k {
        for j in 0..i {
            let v = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    a
}

/// faer's own sequential Cholesky of the lower triangle of `a`, as
/// `cholesky_seq` calls it above `SMALL`: the lower triangle it gives and
/// its result, `Err` carrying the failing pivot's index.
fn faer_seq(a: &Mat<f64>) -> (Mat<f64>, Result<(), usize>) {
    let n = a.nrows();
    let mut l = a.clone();
    let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq,
        Default::default(),
    ));
    let r = llt::factor::cholesky_in_place(
        l.as_mut(),
        Default::default(),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map(|_| ())
    .map_err(|llt::factor::LltError::NonPositivePivot { index }| index);
    (l, r)
}

fn index(e: llt::factor::LltError) -> usize {
    let llt::factor::LltError::NonPositivePivot { index } = e;
    index
}

/// The calls this thread's `small_llt` and `small_solve` have taken, in
/// total so far.
fn taken() -> (usize, usize) {
    SMALL_TAKEN.with(|t| t.get())
}

/// Every matrix of an order `n` the factor is run on, named: the lower
/// triangle (the strict upper triangle is left to the caller). Ones that
/// factor, from a condition number of 1 to past the point where the
/// matrix is singular to rounding, at scales from `1e-150` to `1e150`; a
/// rank-deficient one, the identity and the zero matrix; and ones faer
/// refuses, a pivot that is zero, negative, subnormal, infinite or NaN,
/// at each position in turn, and an off-diagonal entry that is NaN or
/// infinite.
fn cases(n: usize, s: &mut u64) -> Vec<(String, Mat<f64>)> {
    let mut out: Vec<(String, Mat<f64>)> = Vec::new();
    let mut push = |name: String, a: Mat<f64>| out.push((name, a));
    for cond in [1.0, 1e3, 1e8, 1e12, 1e14, 1e16, 1e18] {
        for rep in 0..4 {
            push(format!("spd cond {cond:e} #{rep}"), spd(n, cond, s));
        }
    }
    for scale in [1e-150, 1e-30, 1e30, 1e150] {
        let a = spd(n, 1e4, s);
        push(
            format!("spd scaled {scale:e}"),
            Mat::from_fn(n, n, |i, j| a[(i, j)] * scale),
        );
    }
    // Rank n-1: a Gram of n-1 random vectors, so that a pivot is cancelled
    // to rounding at the last column (no cancellation at order 1).
    if n > 1 {
        let b = Mat::from_fn(n, n - 1, |_, _| draw(s));
        push("rank n-1".into(), &b * b.transpose());
    }
    push("identity".into(), Mat::identity(n, n));
    push("zero".into(), Mat::zeros(n, n));
    for j in 0..n {
        for (what, v) in [
            ("zero", 0.0),
            ("negative", -1.0),
            ("subnormal", 1e-320),
            ("infinite", f64::INFINITY),
            ("NaN", f64::NAN),
            ("-inf", f64::NEG_INFINITY),
        ] {
            let mut a = spd(n, 1e3, s);
            a[(j, j)] = v;
            push(format!("{what} pivot at {j}"), a);
        }
    }
    for i in 1..n {
        for (what, v) in [("NaN", f64::NAN), ("inf", f64::INFINITY)] {
            let j = (i * 5 + n) % i;
            let mut a = spd(n, 1e3, s);
            a[(i, j)] = v;
            push(format!("{what} at ({i}, {j})"), a);
        }
    }
    out.into_iter()
        .map(|(name, a)| {
            (
                name,
                Mat::from_fn(n, n, |i, j| if i >= j { a[(i, j)] } else { 0.0 }),
            )
        })
        .collect()
}

/// `want` and `got` the same factorization: the same result, the same failing
/// pivot, and, where it factors, every entry of the lower triangle to the
/// bit.
fn assert_same(
    n: usize,
    what: &str,
    (want, want_r): &(Mat<f64>, Result<(), usize>),
    (got, got_r): &(Mat<f64>, Result<(), usize>),
) {
    assert_eq!(got_r, want_r, "n {n}, {what}");
    if want_r.is_ok() {
        for j in 0..n {
            for i in j..n {
                assert_eq!(
                    got[(i, j)].to_bits(),
                    want[(i, j)].to_bits(),
                    "n {n}, {what}, entry ({i}, {j})"
                );
            }
        }
    }
}

/// The small factor is faer's sequential Cholesky to the bit at every order
/// it handles, 1 to `SMALL`, over every case above: the lower triangle of
/// each entry and the pivot it fails at. The strict upper triangle is
/// junk in the odd cases, since the blocked factor hands over diagonal
/// blocks whose upper triangle is not zero and no result may read it.
#[test]
fn the_small_factor_is_faers_to_the_bit_at_every_order() {
    let mut s = 11u64;
    let (mut ok, mut refused) = (0, 0);
    for n in 1..=SMALL {
        for (what, m) in cases(n, &mut s) {
            let mut junk = m.clone();
            for j in 1..n {
                for i in 0..j {
                    junk[(i, j)] = draw(&mut s) * 1e3;
                }
            }
            let want = faer_seq(&m);
            for (kind, input) in [("zero upper", &m), ("junk upper", &junk)] {
                let mut l = input.clone();
                let r = small_llt(l.as_mut()).map_err(index);
                assert_same(n, &format!("{what}, {kind}"), &want, &(l, r));
            }
            if want.1.is_ok() {
                ok += 1;
            } else {
                refused += 1;
            }
        }
    }
    // Both kinds are in the run, so that neither half is vacuous.
    assert!(
        ok >= 300 && refused >= 300,
        "{ok} factored, {refused} refused"
    );
}

/// Past the switch faer's own call is the answer, and the entry that
/// switches is `cholesky_seq`: every order from 1 to `SMALL + 2` (13 is the
/// first past it) through it, with and without a kept scratch, is faer's
/// factorization to the bit.
#[test]
fn the_switch_gives_faers_bits_on_both_sides_of_it() {
    let mut s = 17u64;
    let mut work = CholWork::default();
    for n in 1..=SMALL + 2 {
        for (what, m) in cases(n, &mut s) {
            let want = faer_seq(&m);
            let mut l = m.clone();
            let r = cholesky_seq(l.as_mut(), None).map_err(index);
            assert_same(n, &format!("{what}, no scratch"), &want, &(l, r));
            let mut l = m.clone();
            let r = cholesky_seq(l.as_mut(), Some(&mut work)).map_err(index);
            assert_same(n, &format!("{what}, kept scratch"), &want, &(l, r));
        }
    }
}

/// Which path a call takes: an order up to the machine's limit (`SMALL`, 12,
/// here unless x86-64 has AVX2) takes the small factor and the next takes
/// faer's; an order of 0 takes neither the small factor's
/// loops nor faer's scratch. (A threshold moved by one would give the same
/// bits, so the bits cannot say.)
#[test]
fn the_factor_switches_to_faer_after_the_machines_limit() {
    assert_eq!(SMALL, 12);
    let limit = small_limit();
    assert!(limit == 12 || limit == 8, "limit {limit}");
    let mut s = 5u64;
    for n in 1..=SMALL + 3 {
        let m = Mat::from_fn(n, n, |i, j| {
            if i == j {
                1.0 + n as f64
            } else if i > j {
                0.1 * draw(&mut s)
            } else {
                0.0
            }
        });
        let before = taken();
        cholesky_seq(m.clone().as_mut(), None).expect("diagonally dominant");
        let after = taken();
        let went_small = after.0 - before.0;
        assert_eq!(went_small, usize::from(n <= limit), "order {n}");
        assert_eq!(after.1, before.1, "the solve is not taken by a factor");
    }
}

/// The limit is chosen by the path the loop takes: 8 where it is the fused
/// AVX2 one, 12 otherwise, so the x86 value is covered from any machine.
#[test]
fn the_limit_is_8_for_the_fused_x86_loop_and_12_otherwise() {
    assert_eq!(small_limit_for(true), 8);
    assert_eq!(small_limit_for(false), 12);
    assert_eq!(SMALL_AVX2, 8);
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        small_limit(),
        small_limit_for(crate::sgd::x86_v3().is_some())
    );
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(small_limit(), 12);
}

/// The blocked factor's last block is of any order from 1: a matrix of `64 +
/// t` is a block of 64 and a block of `t`, the second through
/// `cholesky_seq`'s small path up to `SMALL`. Each `t` from 1 to `SMALL + 2`
/// factors (the 13th and 14th, past the switch, take faer's call), the
/// same bits at one thread and four, and the failing pivot is named in the
/// whole matrix's numbering at each position of the last block.
#[test]
fn the_last_block_of_a_blocked_factor_may_be_small() {
    for t in 1..=SMALL + 2 {
        let n = 64 + t;
        let mut s = 3 + n as u64;
        let a = spd(n, 1e3, &mut s);
        let lower = Mat::from_fn(n, n, |i, j| if i >= j { a[(i, j)] } else { 0.0 });
        let mut one = lower.clone();
        cholesky_in_place(one.as_mut(), 1, (LEAF, TILE), None).expect("positive definite");
        let mut four = lower.clone();
        cholesky_in_place(four.as_mut(), 4, (LEAF, TILE), None).expect("positive definite");
        let faer = faer_seq(&lower).0;
        for j in 0..n {
            for i in j..n {
                assert_eq!(
                    one[(i, j)].to_bits(),
                    four[(i, j)].to_bits(),
                    "t {t}, ({i}, {j})"
                );
                assert!(
                    (one[(i, j)] - faer[(i, j)]).abs() <= 1e-12 * (1.0 + faer[(i, j)].abs()),
                    "t {t}, ({i}, {j}): {} vs {}",
                    one[(i, j)],
                    faer[(i, j)]
                );
            }
        }
        for p in 64..n {
            let mut bad = lower.clone();
            bad[(p, p)] = -1.0;
            let got = cholesky_in_place(bad.as_mut(), 1, (LEAF, TILE), None).unwrap_err();
            assert_eq!(index(got), p, "t {t}");
        }
    }
}

/// faer's `llt::solve` of `x` by the factor `l`: the oracle.
fn faer_solve(l: &Mat<f64>, x: &Mat<f64>) -> Mat<f64> {
    let (n, m) = (x.nrows(), x.ncols());
    let mut want = x.clone();
    let mut mem = MemBuffer::new(llt::solve::solve_in_place_scratch::<f64>(n, m, Par::Seq));
    llt::solve::solve_in_place_with_conj(
        l.as_ref(),
        Conj::No,
        want.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
    );
    want
}

/// Two floats the same: the same bits, or both NaN (the payload of a NaN
/// is not a result).
fn same_float(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The lower triangular matrices a solve is run on, of an order `n`: the
/// factors of matrices conditioned from 1 to `1e14`, and ones that are not
/// factors of anything faer would give -- a zero, subnormal, negative,
/// infinite or NaN on the diagonal, an infinite or NaN entry below it --
/// where the solve must still do what faer's does (it divides by the
/// diagonal whatever it holds).
fn lowers(n: usize, s: &mut u64) -> Vec<(String, Mat<f64>)> {
    let mut out = Vec::new();
    for cond in [1.0, 1e3, 1e8, 1e14] {
        let a = spd(n, cond, s);
        let lower = Mat::from_fn(n, n, |i, j| if i >= j { a[(i, j)] } else { 0.0 });
        let l = faer_seq(&lower).0;
        out.push((
            format!("factor of cond {cond:e}"),
            Mat::from_fn(n, n, |i, j| if i >= j { l[(i, j)] } else { 0.0 }),
        ));
    }
    let base = |s: &mut u64| {
        Mat::from_fn(n, n, |i, j| match i.cmp(&j) {
            Ordering::Less => 0.0,
            Ordering::Equal => 1.0 + draw(s).abs(),
            Ordering::Greater => draw(s),
        })
    };
    for j in 0..n {
        for (what, v) in [
            ("zero", 0.0),
            ("subnormal", 1e-320),
            ("negative", -2.0),
            ("infinite", f64::INFINITY),
            ("NaN", f64::NAN),
        ] {
            let mut l = base(s);
            l[(j, j)] = v;
            out.push((format!("{what} diagonal at {j}"), l));
        }
    }
    for i in 1..n {
        for (what, v) in [("NaN", f64::NAN), ("inf", f64::INFINITY)] {
            let mut l = base(s);
            l[(i, 0)] = v;
            out.push((format!("{what} at ({i}, 0)"), l));
        }
    }
    out
}

/// The right-hand sides' columns: the counts a solve meets, including 0, 1, a
/// few, 64 and 65 (past which faer splits the columns in halves) and a
/// count past two tiles, which `solve_in_place` cuts into bands.
const COLUMNS: [usize; 8] = [0, 1, 2, 3, 7, 64, 65, 2 * TILE + 3];

/// `solve_in_place` is faer's `llt::solve` to the bit at every order from 1 to
/// `SMALL_SOLVE + 2`, over every [`lowers`] and every count of columns, the
/// right-hand sides ordinary, large and small, with a zero, an infinity and
/// a NaN among them.
#[test]
fn the_solve_is_faers_to_the_bit_across_the_switch() {
    let mut s = 13u64;
    let (mut checked, mut nonfinite) = (0usize, 0usize);
    for n in 1..=SMALL_SOLVE + 2 {
        for (what, l) in lowers(n, &mut s) {
            for m in COLUMNS {
                let x = Mat::from_fn(n, m, |i, j| match (i + 3 * j) % 11 {
                    0 => 0.0,
                    1 => f64::INFINITY,
                    2 if j % 5 == 0 => f64::NAN,
                    3 => draw(&mut s) * 1e150,
                    4 => draw(&mut s) * 1e-150,
                    _ => draw(&mut s) * 1e3,
                });
                let want = faer_solve(&l, &x);
                let mut got = x.clone();
                solve_in_place(&l, got.as_mut());
                for j in 0..m {
                    for i in 0..n {
                        assert!(
                            same_float(got[(i, j)], want[(i, j)]),
                            "n {n}, {what}, {m} columns, entry ({i}, {j}): {} vs {}",
                            got[(i, j)],
                            want[(i, j)]
                        );
                        checked += 1;
                        nonfinite += usize::from(!want[(i, j)].is_finite());
                    }
                }
            }
        }
    }
    assert!(checked > 20_000, "{checked}");
    assert!(nonfinite > 1_000, "{nonfinite} non-finite results");
}

/// Which path a solve takes: an order up to `SMALL_SOLVE` takes the small
/// solve and 5 takes faer's, for one band and for the several that cut a
/// wide right-hand side (each band is one call).
#[test]
fn the_solve_switches_to_faer_after_order_4() {
    assert_eq!(SMALL_SOLVE, 4);
    let mut s = 9u64;
    for n in 1..=SMALL_SOLVE + 2 {
        let a = spd(n, 10.0, &mut s);
        let lower = Mat::from_fn(n, n, |i, j| if i >= j { a[(i, j)] } else { 0.0 });
        let l = faer_seq(&lower).0;
        for m in [1usize, 2, 64, 65, 2 * TILE - 1] {
            let mut x = Mat::from_fn(n, m, |_, _| draw(&mut s));
            let before = taken();
            solve_in_place(&l, x.as_mut());
            let after = taken();
            assert_eq!(
                after.1 - before.1,
                usize::from(n <= SMALL_SOLVE),
                "order {n}, {m} columns"
            );
            assert_eq!(after.0, before.0, "the factor is not taken by a solve");
        }
    }
}

/// The two paths in the one place a caller meets them, a factor and its
/// solves: `SpdFactor` at orders 1 to 14 (each side of both thresholds) is
/// faer's `Mat::llt` and `Llt::solve` to the bit.
#[test]
fn a_factor_and_its_solves_are_faers_across_both_thresholds() {
    let mut s = 21u64;
    for n in 1..=SMALL + 2 {
        for cond in [1.0, 1e6, 1e12] {
            let a = spd(n, cond, &mut s);
            let flat: Vec<f64> = (0..n * n).map(|ij| a[(ij / n, ij % n)]).collect();
            let f = SpdFactor::of(&flat, n).expect("positive definite");
            assert_eq!(f.jitter(), 0.0, "n {n}, cond {cond:e}");
            let llt = a.llt(faer::Side::Lower).expect("positive definite");
            for j in 0..n {
                for i in j..n {
                    assert_eq!(
                        f.l[(i, j)].to_bits(),
                        llt.L()[(i, j)].to_bits(),
                        "n {n}, cond {cond:e}, ({i}, {j})"
                    );
                }
            }
            let b: Vec<f64> = (0..3 * n).map(|_| draw(&mut s)).collect();
            let want = llt.solve(Mat::from_fn(n, 3, |i, j| b[j * n + i]));
            let got = f.solve(&b, n, 3);
            for j in 0..3 {
                for i in 0..n {
                    assert_eq!(
                        got[j * n + i].to_bits(),
                        want[(i, j)].to_bits(),
                        "n {n}, cond {cond:e}, solve ({i}, {j})"
                    );
                }
            }
        }
    }
}

/// The loop fuses where faer's does. Which operation is fused is the
/// instruction set's: pulp's `mul_add_e_f64s` (the one faer's `f64`
/// kernel calls) is fused in AVX2 and in NEON and two roundings in its
/// scalar fallback, and `Arch::default()` is the one faer picks the set
/// with. The probe is a product whose exact value is a hair off its rounded
/// one, so that a fused sum and an unfused one differ. The loop's choice
/// (`llt_dispatched`) must be the one pulp's `Arch` gives, on this CPU:
/// that is the premise that makes the factor faer's where the instruction
/// set is not the one the other tests ran on.
#[test]
fn the_loop_fuses_exactly_where_pulps_arch_does() {
    // (1 + 2^-30)^2 = 1 + 2^-29 + 2^-60; the rounded product is 1 + 2^-29.
    let e = 2f64.powi(-30);
    let (a, b, c) = (1.0 + e, 1.0 + e, -(1.0 + 2.0 * e));
    let fused = a.mul_add(b, c);
    let unfused = a * b + c;
    assert_ne!(fused, unfused, "the probe separates the two");

    struct Probe(f64, f64, f64);
    impl pulp::WithSimd for Probe {
        type Output = (bool, f64);
        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self(a, b, c) = self;
            let r = simd.mul_add_e_f64s(simd.splat_f64s(a), simd.splat_f64s(b), simd.splat_f64s(c));
            (S::IS_SCALAR, simd.reduce_max_f64s(r))
        }
    }
    let (scalar, got) = pulp::Arch::default().dispatch(Probe(a, b, c));
    assert_eq!(
        got.to_bits(),
        if scalar { unfused } else { fused }.to_bits()
    );

    // The loop on a 2x2 whose second pivot is the probe: l_00 = 1 exactly,
    // l_10 = a, and the pivot is (1 + 4e) - a*a, fused or not.
    let mut m = [[1.0, a], [0.0, 1.0 + 4.0 * e]];
    let mut u = m;
    llt_loop::<2, false>(&mut u).ok();
    let mut f = m;
    llt_loop::<2, true>(&mut f).ok();
    assert_ne!(
        u[1][1].to_bits(),
        f[1][1].to_bits(),
        "the loop's two forms differ"
    );
    llt_dispatched::<2>(&mut m).ok();
    let want = if scalar { u } else { f };
    assert_eq!(m[1][1].to_bits(), want[1][1].to_bits());
}

/// Task 472: both forms of the loop, on any CPU. The dispatcher runs only the
/// one this CPU's instruction set picks, so the other form never ran in tests
/// on a machine of the other kind; each is held here, to the bit, to a plain
/// loop of the test's own: `s - a*b` with the product rounded on its own for
/// the unfused form, `a.mul_add(-b, s)` for the fused one.
fn llt_reference(a: &[Vec<f64>], fused: bool) -> Vec<Vec<f64>> {
    let n = a.len();
    // l[j][i]: column j, entry (i, j), as the loop holds it.
    let mut l = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let mut inv = 0.0;
        for i in j..n {
            let mut s = a[i][j];
            for col in l.iter().take(j) {
                let (x, y) = (col[j], col[i]);
                s = if fused {
                    x.mul_add(-y, s)
                } else {
                    let p = std::hint::black_box(x * y);
                    s - p
                };
            }
            if i == j {
                inv = 1.0 / s.sqrt();
            }
            l[j][i] = s;
        }
        for v in &mut l[j][j..] {
            *v *= inv;
        }
    }
    l
}

fn check_llt_loops<const N: usize>(seed: u64) {
    let mut s = seed;
    let g: Vec<Vec<f64>> = (0..N)
        .map(|_| (0..N).map(|_| draw(&mut s)).collect())
        .collect();
    // G Gᵀ + N I: symmetric positive definite, well away from singular.
    let a: Vec<Vec<f64>> = (0..N)
        .map(|i| {
            (0..N)
                .map(|j| {
                    let dot: f64 = (0..N).map(|k| g[i][k] * g[j][k]).sum();
                    dot + if i == j { N as f64 } else { 0.0 }
                })
                .collect()
        })
        .collect();
    for fused in [false, true] {
        let mut b = [[0.0; N]; N];
        for j in 0..N {
            for i in j..N {
                b[j][i] = a[i][j];
            }
        }
        if fused {
            llt_loop::<N, true>(&mut b).unwrap();
        } else {
            llt_loop::<N, false>(&mut b).unwrap();
        }
        let want = llt_reference(&a, fused);
        for j in 0..N {
            for i in j..N {
                assert_eq!(
                    b[j][i].to_bits(),
                    want[j][i].to_bits(),
                    "order {N}, fused {fused}, entry ({i}, {j})"
                );
            }
        }
    }
}

#[test]
fn each_form_of_the_loop_matches_a_plain_reference_to_the_bit() {
    check_llt_loops::<1>(1);
    check_llt_loops::<2>(2);
    check_llt_loops::<3>(3);
    check_llt_loops::<4>(4);
    check_llt_loops::<5>(5);
    check_llt_loops::<6>(6);
    check_llt_loops::<7>(7);
    check_llt_loops::<8>(8);
    check_llt_loops::<9>(9);
    check_llt_loops::<10>(10);
    check_llt_loops::<11>(11);
    check_llt_loops::<12>(12);
}
