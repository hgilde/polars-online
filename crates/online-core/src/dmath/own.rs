//! Branch-free exponentials and logarithms from plain IEEE operations only: `+ - * /`, bit moves and
//! integer arithmetic. No `mul_add`, no libm call, no target-feature dispatch, so a scalar call and
//! a lane of a vectorised loop give the same bits on every target (Rust never contracts `a * b + c`).
#![allow(clippy::excessive_precision)]
use super::coefs::{Q_EXP9, Q_M1_12};

const LOG2E: f64 = f64::from_bits(0x3ff71547652b82fe);
const LN2: f64 = f64::from_bits(0x3fe62e42fefa39ef);
const LN2_HI: f64 = f64::from_bits(0x3fe62e42fee00000);
const LN2_LO: f64 = f64::from_bits(0x3dea39ef35793c76);
const SHIFT: f64 = 6755399441055744.0; // 1.5 * 2^52

/// `expm1(r)` for `|r| <= ln2/2` as `r + (r^2/2 + r^3 q(r))`, `q` of degree 9, Estrin's scheme.
#[inline(always)]
fn expm1_small(r: f64) -> f64 {
    let c = &Q_EXP9;
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let a = (c[0] + c[1] * r) + r2 * (c[2] + c[3] * r);
    let b = (c[4] + c[5] * r) + r2 * (c[6] + c[7] * r);
    let d = c[8] + c[9] * r;
    let q = (a + r4 * b) + r8 * d;
    r + (0.5 * r2 + (r2 * r) * q)
}

/// `expm1(r)` for `|r| <= 1`, `q` of degree 12.
#[inline(always)]
fn expm1_unit(r: f64) -> f64 {
    let c = &Q_M1_12;
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let a = (c[0] + c[1] * r) + r2 * (c[2] + c[3] * r);
    let b = (c[4] + c[5] * r) + r2 * (c[6] + c[7] * r);
    let d = (c[8] + c[9] * r) + r2 * (c[10] + c[11] * r);
    let q = (a + r4 * b) + r8 * (d + r4 * c[12]);
    r + (0.5 * r2 + (r2 * r) * q)
}

/// `exp(x)` for `-700 <= x <= 700` (the fast path; callers guard the range).
#[inline(always)]
fn exp_fast(x: f64) -> f64 {
    let kd_s = x * LOG2E + SHIFT;
    let kd = kd_s - SHIFT;
    let hi = x - kd * LN2_HI; // kd * LN2_HI is exact
    let lo = kd * LN2_LO;
    let r = hi - lo;
    let rl = (hi - r) - lo; // the rounding error of r
    let e = expm1_small(r);
    // 1 + e + rl (1 + e)
    let s = (1.0 + e) + rl * (1.0 + e);
    let ki = kd_s.to_bits() << 52;
    f64::from_bits(s.to_bits().wrapping_add(ki))
}

#[inline(always)]
fn exp2_fast(x: f64) -> f64 {
    let kd_s = x + SHIFT;
    let kd = kd_s - SHIFT;
    let f = x - kd; // exact
    let r = f * LN2;
    let e = expm1_small(r);
    let s = 1.0 + e;
    let ki = kd_s.to_bits() << 52;
    f64::from_bits(s.to_bits().wrapping_add(ki))
}

#[inline(always)]
fn pow2(k: i64) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// `exp` outside the fast range: the same reduction, the scale applied in two halves.
#[cold]
#[inline(never)]
fn exp_slow(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.782712893384 {
        return f64::INFINITY;
    }
    if x < -745.1332191019412 {
        return 0.0;
    }
    let kd_s = x * LOG2E + SHIFT;
    let kd = kd_s - SHIFT;
    let hi = x - kd * LN2_HI;
    let lo = kd * LN2_LO;
    let r = hi - lo;
    let rl = (hi - r) - lo;
    let e = expm1_small(r);
    let s = (1.0 + e) + rl * (1.0 + e);
    let k = kd as i64;
    let k1 = k / 2;
    (s * pow2(k1)) * pow2(k - k1)
}

#[inline(always)]
pub fn exp(x: f64) -> f64 {
    if x.abs() <= 700.0 {
        exp_fast(x)
    } else {
        exp_slow(x)
    }
}

#[cold]
#[inline(never)]
fn exp2_slow(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x >= 1024.0 {
        return f64::INFINITY;
    }
    if x < -1075.0 {
        return 0.0;
    }
    let kd = (x + SHIFT) - SHIFT;
    let e = expm1_small((x - kd) * LN2);
    let s = 1.0 + e;
    let k = kd as i64;
    let k1 = k / 2;
    (s * pow2(k1)) * pow2(k - k1)
}

#[inline(always)]
pub fn exp2(x: f64) -> f64 {
    if x.abs() <= 1000.0 {
        exp2_fast(x)
    } else {
        exp2_slow(x)
    }
}

#[inline(always)]
pub fn exp_m1(x: f64) -> f64 {
    if x.abs() <= 1.0 {
        expm1_unit(x)
    } else if x.abs() <= 700.0 {
        exp_fast(x) - 1.0
    } else if x < 0.0 {
        if x.is_nan() { x } else { -1.0 }
    } else {
        exp_slow(x)
    }
}

// ---- logarithm (fdlibm's e_log.c recipe, one formula for every k) ----
const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;

/// `ln(u)` for a positive normal `u`.
#[inline(always)]
fn ln_fast(u: f64) -> f64 {
    let bits = u.to_bits();
    let tmp = bits.wrapping_sub(0x3fe6a09e667f3bcd);
    let k = (tmp as i64) >> 52;
    let iz = bits.wrapping_sub(tmp & (0xfffu64 << 52));
    let z = f64::from_bits(iz);
    let f = z - 1.0;
    let s = f / (2.0 + f);
    let s2 = s * s;
    let w = s2 * s2;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = s2 * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let dk = k as f64;
    dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
}

#[cold]
#[inline(never)]
fn ln_slow(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return x;
    }
    // subnormal: scale by 2^54
    ln_fast(x * 18014398509481984.0) - 54.0 * std::f64::consts::LN_2
}

#[inline(always)]
pub fn ln(x: f64) -> f64 {
    if x >= f64::MIN_POSITIVE && x < f64::INFINITY {
        ln_fast(x)
    } else {
        ln_slow(x)
    }
}

#[inline(always)]
pub fn ln_1p(x: f64) -> f64 {
    if x > -0.99 && x < 4.0e15 {
        let u = 1.0 + x;
        let t = u - 1.0;
        let c = x - t; // the part of x that 1 + x lost (|x| < 1 + x here or exact anyway)
        // for |x| tiny u == 1 and ln_fast gives 0: then c/u = x
        ln_fast(u) + c / u
    } else {
        ln_1p_slow(x)
    }
}

#[cold]
#[inline(never)]
fn ln_1p_slow(x: f64) -> f64 {
    if x.is_nan() || x < -1.0 {
        return f64::NAN;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return x;
    }
    if x >= 4.0e15 {
        return ln(x); // 1 + x == x to the last bit
    }
    // -1 < x <= -0.99: 1 + x is exact (Sterbenz), no correction
    ln_fast(1.0 + x)
}

/// A slice kernel: the fast path on every element, then the rare out-of-range ones redone.
pub fn exp_slice(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x) {
        let c = v.clamp(-700.0, 700.0);
        *o = exp_fast(c);
    }
    for (o, &v) in out.iter_mut().zip(x) {
        if !(v.abs() <= 700.0) {
            *o = exp_slow(v);
        }
    }
}
pub fn exp2_slice(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x) {
        let c = v.clamp(-1000.0, 1000.0);
        *o = exp2_fast(c);
    }
    for (o, &v) in out.iter_mut().zip(x) {
        if !(v.abs() <= 1000.0) {
            *o = exp2_slow(v);
        }
    }
}
pub fn exp_m1_slice(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x) {
        let c = v.clamp(-1.0, 1.0);
        *o = expm1_unit(c);
    }
    for (o, &v) in out.iter_mut().zip(x) {
        if !(v.abs() <= 1.0) {
            *o = exp_m1(v);
        }
    }
}
pub fn ln_slice(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x) {
        let c = if v >= f64::MIN_POSITIVE && v < f64::INFINITY {
            v
        } else {
            1.0
        };
        *o = ln_fast(c);
    }
    for (o, &v) in out.iter_mut().zip(x) {
        if !(v >= f64::MIN_POSITIVE && v < f64::INFINITY) {
            *o = ln_slow(v);
        }
    }
}
