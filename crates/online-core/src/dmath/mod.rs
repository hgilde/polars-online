//! The library's exponentials and logarithms, behind one choice of implementation (task 426 (k),
//! a measurement branch: nothing here merges). Every model calls `x.dexp()`, `x.dln()` and the rest
//! of [`DMath`]; which function answers is a cargo feature:
//!
//! - none: the platform's libm through `std` (today's bits);
//! - `dmath-libm`: the `libm` crate (a port of FreeBSD's msun, pure Rust, the same bits on every target);
//! - `dmath-pxfm`: the `pxfm` crate (CORE-MATH's correctly rounded functions, pure Rust);
//! - `dmath-own`: this crate's branch-free kernels from `+ - * /` alone ([`own`]); `powf` is `libm`'s.
//!
//! At most one feature may be on.
#[cfg(feature = "dmath-own")]
mod coefs;
#[cfg(feature = "dmath-own")]
pub mod own;

#[cfg(any(
    all(feature = "dmath-libm", feature = "dmath-pxfm"),
    all(feature = "dmath-libm", feature = "dmath-own"),
    all(feature = "dmath-pxfm", feature = "dmath-own")
))]
compile_error!("at most one of dmath-libm, dmath-pxfm, dmath-own may be enabled");

/// The functions the models call, spelled so a method call on an `f64` reads as it did.
pub trait DMath: Sized {
    fn dexp(self) -> f64;
    fn dexp2(self) -> f64;
    fn dexp_m1(self) -> f64;
    fn dln(self) -> f64;
    fn dln_1p(self) -> f64;
    fn dpowf(self, y: f64) -> f64;
}

#[cfg(not(any(feature = "dmath-libm", feature = "dmath-pxfm", feature = "dmath-own")))]
impl DMath for f64 {
    #[inline(always)]
    fn dexp(self) -> f64 {
        self.exp()
    }
    #[inline(always)]
    fn dexp2(self) -> f64 {
        self.exp2()
    }
    #[inline(always)]
    fn dexp_m1(self) -> f64 {
        self.exp_m1()
    }
    #[inline(always)]
    fn dln(self) -> f64 {
        self.ln()
    }
    #[inline(always)]
    fn dln_1p(self) -> f64 {
        self.ln_1p()
    }
    #[inline(always)]
    fn dpowf(self, y: f64) -> f64 {
        self.powf(y)
    }
}

#[cfg(feature = "dmath-libm")]
impl DMath for f64 {
    #[inline(always)]
    fn dexp(self) -> f64 {
        libm::exp(self)
    }
    #[inline(always)]
    fn dexp2(self) -> f64 {
        libm::exp2(self)
    }
    #[inline(always)]
    fn dexp_m1(self) -> f64 {
        libm::expm1(self)
    }
    #[inline(always)]
    fn dln(self) -> f64 {
        libm::log(self)
    }
    #[inline(always)]
    fn dln_1p(self) -> f64 {
        libm::log1p(self)
    }
    #[inline(always)]
    fn dpowf(self, y: f64) -> f64 {
        libm::pow(self, y)
    }
}

#[cfg(feature = "dmath-pxfm")]
impl DMath for f64 {
    #[inline(always)]
    fn dexp(self) -> f64 {
        pxfm::f_exp(self)
    }
    #[inline(always)]
    fn dexp2(self) -> f64 {
        pxfm::f_exp2(self)
    }
    #[inline(always)]
    fn dexp_m1(self) -> f64 {
        pxfm::f_expm1(self)
    }
    #[inline(always)]
    fn dln(self) -> f64 {
        pxfm::f_log(self)
    }
    #[inline(always)]
    fn dln_1p(self) -> f64 {
        pxfm::f_log1p(self)
    }
    #[inline(always)]
    fn dpowf(self, y: f64) -> f64 {
        pxfm::f_pow(self, y)
    }
}

#[cfg(feature = "dmath-own")]
impl DMath for f64 {
    #[inline(always)]
    fn dexp(self) -> f64 {
        own::exp(self)
    }
    #[inline(always)]
    fn dexp2(self) -> f64 {
        own::exp2(self)
    }
    #[inline(always)]
    fn dexp_m1(self) -> f64 {
        own::exp_m1(self)
    }
    #[inline(always)]
    fn dln(self) -> f64 {
        own::ln(self)
    }
    #[inline(always)]
    fn dln_1p(self) -> f64 {
        own::ln_1p(self)
    }
    #[inline(always)]
    fn dpowf(self, y: f64) -> f64 {
        libm::pow(self, y)
    }
}
