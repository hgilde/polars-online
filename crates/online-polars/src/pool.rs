//! The bank's thread pool, sized by `POLARS_ONLINE_MAX_THREADS`.
//!
//! Polars has its own pool, sized by `POLARS_MAX_THREADS`, for its readers
//! and writers. The bank keeps a separate one so that the two can be set
//! apart: polars' count also sizes what its readers hold in flight, so a
//! reader can be kept small for memory while the bank still has every core
//! (README, *Parallelism*). Neither pool ever waits on the other -- a bank
//! task never calls back into polars' pool -- so having both busy at once
//! costs nothing measurable, even with each sized to the whole machine.
//!
//! The pool is built at the first call, which is when the variable is read;
//! set after that, it is ignored. Unset or `0` is one thread per core.

use std::ffi::OsStr;
use std::sync::OnceLock;

use polars::prelude::*;
use rayon::ThreadPool;

/// The environment variable that sizes the bank's pool.
pub const THREADS_VAR: &str = "POLARS_ONLINE_MAX_THREADS";

static POOL: OnceLock<ThreadPool> = OnceLock::new();

/// The bank's pool, built on the first call from [`THREADS_VAR`].
///
/// # Errors
///
/// `ComputeError` naming the variable and its value when it is set to
/// anything but a non-negative integer. Nothing is built, so the call can
/// be repeated once the variable is fixed.
pub fn pool() -> PolarsResult<&'static ThreadPool> {
    if let Some(pool) = POOL.get() {
        return Ok(pool);
    }
    let n = threads_from_env(std::env::var_os(THREADS_VAR).as_deref())
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    Ok(POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("polars-online-{i}"))
            .build()
            .expect("could not spawn the bank's threads")
    }))
}

/// How many threads the bank's pool has, building it if nothing has yet.
///
/// # Errors
///
/// As [`pool`].
pub fn thread_pool_size() -> PolarsResult<usize> {
    Ok(pool()?.current_num_threads())
}

/// The variable's value as a thread count. Unset or `0` is one thread per
/// core, spelled out rather than left as rayon's `num_threads(0)`, which
/// would fall back to `RAYON_NUM_THREADS` -- the knob this one replaces.
fn threads_from_env(val: Option<&OsStr>) -> Result<usize, String> {
    let Some(val) = val else {
        return Ok(one_per_core());
    };
    let text = val.to_string_lossy();
    match text.trim().parse::<usize>() {
        Ok(0) => Ok(one_per_core()),
        Ok(n) => Ok(n),
        Err(_) => Err(format!(
            "{THREADS_VAR}={text:?} is not a number of threads; \
             set a non-negative integer (0 is one thread per core) or unset it"
        )),
    }
}

/// What polars defaults to as well (`polars_config::default_max_threads`).
fn one_per_core() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_or_zero_is_one_per_core() {
        let cores = one_per_core();
        assert!(cores >= 1);
        assert_eq!(threads_from_env(None), Ok(cores));
        assert_eq!(threads_from_env(Some(OsStr::new("0"))), Ok(cores));
    }

    #[test]
    fn a_count_is_read_with_its_whitespace() {
        assert_eq!(threads_from_env(Some(OsStr::new("8"))), Ok(8));
        assert_eq!(threads_from_env(Some(OsStr::new(" 3\n"))), Ok(3));
    }

    #[test]
    fn anything_else_names_the_variable_and_the_value() {
        for bad in ["", "eight", "-1", "2.5"] {
            let err = threads_from_env(Some(OsStr::new(bad))).unwrap_err();
            assert!(
                err.starts_with(&format!("{THREADS_VAR}={bad:?} is not")),
                "{err}"
            );
        }
    }

    #[test]
    fn the_pool_is_built_once_and_reports_its_size() {
        let a = pool().unwrap();
        let b = pool().unwrap();
        assert!(std::ptr::eq(a, b));
        assert_eq!(thread_pool_size().unwrap(), a.current_num_threads());
        assert!(a.current_num_threads() >= 1);
    }
}
