//! The bank's thread pools: the process default, sized by
//! `POLARS_ONLINE_MAX_THREADS`, and a pool per `threads` count asked for.
//!
//! Polars has its own pool, sized by `POLARS_MAX_THREADS`, for its readers
//! and writers. The bank keeps separate ones so that the two can be set
//! apart: polars' count also sizes what its readers hold in flight, so a
//! reader can be kept small for memory while the bank still has every core
//! (README, *Parallelism*). Neither pool ever waits on the other -- a bank
//! task never calls back into polars' pool -- so having both busy at once
//! costs nothing measurable, even with each sized to the whole machine.
//!
//! A bank, a run or a Gram fit given no `threads` runs on the process
//! default, built at the first such call, which is when the variable is
//! read; set after that, it is ignored. Unset or `0` is one thread per core.
//!
//! Given `threads = n`, it runs on a pool of `n` threads from a registry
//! keyed by the size, which holds each pool weakly: every holder of `n`
//! shares one pool, the pool goes away with its last holder, and the next
//! ask for `n` builds it again. When the process default is already built
//! with `n` threads, that pool is the one used, so asking for the size the
//! process already has never starts a second set of threads. The count is
//! not state: no spec, saved state or schema carries it, and the numbers a
//! bank returns do not depend on it.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::sync::{Arc, Mutex, OnceLock, Weak};

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
/// anything but a non-negative integer, or to more threads than the system
/// will start. Nothing is kept, so the call can be repeated once the
/// variable is fixed.
pub fn pool() -> PolarsResult<&'static ThreadPool> {
    if let Some(pool) = POOL.get() {
        return Ok(pool);
    }
    let set = std::env::var_os(THREADS_VAR);
    let n = threads_from_env(set.as_deref()).map_err(|e| polars_err!(ComputeError: "{}", e))?;
    let built = spawned(
        n,
        set.as_deref(),
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("polars-online-{i}"))
            .build(),
    )
    .map_err(|e| polars_err!(ComputeError: "{}", e))?;
    // Two first calls at once each build a pool; the one set first is kept
    // and the other is dropped, its threads with it.
    let _ = POOL.set(built);
    Ok(POOL.get().expect("set above, by this call or another"))
}

/// The pool, or the error naming the variable: a count the system will not
/// start threads for reached an `expect` and panicked, where a bad value of
/// the variable is a `ComputeError` (task 160, PA8). `set` is the variable
/// as read, `None` when unset.
fn spawned<E: std::fmt::Display>(
    n: usize,
    set: Option<&OsStr>,
    built: Result<ThreadPool, E>,
) -> Result<ThreadPool, String> {
    built.map_err(|e| match set {
        Some(v) => format!(
            "{THREADS_VAR}={:?} asks for {n} threads, and the system would not start them \
             ({e}); set a smaller number (0 is one thread per core) or unset it",
            v.to_string_lossy()
        ),
        None => format!(
            "the bank could not start its {n} threads, one per core ({e}); set {THREADS_VAR} \
             to a smaller number"
        ),
    })
}

/// The threads `"auto"` gives one spec's Gram update when `tasks` (spec x
/// group) tasks share a pool of `pool` threads: one unless at least half the
/// pool would be idle (`2 * tasks <= pool`), and then `2 * pool / tasks`,
/// rounded down and at most `pool` (docs/GRAM.md, "One wide Gram on several
/// threads"). A function of the two counts alone, never of timing, and the
/// output does not depend on it: a Gram's bits are the same at every count.
///
/// Measured interleaved on 14 cores (ten fast, four slow) at 800 features,
/// against one thread: a split paid at up to seven groups (1.2x to 1.7x)
/// and cost 2-10% from eight groups to 13 at every count, and the whole pool
/// was 0.89x at 28. Once the fast cores each hold a group, splitting a row's
/// update only adds the cost of handing out its pieces. Below that, twice
/// the even share lets a thread whose group is done take pieces of
/// another's: at four groups the even share, 3, ran 1.24x and 7 ran 1.62x.
#[must_use]
pub fn gram_share(tasks: usize, pool: usize) -> usize {
    if tasks == 0 || 2 * tasks > pool {
        1
    } else {
        (2 * pool / tasks).min(pool)
    }
}

/// How many threads the pool `threads` names has: for `None`, the process
/// default's size, building it if nothing has yet; for `Some(n)`, `n`,
/// checked and building nothing.
///
/// # Errors
///
/// As [`pool`] for `None`, and as [`check_threads`] for `Some(n)`.
pub fn thread_pool_size(threads: Option<usize>) -> PolarsResult<usize> {
    match threads {
        None => Ok(pool()?.current_num_threads()),
        Some(n) => check_threads(n),
    }
}

/// The text that refuses a `threads` argument, `shown` as the caller wrote
/// it. Python says the same of a negative count, a float or a bool, so one
/// message covers every surface.
#[must_use]
pub fn threads_refusal(shown: &str) -> String {
    format!(
        "threads={shown} is not a number of threads: give an integer of 1 or more, or None for \
         the default pool ({THREADS_VAR}, else one thread per core); 0 is not one thread per \
         core here, as it is in the variable"
    )
}

/// `n` as a `threads` count: refused when 0.
///
/// # Errors
///
/// `ComputeError` with [`threads_refusal`]'s text for 0.
pub fn check_threads(n: usize) -> PolarsResult<usize> {
    if n == 0 {
        polars_bail!(ComputeError: "{}", threads_refusal("0"));
    }
    Ok(n)
}

/// The pool a bank, a run or a Gram fit runs on: the process default, or a
/// pool of a given size shared through the registry ([`Threads::new`]).
/// A clone shares the pool. Not state.
#[derive(Clone, Debug, Default)]
pub struct Threads(Held);

#[derive(Clone, Default)]
enum Held {
    /// `threads = None`: [`pool`], built when first entered.
    #[default]
    Process,
    /// `threads = n` where the process default was already built with `n`.
    Default(&'static ThreadPool),
    /// `threads = n` from the registry.
    Sized(Arc<ThreadPool>),
}

impl std::fmt::Debug for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process => f.write_str("Process"),
            Self::Default(p) => write!(f, "Default({})", p.current_num_threads()),
            Self::Sized(p) => write!(f, "Sized({})", p.current_num_threads()),
        }
    }
}

/// The pools built for a `threads` count, by count, held weakly so that a
/// pool lives exactly as long as something holds it.
static SIZED: Mutex<Option<HashMap<usize, Weak<ThreadPool>>>> = Mutex::new(None);

impl Threads {
    /// The pool `threads` names. `None` is the process default, as it was
    /// before this argument existed, read from [`THREADS_VAR`] when first
    /// entered. `Some(n)` is a pool of `n` threads, shared with every other
    /// holder of `n` and built now if nothing holds one; when the process
    /// default is already built with `n` threads, it is that pool.
    ///
    /// # Errors
    ///
    /// `ComputeError` naming the argument for `Some(0)` ([`check_threads`])
    /// or for a count the system will not start threads for.
    pub fn new(threads: Option<usize>) -> PolarsResult<Self> {
        let Some(n) = threads else {
            return Ok(Self(Held::Process));
        };
        check_threads(n)?;
        if let Some(p) = POOL.get()
            && p.current_num_threads() == n
        {
            return Ok(Self(Held::Default(p)));
        }
        // The lock is held across the build, so two asks for one size at
        // once build one pool. A poisoned lock only means another build
        // panicked; the map is still whole.
        let mut guard = SIZED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some(p) = map.get(&n).and_then(Weak::upgrade) {
            return Ok(Self(Held::Sized(p)));
        }
        // A size whose pool has gone is rebuilt; the other dead entries are
        // swept, so the map holds the live sizes and this one.
        map.retain(|_, w| w.strong_count() > 0);
        let built = spawned_for_argument(
            n,
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(move |i| format!("polars-online-{n}t-{i}"))
                .build(),
        )
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;
        let built = Arc::new(built);
        map.insert(n, Arc::downgrade(&built));
        Ok(Self(Held::Sized(built)))
    }

    /// The pool, the process default built here if this is its first use.
    ///
    /// # Errors
    ///
    /// As [`pool`], for the process default.
    pub fn pool(&self) -> PolarsResult<&ThreadPool> {
        match &self.0 {
            Held::Process => pool(),
            Held::Default(p) => Ok(p),
            Held::Sized(p) => Ok(p),
        }
    }

    /// The count asked for: `None` for the process default.
    #[must_use]
    pub fn asked(&self) -> Option<usize> {
        match &self.0 {
            Held::Process => None,
            Held::Default(p) => Some(p.current_num_threads()),
            Held::Sized(p) => Some(p.current_num_threads()),
        }
    }

    /// `f` on the pool, as [`ThreadPool::install`].
    ///
    /// # Errors
    ///
    /// As [`Self::pool`], or `f`'s own.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> PolarsResult<R> + Send) -> PolarsResult<R> {
        self.pool()?.install(f)
    }

    /// Whether the registry holds a live pool of `n` threads: how a test
    /// sees a pool go with its last holder.
    #[doc(hidden)]
    #[must_use]
    pub fn registry_holds(n: usize) -> bool {
        SIZED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|m| m.get(&n))
            .is_some_and(|w| w.strong_count() > 0)
    }
}

/// The sized pool, or the error naming the argument for a count the system
/// will not start threads for, as [`spawned`] names the variable.
fn spawned_for_argument<E: std::fmt::Display>(
    n: usize,
    built: Result<ThreadPool, E>,
) -> Result<ThreadPool, String> {
    built.map_err(|e| {
        format!(
            "threads={n} asks for {n} threads, and the system would not start them ({e}); give a \
             smaller number, or None for the default pool"
        )
    })
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

    /// A count the system will not start threads for -- `100000` met
    /// `EAGAIN` -- is the documented error naming the variable, never a
    /// panic. Tested on the mapping, with the build's error made up: no
    /// test spawns a hundred thousand threads.
    #[test]
    fn a_pool_the_system_will_not_start_names_the_variable() {
        let refused = || -> Result<ThreadPool, std::io::Error> {
            Err(std::io::Error::from(std::io::ErrorKind::WouldBlock))
        };
        let err = spawned(100_000, Some(OsStr::new("100000")), refused()).unwrap_err();
        assert!(
            err.starts_with(&format!("{THREADS_VAR}=\"100000\" asks for 100000 threads")),
            "{err}"
        );
        assert!(err.contains("set a smaller number"), "{err}");
        let err = spawned(14, None, refused()).unwrap_err();
        assert!(
            err.contains(THREADS_VAR) && err.contains("14 threads"),
            "{err}"
        );
        // And a pool that started is the pool.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build();
        assert_eq!(spawned(1, None, pool).unwrap().current_num_threads(), 1);
    }

    #[test]
    fn the_pool_is_built_once_and_reports_its_size() {
        let a = pool().unwrap();
        let b = pool().unwrap();
        assert!(std::ptr::eq(a, b));
        assert_eq!(thread_pool_size(None).unwrap(), a.current_num_threads());
        assert!(a.current_num_threads() >= 1);
    }

    /// A size the process default does not have, so that `Threads::new`
    /// comes from the registry whichever test built the default first.
    /// Each test takes its own `k`: the registry is shared by the binary.
    fn sized(k: usize) -> usize {
        let default = threads_from_env(std::env::var_os(THREADS_VAR).as_deref()).unwrap_or(0);
        if k == default { k + 64 } else { k }
    }

    #[test]
    fn zero_threads_is_refused_by_name_and_says_none_is_one_per_core() {
        let err = Threads::new(Some(0)).unwrap_err().to_string();
        assert!(
            err.contains("threads=0 is not a number of threads"),
            "{err}"
        );
        assert!(err.contains("or None for the default pool"), "{err}");
        assert!(err.contains("0 is not one thread per core here"), "{err}");
        let err = thread_pool_size(Some(0)).unwrap_err().to_string();
        assert!(err.contains(&threads_refusal("0")), "{err}");
    }

    #[test]
    fn two_holders_of_one_size_share_one_pool_and_another_size_does_not() {
        let n = sized(3);
        let a = Threads::new(Some(n)).unwrap();
        let b = Threads::new(Some(n)).unwrap();
        assert!(std::ptr::eq(a.pool().unwrap(), b.pool().unwrap()));
        assert_eq!(a.pool().unwrap().current_num_threads(), n);
        assert_eq!(a.asked(), Some(n));
        let c = Threads::new(Some(sized(2))).unwrap();
        assert!(!std::ptr::eq(a.pool().unwrap(), c.pool().unwrap()));
        assert_eq!(Threads::new(None).unwrap().asked(), None);
        assert_eq!(thread_pool_size(Some(n)).unwrap(), n);
    }

    #[test]
    fn a_pool_goes_with_its_last_holder_and_is_built_again() {
        let n = sized(5);
        let a = Threads::new(Some(n)).unwrap();
        let b = a.clone();
        assert!(Threads::registry_holds(n));
        drop(a);
        assert!(Threads::registry_holds(n), "a clone still holds it");
        drop(b);
        assert!(!Threads::registry_holds(n), "the last holder took it");
        let c = Threads::new(Some(n)).unwrap();
        assert!(Threads::registry_holds(n));
        assert_eq!(c.pool().unwrap().current_num_threads(), n);
    }

    #[test]
    fn the_default_size_asked_by_count_is_the_default_pool() {
        let default = pool().unwrap();
        let n = default.current_num_threads();
        let t = Threads::new(Some(n)).unwrap();
        assert!(std::ptr::eq(t.pool().unwrap(), default));
    }

    #[test]
    fn one_thread_runs_every_task_on_its_one_thread() {
        use rayon::prelude::*;
        let t = Threads::new(Some(1)).unwrap();
        let names: Vec<String> = t
            .install(|| {
                Ok((0..256)
                    .into_par_iter()
                    .map(|_| std::thread::current().name().unwrap_or("").to_string())
                    .collect())
            })
            .unwrap();
        let distinct: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(distinct.len(), 1, "{distinct:?}");
        let name = distinct.into_iter().next().unwrap();
        assert!(name.starts_with("polars-online-"), "{name}");
    }

    #[test]
    fn a_count_the_system_will_not_start_names_the_argument() {
        let refused = || -> Result<ThreadPool, std::io::Error> {
            Err(std::io::Error::from(std::io::ErrorKind::WouldBlock))
        };
        let err = spawned_for_argument(100_000, refused()).unwrap_err();
        assert!(
            err.starts_with("threads=100000 asks for 100000 threads"),
            "{err}"
        );
        assert!(err.contains("give a smaller number"), "{err}");
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build();
        assert_eq!(
            spawned_for_argument(1, pool).unwrap().current_num_threads(),
            1
        );
    }
}
